"""Procedural primitive room generation with GPU BFS solvability checking.

Generates house-like rooms from primitive shapes (walls, furniture, clutter)
at each episode reset. Uses parallel BFS via morphological dilation to
guarantee a solvable path between robot spawn and goal.

Architecture:
    - 44 objects per env in a single RigidObjectCollectionCfg
    - 20 wall segments, 8 furniture pieces, 16 clutter items
    - GPU BFS on 80x80 occupancy grids with robot-radius inflation
    - Spawn points extracted from BFS reachable mask
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollectionCfg, RigidObjectCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ===========================================================================
# Constants
# ===========================================================================

NUM_WALLS = 20       # 8 long + 8 med + 4 short
NUM_FURNITURE = 8    # 2 table + 2 shelf + 2 cabinet + 2 couch
NUM_CLUTTER = 16     # 4 box + 2 cyl + 2 flat + 2 sphere + 2 cone + 2 capsule + 2 tall_cyl
NUM_OBJECTS = NUM_WALLS + NUM_FURNITURE + NUM_CLUTTER  # 44

# Slot index ranges (into the 44-object palette)
WALL_LONG_SLOTS = list(range(0, 8))
WALL_MED_SLOTS = list(range(8, 16))
WALL_SHORT_SLOTS = list(range(16, 20))
WALL_SLOTS = list(range(0, 20))
FURNITURE_SLOTS = list(range(20, 28))
CLUTTER_SLOTS = list(range(28, 44))

# Must track the build order in ``build_proc_room_collection_cfg`` and the
# ``OBJECT_SIZES`` rows.
FURNITURE_SHELF_SLOTS = list(range(22, 24))
FURNITURE_CABINET_SLOTS = list(range(24, 26))
CLUTTER_TALL_CYL_SLOTS = list(range(42, 44))

# Object sizes (X, Y, Z) for AABB computation
# Must match the spawner configs in build_proc_room_collection_cfg().
# Occupancy and the geometric proximity reward read only [:, :2] (XY), so the
# Z entries here are not load-bearing: under enrichment the wall height is
# authoritative from ``generate_proc_room``'s ``wall_height`` argument and the
# shelf / cabinet / tall-cylinder heights from its ``tall_object_heights`` map
# (pose z = height / 2 in both cases) — these Z literals are the open-top default.
OBJECT_SIZES = torch.tensor([
    # Walls - long (8)
    *[(2.0, 0.15, 1.0)] * 8,
    # Walls - med (8)
    *[(1.0, 0.15, 1.0)] * 8,
    # Walls - short (4)
    *[(0.5, 0.15, 1.0)] * 4,
    # Furniture - tables (2)
    *[(0.8, 0.6, 0.4)] * 2,
    # Furniture - shelves (2)
    *[(1.2, 0.3, 0.8)] * 2,
    # Furniture - cabinets (2)
    *[(0.5, 0.5, 0.6)] * 2,
    # Furniture - couches (2)
    *[(1.4, 0.6, 0.35)] * 2,
    # Clutter - boxes (4)
    *[(0.3, 0.3, 0.3)] * 4,
    # Clutter - cylinders (2) - bounding box of r=0.15, h=0.4
    *[(0.3, 0.3, 0.4)] * 2,
    # Clutter - flat boxes (2)
    *[(0.4, 0.4, 0.15)] * 2,
    # Clutter - spheres (2) - bounding box of r=0.15
    *[(0.3, 0.3, 0.3)] * 2,
    # Clutter - cones (2) - bounding box of r=0.12, h=0.35
    *[(0.24, 0.24, 0.35)] * 2,
    # Clutter - capsules (2) - bounding box of r=0.1, h=0.4 (total h=0.6)
    *[(0.2, 0.2, 0.6)] * 2,
    # Clutter - tall cylinders (2) - bounding box of r=0.1, h=0.7
    *[(0.2, 0.2, 0.7)] * 2,
], dtype=torch.float32)  # (44, 3)

assert OBJECT_SIZES.shape[0] == NUM_OBJECTS

# Category -> the slots an enriched height map may raise. The palette builds
# these and the generator poses them from the same map, so the rendered geometry
# and the pose z cannot desync.
TALL_OBJECT_SLOTS = {
    "shelf": FURNITURE_SHELF_SLOTS,
    "cabinet": FURNITURE_CABINET_SLOTS,
    "tall_cyl": CLUTTER_TALL_CYL_SLOTS,
}

# BFS grid parameters
GRID_RES = 0.1          # meters per cell
GRID_SIZE = 80           # cells per axis (covers 8m x 8m)
# Robot footprint modeled as a circle for configuration-space inflation.
# 0.28 m is the chassis circumscribing radius (half the 0.432x0.360 footprint
# diagonal), so a disc of this radius covers the holonomic chassis at any yaw.
ROBOT_HALF_WIDTH = 0.28  # meters (chassis half-diagonal; rotation-invariant)
INFLATION_CELLS = math.ceil(ROBOT_HALF_WIDTH / GRID_RES)  # 3 -> 0.3 m radius
INFLATION_KERNEL = 2 * INFLATION_CELLS + 1  # 7 (disc structuring-element size)
MIN_REACHABLE_CELLS = 100
NUM_SPAWN_POINTS = 200   # per env

# Park position for inactive objects
PARK_POS = torch.tensor([100.0, 100.0, -10.0])
_PARK_INIT_STATE = RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, -10.0))


# ===========================================================================
# Placement configuration
# ===========================================================================

# The slot orders the default path uses, as data. ``FURNITURE_SLOTS[:k]``
# iterated in order is ``FURNITURE_SLOTS[i] for i in range(k)``, and walking the
# rank for the first group with an active member reproduces the two-tier
# "clutter first, then furniture, highest index first" retry ladder.
_VANILLA_FURNITURE_SEQUENCE = tuple(FURNITURE_SLOTS)
_VANILLA_CLUTTER_SEQUENCE = tuple(CLUTTER_SLOTS)
_VANILLA_PARK_RANK = tuple(
    (slot,) for slot in list(reversed(CLUTTER_SLOTS)) + list(reversed(FURNITURE_SLOTS))
)

# A mid-room object must leave the BFS seed cell and the 3x3 ring the wavefront
# first expands into free, or the flood starts inside an obstacle. The disc
# covering that ring is 2 cells, and the footprint is dilated by the robot
# radius before BFS reads it.
BFS_SEED_CLEARANCE = INFLATION_CELLS * GRID_RES + 2 * GRID_RES  # 0.5 m


def column_protected_park_rank(
    column_slots: tuple[int, ...] | list[int] = tuple(CLUTTER_TALL_CYL_SLOTS),
) -> tuple[tuple[int, ...], ...]:
    """The vanilla park rank with ``column_slots`` moved last within clutter.

    Protection is a rank position, never an exemption: a protected slot stays
    parkable and only moves behind its own category, so the ladder's terminal
    state is unchanged and it can never strip a furniture piece to spare a
    column.
    """
    protected = set(column_slots)
    clutter = [s for s in reversed(CLUTTER_SLOTS) if s not in protected]
    clutter += [s for s in reversed(CLUTTER_SLOTS) if s in protected]
    return tuple((slot,) for slot in clutter + list(reversed(FURNITURE_SLOTS)))


@dataclass
class PlacementCfg:
    """Placement-order, park-order and mid-room-column levers.

    An all-default instance reproduces the vanilla placement and park order, so
    every field is a deliberate departure from it. ``None`` sequences resolve to
    the module's vanilla order.

    Not frozen: Isaac Lab's ``configclass`` post-init walks an event term's
    params and re-assigns every nested dataclass field, which a frozen instance
    rejects. Fields are validated at construction.

    Attributes:
        furniture_sequence: Slot order the furniture phase fills; the level's
            furniture budget takes a prefix of it.
        clutter_sequence: Slot order the clutter phase fills; the level's clutter
            budget takes a prefix of it.
        park_rank: Total order over slot *groups* the retry ladder walks, parking
            the first group holding an active member. Groups are parked
            atomically. Perimeter wall slots must stay out of the rank — parking
            one opens a floor-to-ceiling slit the exterior clip does not see.
        column_prob: Per-episode probability the mid-room column phase runs. The
            phase is additive: it does not consume the level's clutter budget.
        column_count: Columns the phase places when it runs.
        column_slots: Slots the column phase poses, and which the clutter phase
            gives up for the episodes the phase runs.
        column_seed_clearance: Minimum distance from env-local origin to a
            column's footprint.
        column_radius_max: Upper bound of the column centre-radius sampler; the
            room's own inset bounds it further per env.
        relocate_blocked_bfs_seed: Move a grid-blocked solvability seed to the
            nearest free cell before the flood. ``_gpu_bfs`` marks its seed
            reachable whether or not the cell is free, so without this a blocked
            centre seeds reachability from inside an obstacle.
    """

    furniture_sequence: tuple[int, ...] | None = None
    clutter_sequence: tuple[int, ...] | None = None
    park_rank: tuple[tuple[int, ...], ...] | None = None
    column_prob: float = 0.0
    column_count: int = 0
    column_slots: tuple[int, ...] = tuple(CLUTTER_TALL_CYL_SLOTS)
    column_seed_clearance: float = BFS_SEED_CLEARANCE
    column_radius_max: float = 2.5
    relocate_blocked_bfs_seed: bool = False

    def __post_init__(self) -> None:
        _validate_sequence(self.furniture_sequence, FURNITURE_SLOTS, "furniture_sequence")
        _validate_sequence(self.clutter_sequence, CLUTTER_SLOTS, "clutter_sequence")
        _validate_park_rank(self.park_rank)
        if not 0.0 <= self.column_prob <= 1.0:
            raise ValueError(f"column_prob must be in [0, 1], got {self.column_prob}")
        if not 0 <= self.column_count <= len(self.column_slots):
            raise ValueError(
                f"column_count must be in [0, {len(self.column_slots)}], "
                f"got {self.column_count}"
            )
        _validate_sequence(self.column_slots, CLUTTER_SLOTS, "column_slots")
        if self.column_seed_clearance < BFS_SEED_CLEARANCE:
            raise ValueError(
                f"column_seed_clearance {self.column_seed_clearance} is below the "
                f"{BFS_SEED_CLEARANCE} m that keeps the BFS seed ring free"
            )
        if self.column_radius_max <= self.column_seed_clearance:
            raise ValueError(
                f"column_radius_max {self.column_radius_max} leaves no admissible "
                f"band above the seed clearance {self.column_seed_clearance}"
            )


def _validate_sequence(seq, allowed, name: str) -> None:
    if seq is None:
        return
    if len(set(seq)) != len(seq):
        raise ValueError(f"{name} repeats a slot: {seq}")
    stray = sorted(set(seq) - set(allowed))
    if stray:
        raise ValueError(f"{name} holds slots outside its category: {stray}")


def _validate_park_rank(rank) -> None:
    if rank is None:
        return
    seen: set[int] = set()
    for group in rank:
        if not group:
            raise ValueError("park_rank holds an empty group")
        overlap = sorted(seen & set(group))
        if overlap:
            raise ValueError(f"park_rank groups are not disjoint, repeated: {overlap}")
        seen |= set(group)
    walls = sorted(seen & set(WALL_SLOTS))
    if walls:
        raise ValueError(f"park_rank holds perimeter wall slots: {walls}")
    missing = sorted((set(FURNITURE_SLOTS) | set(CLUTTER_SLOTS)) - seen)
    if missing:
        raise ValueError(
            f"park_rank does not cover every placeable slot, missing: {missing}"
        )


# ===========================================================================
# Palette builder (for RigidObjectCollectionCfg)
# ===========================================================================

def _make_kinematic_cuboid(size: tuple, color: tuple) -> RigidObjectCfg:
    """Create a kinematic cuboid config."""
    return RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=_PARK_INIT_STATE,
    )


def _make_kinematic_cylinder(radius: float, height: float, color: tuple) -> RigidObjectCfg:
    """Create a kinematic cylinder config."""
    return RigidObjectCfg(
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=_PARK_INIT_STATE,
    )


def _make_kinematic_sphere(radius: float, color: tuple) -> RigidObjectCfg:
    """Create a kinematic sphere config."""
    return RigidObjectCfg(
        spawn=sim_utils.SphereCfg(
            radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=_PARK_INIT_STATE,
    )


def _make_kinematic_cone(radius: float, height: float, color: tuple) -> RigidObjectCfg:
    """Create a kinematic cone config."""
    return RigidObjectCfg(
        spawn=sim_utils.ConeCfg(
            radius=radius,
            height=height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=_PARK_INIT_STATE,
    )


def _make_kinematic_capsule(radius: float, height: float, color: tuple) -> RigidObjectCfg:
    """Create a kinematic capsule config."""
    return RigidObjectCfg(
        spawn=sim_utils.CapsuleCfg(
            radius=radius,
            height=height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=_PARK_INIT_STATE,
    )


def build_proc_room_collection_cfg(
    wall_height: float = 1.0,
    tall_object_heights: dict[str, float] | None = None,
) -> dict[str, RigidObjectCfg]:
    """Build the 44-object palette as a dict for RigidObjectCollectionCfg.

    Each object gets ``prim_path="{ENV_REGEX_NS}/<Name>"`` so Isaac Lab's
    ``InteractiveScene`` can resolve the per-env regex at scene build time.

    Args:
        wall_height: Vertical extent of the wall cuboids in meters. The default
            reproduces the open-top 1.0 m rooms; enriched depth scenes raise it
            to enclose the camera's vertical field. The pose-z the generator
            writes must track this (``wall_height / 2``); see
            ``generate_proc_room``'s ``wall_height`` argument.
        tall_object_heights: Optional ``{category: height}`` override for the
            ``TALL_OBJECT_SLOTS`` categories; absent keys keep the open-top
            default. The same map must reach ``generate_proc_room`` so the pose
            z tracks the geometry. ``None`` reproduces the open-top palette.

    Returns:
        Dict mapping object name → RigidObjectCfg.
    """
    heights = tall_object_heights or {}
    shelf_h = heights.get("shelf", 0.8)
    cabinet_h = heights.get("cabinet", 0.6)
    tall_cyl_h = heights.get("tall_cyl", 0.7)

    objects = {}

    # --- Walls ---
    for i in range(8):
        objects[f"wall_long_{i}"] = _make_kinematic_cuboid((2.0, 0.15, wall_height), (0.75, 0.75, 0.75))
    for i in range(8):
        objects[f"wall_med_{i}"] = _make_kinematic_cuboid((1.0, 0.15, wall_height), (0.70, 0.70, 0.72))
    for i in range(4):
        objects[f"wall_short_{i}"] = _make_kinematic_cuboid((0.5, 0.15, wall_height), (0.65, 0.65, 0.68))

    # --- Furniture ---
    for i in range(2):
        objects[f"furn_table_{i}"] = _make_kinematic_cuboid((0.8, 0.6, 0.4), (0.55, 0.35, 0.15))
    for i in range(2):
        objects[f"furn_shelf_{i}"] = _make_kinematic_cuboid((1.2, 0.3, shelf_h), (0.50, 0.30, 0.12))
    for i in range(2):
        objects[f"furn_cabinet_{i}"] = _make_kinematic_cuboid((0.5, 0.5, cabinet_h), (0.45, 0.28, 0.10))
    for i in range(2):
        objects[f"furn_couch_{i}"] = _make_kinematic_cuboid((1.4, 0.6, 0.35), (0.40, 0.25, 0.55))

    # --- Clutter ---
    for i in range(4):
        objects[f"clutter_box_{i}"] = _make_kinematic_cuboid((0.3, 0.3, 0.3), (0.6, 0.2, 0.2))
    for i in range(2):
        objects[f"clutter_cyl_{i}"] = _make_kinematic_cylinder(0.15, 0.4, (0.2, 0.5, 0.2))
    for i in range(2):
        objects[f"clutter_flat_{i}"] = _make_kinematic_cuboid((0.4, 0.4, 0.15), (0.5, 0.5, 0.2))
    for i in range(2):
        objects[f"clutter_sphere_{i}"] = _make_kinematic_sphere(0.15, (0.7, 0.4, 0.1))
    for i in range(2):
        objects[f"clutter_cone_{i}"] = _make_kinematic_cone(0.12, 0.35, (0.8, 0.5, 0.1))
    for i in range(2):
        objects[f"clutter_capsule_{i}"] = _make_kinematic_capsule(0.1, 0.4, (0.3, 0.3, 0.6))
    for i in range(2):
        objects[f"clutter_tall_cyl_{i}"] = _make_kinematic_cylinder(0.1, tall_cyl_h, (0.4, 0.6, 0.3))

    assert len(objects) == NUM_OBJECTS

    # Set prim_path on each object using its dict key as the USD leaf name
    for name, cfg in objects.items():
        cfg.prim_path = f"{{ENV_REGEX_NS}}/{name}"

    return objects


# ===========================================================================
# GPU BFS
# ===========================================================================

def _build_occupancy_grid(
    poses: torch.Tensor,
    active_mask: torch.Tensor,
    sizes: torch.Tensor,
    room_w: torch.Tensor,
    room_h: torch.Tensor,
    device: torch.device,
    has_room_walls: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rasterize object AABBs onto an occupancy grid.

    Args:
        poses: (B, N, 7) object poses in env-local frame [x, y, z, qx, qy, qz, qw] (XYZW).
        active_mask: (B, N) bool — True if object is placed (not parked).
        sizes: (N, 3) object dimensions (X, Y, Z) — same for all envs.
        room_w: (B,) room width.
        room_h: (B,) room height.
        device: torch device.
        has_room_walls: (B,) long — 1 if env has enclosing walls, 0 for open field.
            When None, all envs are treated as having room walls.

    Returns:
        occupancy: (B, GRID_SIZE, GRID_SIZE) float — 1.0 = occupied, 0.0 = free.
    """
    B = poses.shape[0]
    N = poses.shape[1]
    G = GRID_SIZE

    occupancy = torch.zeros(B, G, G, device=device)

    # Grid origin: bottom-left corner of the bounding area
    # Center the grid on the room (rooms are centered at origin)
    grid_origin_x = -G * GRID_RES / 2.0  # -4.0 for 80 cells
    grid_origin_y = -G * GRID_RES / 2.0

    # Extract XY positions and yaw from poses (XYZW convention)
    cx = poses[:, :, 0]  # (B, N)
    cy = poses[:, :, 1]  # (B, N)
    qz = poses[:, :, 5]  # (B, N) — XYZW: index 5 is qz
    qw = poses[:, :, 6]  # (B, N) — XYZW: index 6 is qw
    yaw = 2.0 * torch.atan2(qz, qw)  # (B, N)

    # Object half-sizes
    wx = sizes[:, 0].unsqueeze(0).expand(B, -1) / 2.0  # (B, N)
    wy = sizes[:, 1].unsqueeze(0).expand(B, -1) / 2.0  # (B, N)

    # Rotated AABB half-extents
    cos_yaw = torch.abs(torch.cos(yaw))
    sin_yaw = torch.abs(torch.sin(yaw))
    hx = wx * cos_yaw + wy * sin_yaw  # (B, N)
    hy = wx * sin_yaw + wy * cos_yaw  # (B, N)

    # Convert to grid cell ranges
    x_min = ((cx - hx - grid_origin_x) / GRID_RES).long().clamp(0, G - 1)
    x_max = ((cx + hx - grid_origin_x) / GRID_RES).long().clamp(0, G - 1)
    y_min = ((cy - hy - grid_origin_y) / GRID_RES).long().clamp(0, G - 1)
    y_max = ((cy + hy - grid_origin_y) / GRID_RES).long().clamp(0, G - 1)

    # Rasterize each object (loop over objects — 44 is small)
    for j in range(N):
        mask = active_mask[:, j]  # (B,)
        if not mask.any():
            continue
        batch_idx = torch.where(mask)[0]
        for b in batch_idx:
            bi = b.item()
            occupancy[bi, x_min[bi, j]:x_max[bi, j] + 1, y_min[bi, j]:y_max[bi, j] + 1] = 1.0

    # Mark cells outside the room bounding box as occupied.
    # Without this, BFS floods through doorway gaps into unbounded exterior,
    # causing robot/goals to spawn outside the room.
    # Skipped for open-field envs (no enclosing walls).
    grid_xs = torch.arange(G, device=device).float() * GRID_RES + grid_origin_x + GRID_RES / 2.0
    grid_ys = torch.arange(G, device=device).float() * GRID_RES + grid_origin_y + GRID_RES / 2.0

    for b in range(B):
        if has_room_walls is not None and has_room_walls[b].item() == 0:
            continue  # Open field — no exterior clipping
        hw = room_w[b] / 2.0
        hh = room_h[b] / 2.0
        outside_x = (grid_xs < -hw) | (grid_xs > hw)  # (G,)
        outside_y = (grid_ys < -hh) | (grid_ys > hh)  # (G,)
        # Any cell where row OR column is outside the room is exterior
        exterior = outside_x.unsqueeze(1) | outside_y.unsqueeze(0)  # (G, G)
        occupancy[b][exterior] = 1.0

    return occupancy


# Disc (Euclidean) structuring element for obstacle inflation, cached per
# device. A square max_pool kernel dilates obstacles in Chebyshev distance,
# which over-inflates corners (a box obstacle grows square corners reaching
# ~0.42 m diagonally for a 3-cell radius). A disc gives the true
# configuration-space obstacle for a circular footprint — straight faces push
# out by the radius, box corners round off — so the robot can legitimately
# round obstacle corners instead of being pushed back a full diagonal.
_DISC_KERNEL_CACHE: dict[torch.device, torch.Tensor] = {}


def _disc_kernel(device: torch.device) -> torch.Tensor:
    kern = _DISC_KERNEL_CACHE.get(device)
    if kern is None:
        offs = torch.arange(INFLATION_KERNEL, device=device) - INFLATION_CELLS
        rr, cc = torch.meshgrid(offs, offs, indexing="ij")
        kern = ((rr**2 + cc**2) <= INFLATION_CELLS**2).float().view(
            1, 1, INFLATION_KERNEL, INFLATION_KERNEL
        )
        _DISC_KERNEL_CACHE[device] = kern
    return kern


def _inflate_obstacles(occupancy: torch.Tensor) -> torch.Tensor:
    """Dilate the occupancy grid by the robot radius with a DISC structuring
    element — the Minkowski sum of obstacles with the robot's footprint circle.

    A cell is blocked iff any obstacle cell lies within ``INFLATION_CELLS`` of
    it in Euclidean distance (conv2d with the 0/1 disc kernel counts obstacle
    cells inside the disc; >0 means blocked).

    Args:
        occupancy: (B, Gx, Gy) float — 1.0 = occupied.

    Returns:
        free_space: (B, Gx, Gy) bool — True = passable after inflation.
    """
    kern = _disc_kernel(occupancy.device)
    inflated = F.conv2d(
        occupancy.unsqueeze(1), kern, padding=INFLATION_CELLS
    ).squeeze(1)
    return inflated < 0.5


def _erode_reachable(reachable: torch.Tensor, cells: int) -> torch.Tensor:
    """Shrink a boolean reachable mask inward by ``cells`` (extra clearance).

    A cell survives iff every cell within Chebyshev distance ``cells`` is
    reachable — i.e. it clears obstacles by an extra ``cells * GRID_RES`` beyond
    the robot-radius inflation already baked into ``reachable``. Used to draw
    robot spawn poses from a more-cleared interior than the goal/subgoal pool.

    Args:
        reachable: (B, Gx, Gy) bool — BFS reachability mask.
        cells: Extra erosion radius in grid cells (0 returns the input).

    Returns:
        eroded: (B, Gx, Gy) bool.
    """
    if cells <= 0:
        return reachable
    inv = (~reachable).float().unsqueeze(1)
    eroded = F.max_pool2d(inv, kernel_size=2 * cells + 1, stride=1, padding=cells)
    return eroded.squeeze(1) < 0.5


def _gpu_bfs(free_space: torch.Tensor, start_cells: torch.Tensor, max_iterations: int = 200) -> torch.Tensor:
    """Parallel BFS via iterative morphological dilation.

    Args:
        free_space: (B, Gx, Gy) bool — True = passable.
        start_cells: (B, 2) int — (row, col) of BFS seed.
        max_iterations: Max wavefront expansion steps.

    Returns:
        reachable: (B, Gx, Gy) bool — True = reachable from start.
    """
    B, Gx, Gy = free_space.shape
    device = free_space.device

    reachable = torch.zeros(B, 1, Gx, Gy, device=device)
    # Clamp start cells to valid range
    r = start_cells[:, 0].clamp(0, Gx - 1)
    c = start_cells[:, 1].clamp(0, Gy - 1)
    reachable[torch.arange(B, device=device), 0, r, c] = 1.0
    free = free_space.unsqueeze(1).float()

    for _ in range(max_iterations):
        expanded = F.max_pool2d(reachable, kernel_size=3, stride=1, padding=1)
        new_cells = expanded * free * (1.0 - reachable)
        if new_cells.sum() == 0:
            break
        reachable = (reachable + new_cells).clamp_(max=1.0)

    return reachable.squeeze(1) > 0.5


def _relocate_blocked_seeds(
    free_space: torch.Tensor, start_cells: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move each grid-blocked BFS seed to its nearest free cell.

    ``_gpu_bfs`` marks its seed reachable whether the cell is free or not, so a
    blocked seed floods reachability outward from inside an obstacle. Ties break
    on flattened cell order, so the correction consumes no randomness.

    Args:
        free_space: (B, Gx, Gy) bool — True = passable after inflation.
        start_cells: (B, 2) int — (row, col) of the requested seeds.

    Returns:
        cells: (B, 2) int — seeds, corrected where they were blocked.
        moved: (B,) bool — True where the seed was relocated.
    """
    B, Gx, Gy = free_space.shape
    device = free_space.device
    rows = start_cells[:, 0]
    cols = start_cells[:, 1]
    blocked = ~free_space[torch.arange(B, device=device), rows, cols]
    moved = blocked & free_space.view(B, -1).any(dim=1)
    if not moved.any():
        return start_cells, moved

    idx = torch.where(moved)[0]
    dr = torch.arange(Gx, device=device).view(1, Gx, 1) - rows[idx].view(-1, 1, 1)
    dc = torch.arange(Gy, device=device).view(1, 1, Gy) - cols[idx].view(-1, 1, 1)
    dist2 = dr * dr + dc * dc
    unreachable = Gx * Gx + Gy * Gy + 1
    dist2 = torch.where(free_space[idx], dist2, torch.full_like(dist2, unreachable))
    flat = dist2.view(len(idx), -1).argmin(dim=1)

    cells = start_cells.clone()
    cells[idx, 0] = flat // Gy
    cells[idx, 1] = flat % Gy
    return cells, moved


def _xy_to_grid(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert env-local XY coordinates to grid cell indices.

    Args:
        x, y: (B,) or scalar coordinates in meters (env-local frame).

    Returns:
        row, col: Grid cell indices (long tensors).
    """
    origin = -GRID_SIZE * GRID_RES / 2.0
    row = ((x - origin) / GRID_RES).long().clamp(0, GRID_SIZE - 1)
    col = ((y - origin) / GRID_RES).long().clamp(0, GRID_SIZE - 1)
    return row, col


def _grid_to_xy(row: torch.Tensor, col: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert grid cell indices to env-local XY coordinates (cell centers).

    Args:
        row, col: (B, K) or (K,) grid indices.

    Returns:
        x, y: Coordinates in meters (env-local frame).
    """
    origin = -GRID_SIZE * GRID_RES / 2.0
    x = row.float() * GRID_RES + origin + GRID_RES / 2.0
    y = col.float() * GRID_RES + origin + GRID_RES / 2.0
    return x, y


def _extract_spawn_points(
    reachable: torch.Tensor,
    num_points: int = NUM_SPAWN_POINTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random reachable cells as spawn/goal candidates.

    Args:
        reachable: (B, Gx, Gy) bool — reachability mask from BFS.
        num_points: Number of points to sample per env.

    Returns:
        spawn_xy: (B, num_points, 2) float — XY positions in env-local frame.
        spawn_count: (B,) int — actual number of valid points per env
            (may be < num_points if fewer than num_points cells are reachable).
    """
    B, Gx, Gy = reachable.shape
    device = reachable.device

    spawn_xy = torch.zeros(B, num_points, 2, device=device)
    spawn_count = torch.zeros(B, dtype=torch.long, device=device)

    for b in range(B):
        reachable_cells = reachable[b].nonzero()  # (K, 2)
        K = reachable_cells.shape[0]
        if K == 0:
            continue
        n = min(K, num_points)
        indices = torch.randperm(K, device=device)[:n]
        selected = reachable_cells[indices]  # (n, 2)
        x, y = _grid_to_xy(selected[:, 0], selected[:, 1])
        spawn_xy[b, :n, 0] = x
        spawn_xy[b, :n, 1] = y
        spawn_count[b] = n

    return spawn_xy, spawn_count


# ===========================================================================
# Room Generation
# ===========================================================================

def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles to quaternions (x, y, z, w) — XYZW (Isaac Lab 3.0).

    Args:
        yaw: (...) tensor of yaw angles in radians.

    Returns:
        quat: (..., 4) tensor of quaternions in XYZW format.
    """
    half = yaw / 2.0
    zeros = torch.zeros_like(yaw)
    # XYZW: [x=0, y=0, z=sin(yaw/2), w=cos(yaw/2)]
    quat = torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)
    return quat


def _pack_wall_segments(
    total_length: float,
    long_budget: list[int],
    med_budget: list[int],
    short_budget: list[int],
) -> list[tuple[int, float]]:
    """Greedy-pack wall segments for a given length.

    Returns list of (slot_index, segment_length) pairs.
    Consumes from budget lists in-place.
    """
    segments = []
    remaining = total_length
    seg_sizes = [(2.0, long_budget), (1.0, med_budget), (0.5, short_budget)]

    for seg_len, budget in seg_sizes:
        while remaining >= seg_len - 0.01 and budget:
            slot = budget.pop(0)
            segments.append((slot, seg_len))
            remaining -= seg_len

    return segments


def _pose_ceiling_slab(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    entity_name: str,
    p_ceil: float,
    height_range: tuple[float, float],
    device: torch.device,
) -> None:
    """Teleport the standalone ceiling slab per env (or park it).

    The slab is a scene entity *outside* the room-primitive collection, so it
    never enters the occupancy grid, the BFS retry ladder, or the geometric
    ``obstacle_proximity`` active mask (which is keyed on the collection). With
    probability ``p_ceil`` the env is enclosed — the slab drops to a per-episode
    height in ``height_range``; otherwise it parks below the floor (open-top),
    so the far-clamp top-band stays a scene-class mixture rather than always
    enclosed.
    """
    B = len(env_ids)
    ceiling = env.scene[entity_name]
    env_origins = env.scene.env_origins[env_ids]  # (B, 3)

    has_ceil = torch.rand(B, device=device) < p_ceil
    ceil_h = torch.rand(B, device=device) * (height_range[1] - height_range[0]) + height_range[0]
    z_local = torch.where(has_ceil, ceil_h, torch.full_like(ceil_h, PARK_POS[2].item()))

    root_pose = torch.zeros(B, 7, device=device)
    root_pose[:, 0] = env_origins[:, 0]
    root_pose[:, 1] = env_origins[:, 1]
    root_pose[:, 2] = env_origins[:, 2] + z_local
    root_pose[:, 6] = 1.0  # qw (identity) — XYZW
    ceiling.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)


def generate_proc_room(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    collection_name: str = "room_primitives",
    max_internal_walls: int = 0,
    max_furniture: int = 0,
    max_clutter: int = 0,
    wall_height: float = 1.0,
    door_width_max: float = 1.2,
    span_max: float = 7.0,
    max_span_sum: float | None = None,
    clutter_wall_bias_prob: float = 0.0,
    robot_spawn_inflation_cells: int = 0,
    ceiling_entity_name: str | None = None,
    p_ceil: float = 0.0,
    ceiling_height_range: tuple[float, float] = (2.2, 2.9),
    tall_object_heights: dict[str, float] | None = None,
    placement: PlacementCfg | None = None,
    health_sink: dict | None = None,
) -> None:
    """Generate procedural rooms for specified environments.

    This is the main EventTerm function. Called at episode reset BEFORE
    robot reset, so spawn points are available for robot placement.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to regenerate rooms for.
        collection_name: Scene entity name for the RigidObjectCollectionCfg.
        max_internal_walls: Max internal wall segments (from curriculum).
        max_furniture: Max furniture pieces to place (from curriculum).
        max_clutter: Max clutter items to place (from curriculum).
        wall_height: Wall vertical extent; the wall pose-z is ``wall_height / 2``.
            Must match the ``wall_height`` the palette was built with.
        door_width_max: Upper bound of the doorway width sampler ``U[0.8, x]``.
        span_max: Upper bound of the per-axis room span sampler ``U[4.0, x]``.
        max_span_sum: When set, caps ``room_w + room_h`` (proportional shrink) so
            the wall-segment budget can seal the perimeter under tall walls.
        clutter_wall_bias_prob: Per-episode probability of the perimeter-biased
            clutter placement law (large objects near walls, floor center open)
            instead of the uniform interior scatter.
        robot_spawn_inflation_cells: Extra erosion (in grid cells) for a
            robot-spawn-only pool, giving the robot more standoff than the
            goal/subgoal endpoints. 0 keeps the single shared pool.
        ceiling_entity_name: Scene entity name of the standalone ceiling slab
            (outside the collection). When set, the slab is posed per env.
        p_ceil: Per-episode probability the ceiling slab is present.
        ceiling_height_range: Per-episode ceiling height sampler ``U[lo, hi]``.
        tall_object_heights: Optional ``{category: height}`` override for the
            ``TALL_OBJECT_SLOTS`` categories; their pose z becomes ``height / 2``
            instead of the ``OBJECT_SIZES`` default. Must match the map the
            palette was built with. ``None`` keeps every slot at its
            ``OBJECT_SIZES`` height.
        placement: Placement-order, park-order and mid-room-column levers.
            ``None`` keeps the vanilla slot sequences and retry-ladder order.
        health_sink: When given, a dict the generator fills with per-call
            placement, park, BFS and seed-relocation counters. Production passes
            nothing.

    All enrichment arguments default to the open-top, single-pool, uniform-scatter
    behavior, so the NOCAM / vanilla-depth generator path is unchanged.
    """
    B = len(env_ids)
    if B == 0:
        return

    device = env.device
    sizes = OBJECT_SIZES.to(device)
    wall_z = wall_height / 2.0

    # Resolve the placement levers once; every one of them is a plain module
    # constant when ``placement`` is None.
    furn_seq = _VANILLA_FURNITURE_SEQUENCE
    clut_seq = _VANILLA_CLUTTER_SEQUENCE
    park_rank = _VANILLA_PARK_RANK
    column_prob = 0.0
    column_count = 0
    column_slots: tuple[int, ...] = ()
    clut_seq_columns = clut_seq
    relocate_seed = False
    if placement is not None:
        if placement.furniture_sequence is not None:
            furn_seq = placement.furniture_sequence
        if placement.clutter_sequence is not None:
            clut_seq = placement.clutter_sequence
        if placement.park_rank is not None:
            park_rank = placement.park_rank
        relocate_seed = placement.relocate_blocked_bfs_seed
        if placement.column_prob > 0.0 and placement.column_count > 0:
            column_prob = placement.column_prob
            column_count = placement.column_count
            column_slots = placement.column_slots
            # The column phase owns these slots on the episodes it runs, so the
            # clutter budget still fills with ordinary items rather than losing
            # two of them to the lever it is being measured against.
            clut_seq_columns = tuple(s for s in clut_seq if s not in set(column_slots))
            column_r_min = placement.column_seed_clearance + max(
                math.hypot(OBJECT_SIZES[s, 0].item(), OBJECT_SIZES[s, 1].item()) / 2.0
                for s in column_slots
            )
            column_r_max = placement.column_radius_max
    park_groups = [list(group) for group in park_rank]
    n_column_promotions = 0
    ladder_passes = 0
    seed_relocations = 0

    # Floor-standing pose z (base on the floor). Walls are posed from ``wall_z``
    # below, not this vector.
    object_z = sizes[:, 2] / 2.0  # (N,)
    if tall_object_heights is not None:
        for key, slots in TALL_OBJECT_SLOTS.items():
            if key in tall_object_heights:
                object_z[list(slots)] = tall_object_heights[key] / 2.0

    # Read curriculum difficulty if available
    if hasattr(env, "_proc_room_difficulty"):
        difficulty = env._proc_room_difficulty[env_ids]
        # Curriculum levels override the max params per-env
        # Columns: (internal_walls, furniture, clutter, has_room_walls)
        # Levels 0-1: open field (no enclosing walls) for easy early learning
        # Levels 2+: enclosed rooms with increasing complexity
        level_table = torch.tensor([
            [0, 0, 0, 0],   # level 0: open field — just goals
            [0, 2, 4, 0],   # level 1: scattered obstacles on open ground
            [0, 0, 0, 1],   # level 2: empty rectangular room
            [0, 2, 0, 1],   # level 3: room + sparse furniture
            [0, 4, 4, 1],   # level 4: room + moderate obstacles
            [1, 4, 8, 1],   # level 5: room + internal wall + clutter
            [1, 6, 12, 1],  # level 6: room + dense clutter
            [2, 8, 16, 1],  # level 7: full complexity
        ], device=device, dtype=torch.long)
        difficulty_clamped = difficulty.clamp(0, 7).long()
        per_env_limits = level_table[difficulty_clamped]  # (B, 4)
        max_iw = per_env_limits[:, 0]
        max_furn = per_env_limits[:, 1]
        max_clut = per_env_limits[:, 2]
        has_room_walls = per_env_limits[:, 3]  # (B,) 0=open field, 1=room
    else:
        max_iw = torch.full((B,), max_internal_walls, device=device, dtype=torch.long)
        max_furn = torch.full((B,), max_furniture, device=device, dtype=torch.long)
        max_clut = torch.full((B,), max_clutter, device=device, dtype=torch.long)
        has_room_walls = torch.ones(B, device=device, dtype=torch.long)  # default: room walls on

    # --- Phase 1: Parameter sampling ---
    span_range = span_max - 4.0
    room_w = torch.rand(B, device=device) * span_range + 4.0  # [4, span_max]
    room_h = torch.rand(B, device=device) * span_range + 4.0  # [4, span_max]
    if max_span_sum is not None:
        # Keep the perimeter within the wall-segment budget: shrink both axes
        # equally when their sum exceeds what the walls can seal (tall enclosed
        # walls turn any pack-gap into a floor-to-ceiling far-clamp slit).
        excess = (room_w + room_h - max_span_sum).clamp(min=0.0)
        room_w = room_w - 0.5 * excess
        room_h = room_h - 0.5 * excess

    # Initialize poses: all parked
    poses = torch.zeros(B, NUM_OBJECTS, 7, device=device)
    poses[:, :, 0] = PARK_POS[0]
    poses[:, :, 1] = PARK_POS[1]
    poses[:, :, 2] = PARK_POS[2]
    poses[:, :, 6] = 1.0  # qw = 1 (identity) — XYZW: [x,y,z,qx,qy,qz,qw]
    active_mask = torch.zeros(B, NUM_OBJECTS, dtype=torch.bool, device=device)

    # --- Phase 2: Wall construction (per-env loop — small B) ---
    for b_idx in range(B):
        w = room_w[b_idx].item()
        h = room_h[b_idx].item()

        # --- Phase 2a: Wall construction (skip for open-field levels) ---
        if has_room_walls[b_idx].item() != 0:
            # Budget tracking (available slot indices)
            long_avail = list(WALL_LONG_SLOTS)
            med_avail = list(WALL_MED_SLOTS)
            short_avail = list(WALL_SHORT_SLOTS)

            # Doorway: pick 1 random side
            door_side = torch.randint(0, 4, (1,), device=device).item()
            door_pos = torch.rand(1, device=device).item()  # 0-1 along wall
            door_width = torch.rand(1, device=device).item() * (door_width_max - 0.8) + 0.8  # [0.8, door_width_max]

            # Build 4 walls: N(top), S(bottom), E(right), W(left)
            # Convention: room spans [-w/2, w/2] x [-h/2, h/2]
            sides = [
                # (start_x, start_y, end_x, end_y, is_horizontal)
                (-w / 2, h / 2, w / 2, h / 2, True),     # N
                (-w / 2, -h / 2, w / 2, -h / 2, True),   # S
                (w / 2, -h / 2, w / 2, h / 2, False),     # E
                (-w / 2, -h / 2, -w / 2, h / 2, False),   # W
            ]

            for side_idx, (sx, sy, ex, ey, is_horiz) in enumerate(sides):
                wall_length = w if is_horiz else h

                if side_idx == door_side:
                    # Split wall around doorway
                    door_center = door_pos * (wall_length - door_width - 0.5) + door_width / 2 + 0.25
                    left_len = door_center - door_width / 2
                    right_len = wall_length - door_center - door_width / 2

                    left_segs = _pack_wall_segments(left_len, long_avail, med_avail, short_avail)
                    right_segs = _pack_wall_segments(right_len, long_avail, med_avail, short_avail)

                    # Place left segments
                    cursor = 0.0
                    for slot, seg_len in left_segs:
                        if is_horiz:
                            px = sx + cursor + seg_len / 2
                            py = sy
                            yaw = 0.0
                        else:
                            px = sx
                            py = sy + cursor + seg_len / 2
                            yaw = math.pi / 2
                        poses[b_idx, slot, 0] = px
                        poses[b_idx, slot, 1] = py
                        poses[b_idx, slot, 2] = wall_z  # wall_height/2
                        q = _yaw_to_quat(torch.tensor(yaw, device=device))
                        poses[b_idx, slot, 3:7] = q
                        active_mask[b_idx, slot] = True
                        cursor += seg_len

                    # Place right segments (start after doorway gap)
                    cursor = door_center + door_width / 2
                    for slot, seg_len in right_segs:
                        if is_horiz:
                            px = sx + cursor + seg_len / 2
                            py = sy
                            yaw = 0.0
                        else:
                            px = sx
                            py = sy + cursor + seg_len / 2
                            yaw = math.pi / 2
                        poses[b_idx, slot, 0] = px
                        poses[b_idx, slot, 1] = py
                        poses[b_idx, slot, 2] = wall_z
                        q = _yaw_to_quat(torch.tensor(yaw, device=device))
                        poses[b_idx, slot, 3:7] = q
                        active_mask[b_idx, slot] = True
                        cursor += seg_len
                else:
                    # Full wall (no doorway)
                    segs = _pack_wall_segments(wall_length, long_avail, med_avail, short_avail)
                    cursor = 0.0
                    for slot, seg_len in segs:
                        if is_horiz:
                            px = sx + cursor + seg_len / 2
                            py = sy
                            yaw = 0.0
                        else:
                            px = sx
                            py = sy + cursor + seg_len / 2
                            yaw = math.pi / 2
                        poses[b_idx, slot, 0] = px
                        poses[b_idx, slot, 1] = py
                        poses[b_idx, slot, 2] = wall_z
                        q = _yaw_to_quat(torch.tensor(yaw, device=device))
                        poses[b_idx, slot, 3:7] = q
                        active_mask[b_idx, slot] = True
                        cursor += seg_len

        # --- Phase 3: Furniture placement ---
        n_furn = max_furn[b_idx].item()
        placed_furn_xy = []
        is_open_field = has_room_walls[b_idx].item() == 0
        for slot in furn_seq[:n_furn]:
            furn_sx = OBJECT_SIZES[slot, 0].item()
            furn_sy = OBJECT_SIZES[slot, 1].item()

            for _ in range(10):
                if is_open_field:
                    # Open field: scatter randomly within the area
                    inset_f = 0.5
                    fx = (torch.rand(1, device=device).item() - 0.5) * (w - 2 * inset_f)
                    fy = (torch.rand(1, device=device).item() - 0.5) * (h - 2 * inset_f)
                    fyaw = torch.rand(1, device=device).item() * 2.0 * math.pi - math.pi
                else:
                    # Room: align against a wall
                    wall_side = torch.randint(0, 4, (1,), device=device).item()
                    if wall_side == 0:  # N wall
                        fx = (torch.rand(1, device=device).item() - 0.5) * (w - furn_sx - 0.5)
                        fy = h / 2 - 0.075 - furn_sy / 2
                        fyaw = 0.0
                    elif wall_side == 1:  # S wall
                        fx = (torch.rand(1, device=device).item() - 0.5) * (w - furn_sx - 0.5)
                        fy = -h / 2 + 0.075 + furn_sy / 2
                        fyaw = math.pi
                    elif wall_side == 2:  # E wall
                        fy = (torch.rand(1, device=device).item() - 0.5) * (h - furn_sx - 0.5)
                        fx = w / 2 - 0.075 - furn_sy / 2
                        fyaw = math.pi / 2
                    else:  # W wall
                        fy = (torch.rand(1, device=device).item() - 0.5) * (h - furn_sx - 0.5)
                        fx = -w / 2 + 0.075 + furn_sy / 2
                        fyaw = -math.pi / 2

                # Check min distance from previously placed furniture
                too_close = False
                for pfx, pfy in placed_furn_xy:
                    if abs(fx - pfx) < 0.5 and abs(fy - pfy) < 0.5:
                        too_close = True
                        break
                if too_close:
                    continue

                poses[b_idx, slot, 0] = fx
                poses[b_idx, slot, 1] = fy
                poses[b_idx, slot, 2] = object_z[slot].item()
                q = _yaw_to_quat(torch.tensor(fyaw, device=device))
                poses[b_idx, slot, 3:7] = q
                active_mask[b_idx, slot] = True
                placed_furn_xy.append((fx, fy))
                break
            # If not placed after 10 attempts, slot stays parked

        # --- Phase 4: Clutter scatter ---
        n_clut = max_clut[b_idx].item()
        placed_clutter_xy = []
        inset = 0.5

        # --- Phase 4a: Mid-room columns (additive, enriched-only) ---
        # Runs before the scatter so the columns get the interior the perimeter
        # bias vacates and the scatter then avoids them. The radial law is
        # bounded below by the BFS seed clearance and above by the room inset,
        # so no candidate can enclose the solvability seed or leave the room.
        env_clut_seq = clut_seq
        if column_prob > 0.0:
            if torch.rand(1, device=device).item() < column_prob:
                env_clut_seq = clut_seq_columns
                n_column_promotions += 1
                r_hi = min(column_r_max, min(w, h) / 2.0 - inset)
                for slot in column_slots[:column_count]:
                    if r_hi <= column_r_min:
                        break
                    for _ in range(10):
                        theta = torch.rand(1, device=device).item() * 2.0 * math.pi - math.pi
                        radius = column_r_min + torch.rand(1, device=device).item() * (
                            r_hi - column_r_min
                        )
                        cx_ = radius * math.cos(theta)
                        cy_ = radius * math.sin(theta)

                        too_close = False
                        for pfx, pfy in placed_furn_xy:
                            if abs(cx_ - pfx) < 0.4 and abs(cy_ - pfy) < 0.4:
                                too_close = True
                                break
                        if too_close:
                            continue
                        for pcx, pcy in placed_clutter_xy:
                            if abs(cx_ - pcx) < 0.3 and abs(cy_ - pcy) < 0.3:
                                too_close = True
                                break
                        if too_close:
                            continue

                        cyaw = torch.rand(1, device=device).item() * 2.0 * math.pi - math.pi
                        poses[b_idx, slot, 0] = cx_
                        poses[b_idx, slot, 1] = cy_
                        poses[b_idx, slot, 2] = object_z[slot].item()
                        q = _yaw_to_quat(torch.tensor(cyaw, device=device))
                        poses[b_idx, slot, 3:7] = q
                        active_mask[b_idx, slot] = True
                        placed_clutter_xy.append((cx_, cy_))
                        break

        # Per-episode placement law: uniform interior scatter (default) or the
        # perimeter-biased mixture (clutter hugs the walls, floor center open —
        # the constraint-solver look). The draw is skipped when disabled so the
        # default RNG stream is unchanged.
        clutter_wall_bias = False
        if clutter_wall_bias_prob > 0.0:
            clutter_wall_bias = torch.rand(1, device=device).item() < clutter_wall_bias_prob

        for slot in env_clut_seq[:n_clut]:
            for _ in range(10):
                if clutter_wall_bias:
                    # Push the item toward a random wall, small inward margin.
                    c_side = torch.randint(0, 4, (1,), device=device).item()
                    c_margin = inset + torch.rand(1, device=device).item() * 0.6  # 0.5-1.1 m from wall
                    if c_side == 0:      # N
                        cx_ = (torch.rand(1, device=device).item() - 0.5) * (w - 2 * inset)
                        cy_ = h / 2 - c_margin
                    elif c_side == 1:    # S
                        cx_ = (torch.rand(1, device=device).item() - 0.5) * (w - 2 * inset)
                        cy_ = -h / 2 + c_margin
                    elif c_side == 2:    # E
                        cx_ = w / 2 - c_margin
                        cy_ = (torch.rand(1, device=device).item() - 0.5) * (h - 2 * inset)
                    else:                # W
                        cx_ = -w / 2 + c_margin
                        cy_ = (torch.rand(1, device=device).item() - 0.5) * (h - 2 * inset)
                else:
                    cx_ = (torch.rand(1, device=device).item() - 0.5) * (w - 2 * inset)
                    cy_ = (torch.rand(1, device=device).item() - 0.5) * (h - 2 * inset)

                # Min distance from furniture
                too_close = False
                for pfx, pfy in placed_furn_xy:
                    if abs(cx_ - pfx) < 0.4 and abs(cy_ - pfy) < 0.4:
                        too_close = True
                        break
                if too_close:
                    continue

                # Min distance from other clutter
                for pcx, pcy in placed_clutter_xy:
                    if abs(cx_ - pcx) < 0.3 and abs(cy_ - pcy) < 0.3:
                        too_close = True
                        break
                if too_close:
                    continue

                cyaw = torch.rand(1, device=device).item() * 2.0 * math.pi - math.pi
                poses[b_idx, slot, 0] = cx_
                poses[b_idx, slot, 1] = cy_
                poses[b_idx, slot, 2] = object_z[slot].item()
                q = _yaw_to_quat(torch.tensor(cyaw, device=device))
                poses[b_idx, slot, 3:7] = q
                active_mask[b_idx, slot] = True
                placed_clutter_xy.append((cx_, cy_))
                break

    # --- Phase 5: BFS solvability check ---
    placed_mask = active_mask.clone() if health_sink is not None else None
    occupancy = _build_occupancy_grid(poses, active_mask, sizes, room_w, room_h, device, has_room_walls)
    free_space = _inflate_obstacles(occupancy)

    # BFS from room center (0, 0)
    center_r, center_c = _xy_to_grid(
        torch.zeros(B, device=device), torch.zeros(B, device=device)
    )
    center_cells = torch.stack([center_r, center_c], dim=-1)  # (B, 2)
    start_cells = center_cells
    if relocate_seed:
        start_cells, moved = _relocate_blocked_seeds(free_space, center_cells)
        seed_relocations = seed_relocations + moved.sum()
    reachable = _gpu_bfs(free_space, start_cells)

    # Check reachable count per env
    reachable_count = reachable.view(B, -1).sum(dim=-1)  # (B,)
    failed = reachable_count < MIN_REACHABLE_CELLS

    # Retry: walk the park rank for failed envs, parking the first group that
    # still holds an active member. With dense level-7 rooms (16 clutter + 8
    # furniture) we may need many passes. Each pass removes one group and
    # re-checks BFS, keeping rooms as full as possible. Groups are pairwise
    # disjoint and nothing re-activates, so the first-active index strictly
    # increases and the ladder terminates in at most one pass per group.
    if failed.any():
        for retry in range(len(park_groups)):
            if not failed.any():
                break
            failed_ids = torch.where(failed)[0]
            for fi in failed_ids:
                for members in park_groups:
                    if active_mask[fi, members].any():
                        active_mask[fi, members] = False
                        poses[fi, members, :3] = PARK_POS.to(device)
                        break

            # Rebuild and re-check
            occupancy = _build_occupancy_grid(poses, active_mask, sizes, room_w, room_h, device, has_room_walls)
            free_space = _inflate_obstacles(occupancy)
            start_cells = center_cells
            if relocate_seed:
                start_cells, moved = _relocate_blocked_seeds(free_space, center_cells)
                seed_relocations = seed_relocations + moved.sum()
            reachable = _gpu_bfs(free_space, start_cells)
            reachable_count = reachable.view(B, -1).sum(dim=-1)
            failed = reachable_count < MIN_REACHABLE_CELLS
            ladder_passes += 1

    # --- Phase 6: Extract spawn points ---
    # The shared pool feeds robot reset, the goal command, and the subgoal
    # planner endpoints (0.3 m robot-radius inflation).
    spawn_xy, spawn_count = _extract_spawn_points(reachable)

    # Store on env for robot reset, goal command, and path planning
    if not hasattr(env, "_proc_room_spawn_pts"):
        env._proc_room_spawn_pts = torch.zeros(
            env.num_envs, NUM_SPAWN_POINTS, 2, device=device
        )
        env._proc_room_spawn_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=device
        )
        env._proc_room_active_mask = torch.zeros(
            env.num_envs, NUM_OBJECTS, dtype=torch.bool, device=device
        )
        env._proc_room_free_space = torch.zeros(
            env.num_envs, GRID_SIZE, GRID_SIZE, dtype=torch.bool, device=device
        )

    env._proc_room_spawn_pts[env_ids] = spawn_xy
    env._proc_room_spawn_count[env_ids] = spawn_count
    env._proc_room_active_mask[env_ids] = active_mask
    env._proc_room_free_space[env_ids] = free_space

    # Robot-spawn-only pool: a more-eroded interior read solely by the robot
    # reset event, so the robot starts with more clearance than the goal /
    # subgoal endpoints (which stay on the shared pool). Falls back per-env to
    # the shared pool where the extra erosion emptied the room.
    if robot_spawn_inflation_cells > 0:
        robot_reachable = _erode_reachable(reachable, robot_spawn_inflation_cells)
        robot_spawn_xy, robot_spawn_count = _extract_spawn_points(robot_reachable)
        empty = robot_spawn_count == 0
        if empty.any():
            robot_spawn_xy[empty] = spawn_xy[empty]
            robot_spawn_count[empty] = spawn_count[empty]
        if not hasattr(env, "_proc_room_robot_spawn_pts"):
            env._proc_room_robot_spawn_pts = torch.zeros(
                env.num_envs, NUM_SPAWN_POINTS, 2, device=device
            )
            env._proc_room_robot_spawn_count = torch.zeros(
                env.num_envs, dtype=torch.long, device=device
            )
        env._proc_room_robot_spawn_pts[env_ids] = robot_spawn_xy
        env._proc_room_robot_spawn_count[env_ids] = robot_spawn_count

    if health_sink is not None:
        column_slots_seen = column_slots or tuple(CLUTTER_TALL_CYL_SLOTS)
        ordinary = [s for s in CLUTTER_SLOTS if s not in set(column_slots_seen)]
        parked = placed_mask & ~active_mask
        health_sink.update({
            "envs": B,
            "column_phase_fired": n_column_promotions,
            "placed_furniture": int(placed_mask[:, FURNITURE_SLOTS].sum()),
            "placed_clutter": int(placed_mask[:, ordinary].sum()),
            "placed_columns": int(placed_mask[:, list(column_slots_seen)].sum()),
            "parked_furniture": int(parked[:, FURNITURE_SLOTS].sum()),
            "parked_clutter": int(parked[:, ordinary].sum()),
            "parked_columns": int(parked[:, list(column_slots_seen)].sum()),
            "ladder_passes": ladder_passes,
            "bfs_failed": int(failed.sum()),
            "bfs_seed_relocated": int(seed_relocations),
            "spawn_count_min": int(spawn_count.min()),
        })

    # --- Phase 7: Offset by env origins and write ---
    env_origins = env.scene.env_origins[env_ids]  # (B, 3)
    poses[:, :, 0] += env_origins[:, 0:1]
    poses[:, :, 1] += env_origins[:, 1:2]
    poses[:, :, 2] += env_origins[:, 2:3]

    # Write all 44 objects in one batched call
    collection = env.scene[collection_name]
    all_object_ids = torch.arange(NUM_OBJECTS, device=device)
    collection.write_body_link_pose_to_sim_index(body_poses=poses, env_ids=env_ids, body_ids=all_object_ids)

    # Enclosure: pose the standalone ceiling slab (outside the collection).
    if ceiling_entity_name is not None:
        _pose_ceiling_slab(
            env, env_ids, ceiling_entity_name, p_ceil, ceiling_height_range, device
        )
