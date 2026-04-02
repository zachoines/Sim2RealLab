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

# Object sizes (X, Y, Z) for AABB computation
# Must match the spawner configs in _build_proc_room_palette()
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

# BFS grid parameters
GRID_RES = 0.1          # meters per cell
GRID_SIZE = 80           # cells per axis (covers 8m x 8m)
ROBOT_HALF_WIDTH = 0.28  # meters (~max of 0.48x0.44 footprint diagonal / 2)
INFLATION_CELLS = math.ceil(ROBOT_HALF_WIDTH / GRID_RES)  # 3
INFLATION_KERNEL = 2 * INFLATION_CELLS + 1  # 7
MIN_REACHABLE_CELLS = 100
NUM_SPAWN_POINTS = 200   # per env

# Park position for inactive objects
PARK_POS = torch.tensor([100.0, 100.0, -10.0])
_PARK_INIT_STATE = RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, -10.0))


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


def build_proc_room_collection_cfg() -> dict[str, RigidObjectCfg]:
    """Build the 44-object palette as a dict for RigidObjectCollectionCfg.

    Each object gets ``prim_path="{ENV_REGEX_NS}/<Name>"`` so Isaac Lab's
    ``InteractiveScene`` can resolve the per-env regex at scene build time.

    Returns:
        Dict mapping object name → RigidObjectCfg.
    """
    objects = {}

    # --- Walls ---
    for i in range(8):
        objects[f"wall_long_{i}"] = _make_kinematic_cuboid((2.0, 0.15, 1.0), (0.75, 0.75, 0.75))
    for i in range(8):
        objects[f"wall_med_{i}"] = _make_kinematic_cuboid((1.0, 0.15, 1.0), (0.70, 0.70, 0.72))
    for i in range(4):
        objects[f"wall_short_{i}"] = _make_kinematic_cuboid((0.5, 0.15, 1.0), (0.65, 0.65, 0.68))

    # --- Furniture ---
    for i in range(2):
        objects[f"furn_table_{i}"] = _make_kinematic_cuboid((0.8, 0.6, 0.4), (0.55, 0.35, 0.15))
    for i in range(2):
        objects[f"furn_shelf_{i}"] = _make_kinematic_cuboid((1.2, 0.3, 0.8), (0.50, 0.30, 0.12))
    for i in range(2):
        objects[f"furn_cabinet_{i}"] = _make_kinematic_cuboid((0.5, 0.5, 0.6), (0.45, 0.28, 0.10))
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
        objects[f"clutter_tall_cyl_{i}"] = _make_kinematic_cylinder(0.1, 0.7, (0.4, 0.6, 0.3))

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
        poses: (B, N, 7) object poses in env-local frame [x, y, z, qw, qx, qy, qz].
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

    # Extract XY positions and yaw from poses
    cx = poses[:, :, 0]  # (B, N)
    cy = poses[:, :, 1]  # (B, N)
    qw = poses[:, :, 3]  # (B, N)
    qz = poses[:, :, 6]  # (B, N)
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


def _inflate_obstacles(occupancy: torch.Tensor) -> torch.Tensor:
    """Dilate occupancy grid by robot radius using max_pool2d.

    Args:
        occupancy: (B, Gx, Gy) float.

    Returns:
        free_space: (B, Gx, Gy) bool — True = passable after inflation.
    """
    inflated = F.max_pool2d(
        occupancy.unsqueeze(1),
        kernel_size=INFLATION_KERNEL,
        stride=1,
        padding=INFLATION_CELLS,
    ).squeeze(1)
    return inflated < 0.5


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
    return torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)


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


def generate_proc_room(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    collection_name: str = "room_primitives",
    max_internal_walls: int = 0,
    max_furniture: int = 0,
    max_clutter: int = 0,
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
    """
    B = len(env_ids)
    if B == 0:
        return

    device = env.device
    sizes = OBJECT_SIZES.to(device)

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
    room_w = torch.rand(B, device=device) * 3.0 + 4.0  # [4, 7]
    room_h = torch.rand(B, device=device) * 3.0 + 4.0  # [4, 7]

    # Initialize poses: all parked
    poses = torch.zeros(B, NUM_OBJECTS, 7, device=device)
    poses[:, :, 0] = PARK_POS[0]
    poses[:, :, 1] = PARK_POS[1]
    poses[:, :, 2] = PARK_POS[2]
    poses[:, :, 3] = 1.0  # qw = 1 (identity rotation)
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
            door_width = torch.rand(1, device=device).item() * 0.4 + 0.8  # [0.8, 1.2]

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
                        poses[b_idx, slot, 2] = 0.5  # height/2 for 1.0m walls
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
                        poses[b_idx, slot, 2] = 0.5
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
                        poses[b_idx, slot, 2] = 0.5
                        q = _yaw_to_quat(torch.tensor(yaw, device=device))
                        poses[b_idx, slot, 3:7] = q
                        active_mask[b_idx, slot] = True
                        cursor += seg_len

        # --- Phase 3: Furniture placement ---
        n_furn = max_furn[b_idx].item()
        placed_furn_xy = []
        is_open_field = has_room_walls[b_idx].item() == 0
        for f_idx in range(min(n_furn, NUM_FURNITURE)):
            slot = FURNITURE_SLOTS[f_idx]
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
                poses[b_idx, slot, 2] = OBJECT_SIZES[slot, 2].item() / 2
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

        for c_idx in range(min(n_clut, NUM_CLUTTER)):
            slot = CLUTTER_SLOTS[c_idx]
            for _ in range(10):
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
                poses[b_idx, slot, 2] = OBJECT_SIZES[slot, 2].item() / 2
                q = _yaw_to_quat(torch.tensor(cyaw, device=device))
                poses[b_idx, slot, 3:7] = q
                active_mask[b_idx, slot] = True
                placed_clutter_xy.append((cx_, cy_))
                break

    # --- Phase 5: BFS solvability check ---
    occupancy = _build_occupancy_grid(poses, active_mask, sizes, room_w, room_h, device, has_room_walls)
    free_space = _inflate_obstacles(occupancy)

    # BFS from room center (0, 0)
    center_r, center_c = _xy_to_grid(
        torch.zeros(B, device=device), torch.zeros(B, device=device)
    )
    start_cells = torch.stack([center_r, center_c], dim=-1)  # (B, 2)
    reachable = _gpu_bfs(free_space, start_cells)

    # Check reachable count per env
    reachable_count = reachable.view(B, -1).sum(dim=-1)  # (B,)
    failed = reachable_count < MIN_REACHABLE_CELLS

    # Retry: park clutter then furniture for failed envs, one at a time.
    # With dense level-7 rooms (16 clutter + 8 furniture) we may need many
    # passes.  Each pass removes one object (clutter first, then furniture)
    # and re-checks BFS.  This keeps rooms as full as possible.
    if failed.any():
        max_retries = NUM_CLUTTER + NUM_FURNITURE  # worst case: remove everything
        for retry in range(max_retries):
            if not failed.any():
                break
            failed_ids = torch.where(failed)[0]
            for fi in failed_ids:
                # Try parking one clutter first
                parked = False
                for s in reversed(CLUTTER_SLOTS):
                    if active_mask[fi, s]:
                        active_mask[fi, s] = False
                        poses[fi, s, :3] = PARK_POS.to(device)
                        parked = True
                        break
                if not parked:
                    # No clutter left, park one furniture
                    for s in reversed(FURNITURE_SLOTS):
                        if active_mask[fi, s]:
                            active_mask[fi, s] = False
                            poses[fi, s, :3] = PARK_POS.to(device)
                            break

            # Rebuild and re-check
            occupancy = _build_occupancy_grid(poses, active_mask, sizes, room_w, room_h, device, has_room_walls)
            free_space = _inflate_obstacles(occupancy)
            reachable = _gpu_bfs(free_space, start_cells)
            reachable_count = reachable.view(B, -1).sum(dim=-1)
            failed = reachable_count < MIN_REACHABLE_CELLS

    # --- Phase 6: Extract spawn points ---
    spawn_xy, spawn_count = _extract_spawn_points(reachable)

    # Store on env for robot reset and goal command
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

    env._proc_room_spawn_pts[env_ids] = spawn_xy
    env._proc_room_spawn_count[env_ids] = spawn_count
    env._proc_room_active_mask[env_ids] = active_mask

    # --- Phase 7: Offset by env origins and write ---
    env_origins = env.scene.env_origins[env_ids]  # (B, 3)
    poses[:, :, 0] += env_origins[:, 0:1]
    poses[:, :, 1] += env_origins[:, 1:2]
    poses[:, :, 2] += env_origins[:, 2:3]

    # Write all 44 objects in one batched call
    collection = env.scene[collection_name]
    all_object_ids = torch.arange(NUM_OBJECTS, device=device)
    collection.write_body_link_pose_to_sim_index(poses, env_ids, all_object_ids)
