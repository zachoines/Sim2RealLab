"""Utility functions for the Infinigen + Replicator SDG pipeline.

Implements the infinigen_sdg_utils API based on the NVIDIA tutorial:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_infinigen_sdg.html

This module is loaded by compose_scenes_replicator.py and must run inside
Isaac Sim's Python environment (omni.usd, omni.replicator.core, pxr).
"""

from __future__ import annotations

import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import omni.kit.app
import omni.physx
import omni.replicator.core as rep
import omni.replicator.core.functional as F
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

# Semantics API moved out of pxr in Isaac Sim 4.2+
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

def get_usd_paths(
    files: list[str] | None = None,
    folders: list[str] | None = None,
) -> list[str]:
    """Collect USD file paths from explicit files and folder scans.

    Returns absolute paths so that USD reference resolution can find
    textures and sublayers relative to the asset's own directory.
    """
    paths: list[str] = []
    for f in files or []:
        p = Path(f)
        if p.is_file():
            paths.append(str(p.resolve()))
    for folder in folders or []:
        folder_path = Path(folder)
        if not folder_path.is_dir():
            print(f"[sdg_utils] WARNING: folder not found: {folder}")
            continue
        for ext in ("*.usd", "*.usdc", "*.usda"):
            paths.extend(str(p.resolve()) for p in sorted(folder_path.rglob(ext)))
    return paths


def load_env(url: str, prim_path: str = "/Environment") -> None:
    """Load an Infinigen room USD onto the stage as a reference."""
    stage = omni.usd.get_context().get_stage()

    # Resolve to absolute path for correct texture/sublayer resolution
    abs_url = str(Path(url).resolve())

    # Remove previous environment if present
    existing = stage.GetPrimAtPath(prim_path)
    if existing and existing.IsValid():
        stage.RemovePrim(prim_path)

    # Create prim and add reference to the USD file
    env_prim = stage.DefinePrim(prim_path, "Xform")
    env_prim.GetReferences().AddReference(abs_url)

    # Wait for stage to load
    omni.kit.app.get_app().update()
    omni.kit.app.get_app().update()
    print(f"[sdg_utils] Loaded environment: {url}")


def setup_env(
    root_path: str = "/Environment",
    hide_ceiling: bool = True,
) -> None:
    """Add collision to the environment and hide ceiling prims."""
    stage = omni.usd.get_context().get_stage()

    # Ensure a physics scene exists
    physics_scene_path = "/PhysicsScene"
    if not stage.GetPrimAtPath(physics_scene_path).IsValid():
        scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        print(f"[sdg_utils] WARNING: root prim {root_path} not found")
        return

    # Add collision to all meshes in the environment
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            # Add collision API
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            # Add mesh collision API for triangle mesh collider
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_col.GetApproximationAttr().Set("meshSimplification")

    # Hide ceiling prims so they don't block overhead view or scatter queries.
    # Infinigen's overhead.gin uses an InvisibleToCamera shader that is lost
    # during texture baking, so we hide by USD visibility instead.
    if hide_ceiling:
        for prim in Usd.PrimRange(root_prim):
            name = prim.GetName().lower()
            if "ceiling" in name or "top_wall" in name:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()

    omni.kit.app.get_app().update()
    print(f"[sdg_utils] Environment setup complete (colliders added)")


def get_floor_prims(root_path: str = "/Environment") -> list:
    """Find all floor mesh prims under the environment root.

    Infinigen rooms name floor prims like ``dining_room_0_0_floor``.
    """
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []

    floor_prims = []
    for prim in Usd.PrimRange(root):
        if "floor" in prim.GetName().lower() and prim.IsA(UsdGeom.Mesh):
            floor_prims.append(prim)

    print(f"[sdg_utils] Found {len(floor_prims)} floor mesh prim(s): "
          f"{[p.GetPath().pathString for p in floor_prims]}")
    return floor_prims


def get_matching_prim_location(
    name: str,
    root_path: str = "/Environment",
) -> tuple[float, float, float]:
    """Find the world position of a prim whose name contains the given string."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Root prim {root_path} not found")

    for prim in Usd.PrimRange(root):
        if name.lower() in prim.GetName().lower():
            xformable = UsdGeom.Xformable(prim)
            if xformable:
                world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                translation = world_xform.ExtractTranslation()
                return (translation[0], translation[1], translation[2])

    raise RuntimeError(
        f"No prim matching '{name}' found under {root_path}"
    )


# ---------------------------------------------------------------------------
# Asset spawning
# ---------------------------------------------------------------------------

# Cache metersPerUnit lookups to avoid re-opening USD files
_mpu_cache: dict[str, float] = {}


def _get_asset_meters_per_unit(asset_url: str) -> float:
    """Read metersPerUnit from a USD asset file (cached)."""
    if asset_url in _mpu_cache:
        return _mpu_cache[asset_url]
    try:
        ref_stage = Usd.Stage.Open(asset_url)
        mpu = ref_stage.GetMetadata("metersPerUnit") or 1.0
    except Exception:
        mpu = 1.0
    _mpu_cache[asset_url] = mpu
    return mpu


def _unit_scale_for_asset(asset_url: str, stage) -> float:
    """Compute the uniform scale needed to convert asset units to stage units."""
    asset_mpu = _get_asset_meters_per_unit(asset_url)
    stage_mpu = stage.GetMetadata("metersPerUnit") or 1.0
    return asset_mpu / stage_mpu


def get_env_floor_bbox(
    root_path: str = "/Environment",
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Get the XY bounding box and floor Z of the environment.

    Returns ((x_min, x_max), (y_min, y_max), floor_z).
    """
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return ((-5, 5), (-5, 5), 0.0)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(root)
    rng = bbox.ComputeAlignedRange()
    lo = rng.GetMin()
    hi = rng.GetMax()

    # Inset by 0.5m from walls to avoid spawning inside them
    inset = 0.5
    return (
        (lo[0] + inset, hi[0] - inset),
        (lo[1] + inset, hi[1] - inset),
        lo[2],  # floor Z
    )


def _random_floor_position(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    floor_z: float,
    drop_height: float = 0.3,
) -> tuple[float, float, float]:
    """Random XY position within the room, slightly above floor for physics drop."""
    return (
        random.uniform(*x_range),
        random.uniform(*y_range),
        floor_z + drop_height,
    )


def _add_rigid_body(prim, gravity_disabled: bool = False) -> None:
    """Add rigid body physics to a prim."""
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)

    # Dynamic bodies require convex approximation (not triangle mesh)
    if not gravity_disabled:
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
        else:
            mesh_col = UsdPhysics.MeshCollisionAPI(prim)
        mesh_col.GetApproximationAttr().Set("convexDecomposition")

    if gravity_disabled:
        rb = UsdPhysics.RigidBodyAPI(prim)
        rb.GetRigidBodyEnabledAttr().Set(False)


def spawn_labeled_assets(
    config: dict,
    working_area_loc: tuple[float, float, float],
) -> list:
    """Spawn labeled assets from config folders/files within the room."""
    stage = omni.usd.get_context().get_stage()
    spawned = []

    auto_label = config.get("auto_label", {})
    num = auto_label.get("num", 0)
    gravity_chance = auto_label.get("gravity_disabled_chance", 0.25)

    # Collect USD paths for auto_label
    asset_paths = get_usd_paths(
        files=auto_label.get("files", []),
        folders=auto_label.get("folders", []),
    )
    if not asset_paths:
        print("[sdg_utils] WARNING: No labeled asset paths found")
        return spawned

    for i in range(num):
        asset_url = random.choice(asset_paths)
        prim_path = f"/LabeledAssets/asset_{i}"
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(asset_url)

        # Derive label from filename
        label = Path(asset_url).stem
        regex_pat = auto_label.get("regex_replace_pattern")
        regex_repl = auto_label.get("regex_replace_repl", "")
        if regex_pat:
            label = re.sub(regex_pat, regex_repl, label)

        # Add semantic label
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set(label)

        # Unit-aware scale (Residential assets use cm, stage uses m)
        unit_scale = _unit_scale_for_asset(asset_url, stage)

        # Scale only — position is set later by scatter_on_floor()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddScaleOp().Set(Gf.Vec3f(unit_scale, unit_scale, unit_scale))
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

        if unit_scale != 1.0:
            print(f"  [sdg_utils] asset_{i}: {label} scaled {unit_scale:.4f}x (unit conversion)")

        # Physics — all labeled assets get gravity for settling
        gravity_disabled = random.random() < gravity_chance
        _add_rigid_body(prim, gravity_disabled=gravity_disabled)

        spawned.append(prim)

    # Manual label assets
    for manual in config.get("manual_label", []):
        url = manual.get("url", "")
        label = manual.get("label", "unknown")
        count = manual.get("num", 1)
        grav_chance = manual.get("gravity_disabled_chance", 0.25)

        unit_scale = _unit_scale_for_asset(url, stage)

        for j in range(count):
            prim_path = f"/LabeledAssets/{label}_{j}"
            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(url)

            # Scale only — position is set later by scatter_on_floor()
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddScaleOp().Set(Gf.Vec3f(unit_scale, unit_scale, unit_scale))
            xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

            gravity_disabled = random.random() < grav_chance
            _add_rigid_body(prim, gravity_disabled=gravity_disabled)
            spawned.append(prim)

    print(f"[sdg_utils] Spawned {len(spawned)} labeled assets")
    return spawned


def spawn_shape_distractors(
    config: dict,
    working_area_loc: tuple[float, float, float],
) -> list:
    """Spawn primitive shape distractors (capsule, cone, cylinder, sphere, cube)."""
    stage = omni.usd.get_context().get_stage()
    spawned = []

    num = config.get("num", 0)
    gravity_chance = config.get("gravity_disabled_chance", 0.25)
    shape_types = config.get("types", ["capsule", "cone", "cylinder", "sphere", "cube"])

    # Map shape names to USD prim types
    type_map = {
        "capsule": "Capsule",
        "cone": "Cone",
        "cylinder": "Cylinder",
        "sphere": "Sphere",
        "cube": "Cube",
    }

    x_range, y_range, floor_z = get_env_floor_bbox()

    for i in range(num):
        shape_name = random.choice(shape_types)
        usd_type = type_map.get(shape_name, "Cube")
        prim_path = f"/ShapeDistractors/shape_{i}"
        prim = stage.DefinePrim(prim_path, usd_type)

        # Random scale — realistic small/medium object sizes (5cm to 30cm)
        scale = random.uniform(0.05, 0.30)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()

        # Scale only — position is set later by scatter_on_floor()
        xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

        # Random color
        material_path = f"/ShapeDistractors/mat_{i}"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(random.random(), random.random(), random.random())
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(material)

        # Physics
        gravity_disabled = random.random() < gravity_chance
        _add_rigid_body(prim, gravity_disabled=gravity_disabled)

        spawned.append(prim)

    print(f"[sdg_utils] Spawned {len(spawned)} shape distractors")
    return spawned


def spawn_mesh_distractors(
    config: dict,
    working_area_loc: tuple[float, float, float],
) -> list:
    """Spawn mesh distractors from USD files in configured folders."""
    stage = omni.usd.get_context().get_stage()
    spawned = []

    num = config.get("num", 0)
    gravity_chance = config.get("gravity_disabled_chance", 0.25)

    asset_paths = get_usd_paths(
        files=config.get("files", []),
        folders=config.get("folders", []),
    )
    if not asset_paths:
        print("[sdg_utils] WARNING: No mesh distractor paths found")
        return spawned

    x_range, y_range, floor_z = get_env_floor_bbox()

    for i in range(num):
        asset_url = random.choice(asset_paths)
        prim_path = f"/MeshDistractors/mesh_{i}"
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(asset_url)

        # Unit-aware scale
        unit_scale = _unit_scale_for_asset(asset_url, stage)
        # Small random variation on top of unit correction (0.8x to 1.2x)
        variation = random.uniform(0.8, 1.2)
        total_scale = unit_scale * variation

        # Scale only — position is set later by scatter_on_floor()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddScaleOp().Set(Gf.Vec3f(total_scale, total_scale, total_scale))
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

        if unit_scale != 1.0:
            print(f"  [sdg_utils] mesh_{i}: scaled {total_scale:.4f}x (unit={unit_scale}, var={variation:.2f})")

        # Physics
        gravity_disabled = random.random() < gravity_chance
        _add_rigid_body(prim, gravity_disabled=gravity_disabled)

        spawned.append(prim)

    print(f"[sdg_utils] Spawned {len(spawned)} mesh distractors")
    return spawned


# ---------------------------------------------------------------------------
# Randomization
# ---------------------------------------------------------------------------

def randomize_poses(
    assets: list,
    working_area_loc: tuple[float, float, float],
) -> None:
    """Re-randomize positions within the room. Preserves existing scale."""
    x_range, y_range, floor_z = get_env_floor_bbox()

    for prim in assets:
        xform = UsdGeom.Xformable(prim)

        # Read existing scale before clearing
        existing_scale = Gf.Vec3f(1, 1, 1)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                existing_scale = op.Get()
                break

        xform.ClearXformOpOrder()

        pos = _random_floor_position(x_range, y_range, floor_z, drop_height=0.5)
        xform.AddScaleOp().Set(existing_scale)
        xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, random.uniform(0, 360)))
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

    omni.kit.app.get_app().update()
    print(f"[sdg_utils] Randomized poses for {len(assets)} assets")


def scatter_on_floor(
    assets: list,
    floor_prims: list,
    check_collisions: bool = False,
    offset: float = 0.05,
) -> None:
    """Place assets on floor surfaces using Replicator scatter_2d.

    Uses area-weighted barycentric sampling on the actual floor mesh
    so objects land only on walkable floor area.  Falls back to AABB
    placement if no floor prims are found.

    Note: check_collisions defaults to False because the collision
    checker can segfault on non-Mesh prims (primitives like Capsule).
    Physics simulation handles overlap resolution instead.
    """
    if not assets:
        return

    if not floor_prims:
        print("[sdg_utils] WARNING: No floor prims — falling back to AABB placement")
        x_range, y_range, floor_z = get_env_floor_bbox()
        for prim in assets:
            xform = UsdGeom.Xformable(prim)
            pos = _random_floor_position(x_range, y_range, floor_z, drop_height=0.3)
            # Update existing translate op (set during spawn)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(*pos))
                    break
            xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, random.uniform(0, 360)))
        omni.kit.app.get_app().update()
        return

    try:
        F.randomizer.scatter_2d(
            prims=assets,
            surface_prims=floor_prims,
            offset=offset,
            check_for_collisions=check_collisions,
        )
    except ValueError as e:
        # scatter_2d raises ValueError when collision retries are exhausted
        print(f"[sdg_utils] WARNING: scatter_2d failed ({e}), falling back to AABB")
        x_range, y_range, floor_z = get_env_floor_bbox()
        for prim in assets:
            xform = UsdGeom.Xformable(prim)
            pos = _random_floor_position(x_range, y_range, floor_z, drop_height=0.3)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(*pos))
                    break
        omni.kit.app.get_app().update()

    # scatter_2d only sets translate — apply random yaw rotation
    for prim in assets:
        xform = UsdGeom.Xformable(prim)
        # Find existing rotateXYZ op or add one
        rot_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rot_op = op
                break
        if rot_op is None:
            rot_op = xform.AddRotateXYZOp()
        rot_op.Set(Gf.Vec3f(0, 0, random.uniform(0, 360)))

    omni.kit.app.get_app().update()
    print(f"[sdg_utils] Scattered {len(assets)} assets on "
          f"{len(floor_prims)} floor prim(s)")


# ---------------------------------------------------------------------------
# Wall-snap placement
# ---------------------------------------------------------------------------

@dataclass
class WallSegment:
    """A floor-level wall segment with computed inward normal."""
    start: tuple[float, float]        # (x, y) world space
    end: tuple[float, float]          # (x, y) world space
    inward_normal: tuple[float, float] # unit 2D vector pointing into room
    length: float


def get_wall_prims(root_path: str = "/Environment") -> list:
    """Find wall mesh prims under the environment root.

    Excludes ``top_wall`` prims (ceiling-facing surfaces hidden by setup_env).
    """
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []

    wall_prims = []
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if "wall" in name and "top_wall" not in name and prim.IsA(UsdGeom.Mesh):
            wall_prims.append(prim)

    print(f"[sdg_utils] Found {len(wall_prims)} wall mesh prim(s): "
          f"{[p.GetPath().pathString for p in wall_prims]}")
    return wall_prims


def _extract_wall_segments(
    wall_prim,
    z_tolerance: float = 0.05,
) -> list[WallSegment]:
    """Extract floor-level wall segments with inward normals from a wall mesh."""
    mesh = UsdGeom.Mesh(wall_prim)
    points_local = mesh.GetPointsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()

    if not points_local or not face_indices or not face_counts:
        return []

    # Transform points to world space
    xformable = UsdGeom.Xformable(wall_prim)
    world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    points = []
    for pt in points_local:
        p = world_xform.Transform(Gf.Vec3d(pt[0], pt[1], pt[2]))
        points.append((p[0], p[1], p[2]))

    # Identify floor-level vertices
    z_values = [p[2] for p in points]
    floor_z = min(z_values)
    floor_verts = {i for i, p in enumerate(points) if abs(p[2] - floor_z) < z_tolerance}

    if len(floor_verts) < 2:
        return []

    # Extract floor-level edges from face indices
    floor_edges: set[frozenset] = set()
    offset = 0
    for count in face_counts:
        face = face_indices[offset:offset + count]
        offset += count
        for i in range(count):
            v0, v1 = face[i], face[(i + 1) % count]
            if v0 in floor_verts and v1 in floor_verts:
                floor_edges.add(frozenset((v0, v1)))

    if not floor_edges:
        return []

    # Build adjacency and chain edges into ordered polylines
    adj: dict[int, set[int]] = defaultdict(set)
    for edge in floor_edges:
        v0, v1 = tuple(edge)
        adj[v0].add(v1)
        adj[v1].add(v0)

    visited: set[frozenset] = set()
    chains: list[list[int]] = []
    for start_v in adj:
        for next_v in list(adj[start_v]):
            ek = frozenset((start_v, next_v))
            if ek in visited:
                continue
            chain = [start_v, next_v]
            visited.add(ek)
            cur, prev = next_v, start_v
            while True:
                unvis = [n for n in adj[cur] - {prev}
                         if frozenset((cur, n)) not in visited]
                if not unvis:
                    break
                nxt = unvis[0]
                visited.add(frozenset((cur, nxt)))
                chain.append(nxt)
                prev, cur = cur, nxt
            chains.append(chain)

    # Room center from floor-level vertices (for inward normal orientation)
    floor_pts = [points[i] for i in floor_verts]
    cx = sum(p[0] for p in floor_pts) / len(floor_pts)
    cy = sum(p[1] for p in floor_pts) / len(floor_pts)

    # Convert chains to WallSegment objects
    segments: list[WallSegment] = []
    for chain in chains:
        for i in range(len(chain) - 1):
            ax, ay = points[chain[i]][0], points[chain[i]][1]
            bx, by = points[chain[i + 1]][0], points[chain[i + 1]][1]
            dx, dy = bx - ax, by - ay
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-6:
                continue
            dx /= length
            dy /= length

            # Two candidate perpendicular normals
            n1x, n1y = -dy, dx
            # Pick the one pointing toward room center
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            to_center_x, to_center_y = cx - mx, cy - my
            if n1x * to_center_x + n1y * to_center_y > 0:
                inward = (n1x, n1y)
            else:
                inward = (dy, -dx)

            segments.append(WallSegment(
                start=(ax, ay), end=(bx, by),
                inward_normal=inward, length=length,
            ))

    print(f"[sdg_utils] Extracted {len(segments)} wall segments from "
          f"{wall_prim.GetPath().pathString}")
    return segments


def _filter_doorway_segments(
    segments: list[WallSegment],
    root_path: str = "/Environment",
    margin: float = 0.5,
    min_length: float = 0.3,
) -> list[WallSegment]:
    """Remove wall segments near doorways or too short for furniture."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)

    # Collect door XY positions
    door_positions: list[tuple[float, float]] = []
    if root and root.IsValid():
        for prim in Usd.PrimRange(root):
            if "door" in prim.GetName().lower():
                xformable = UsdGeom.Xformable(prim)
                if xformable:
                    xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    t = xf.ExtractTranslation()
                    door_positions.append((t[0], t[1]))

    filtered = []
    for seg in segments:
        if seg.length < min_length:
            continue
        mx, my = (seg.start[0] + seg.end[0]) / 2.0, (seg.start[1] + seg.end[1]) / 2.0
        near_door = False
        for dx, dy in door_positions:
            if math.sqrt((mx - dx) ** 2 + (my - dy) ** 2) < margin:
                near_door = True
                break
        if not near_door:
            filtered.append(seg)

    removed = len(segments) - len(filtered)
    if removed:
        print(f"[sdg_utils] Filtered {removed} wall segments "
              f"(doorways/short), {len(filtered)} remain")
    return filtered


def snap_to_walls(
    assets: list,
    wall_prims: list,
    floor_z: float,
    offset: float = 0.15,
    min_spacing: float = 0.8,
    root_path: str = "/Environment",
) -> list:
    """Place assets against walls with inward-facing orientation.

    Returns list of assets that could NOT be wall-snapped (for fallback
    to scatter_on_floor).
    """
    if not assets or not wall_prims:
        return list(assets)

    # Extract and merge segments from all wall prims
    all_segments: list[WallSegment] = []
    for wp in wall_prims:
        all_segments.extend(_extract_wall_segments(wp))

    if not all_segments:
        print("[sdg_utils] WARNING: No wall segments extracted")
        return list(assets)

    # Filter doorway regions
    all_segments = _filter_doorway_segments(all_segments, root_path)
    if not all_segments:
        print("[sdg_utils] WARNING: All wall segments filtered out")
        return list(assets)

    # Length-weighted sampling
    weights = [s.length for s in all_segments]
    placed_xy: list[tuple[float, float]] = []
    failed: list = []

    for prim in assets:
        placed = False
        for _ in range(20):
            seg = random.choices(all_segments, weights=weights, k=1)[0]
            t = random.uniform(0.1, 0.9)
            px = seg.start[0] + t * (seg.end[0] - seg.start[0])
            py = seg.start[1] + t * (seg.end[1] - seg.start[1])

            # Enforce min spacing
            if any(math.sqrt((px - ox) ** 2 + (py - oy) ** 2) < min_spacing
                   for ox, oy in placed_xy):
                continue

            # Wall-snapped position
            wx = px + offset * seg.inward_normal[0]
            wy = py + offset * seg.inward_normal[1]
            wz = floor_z + 0.01

            # Yaw: furniture faces into room (along inward normal)
            yaw_deg = math.degrees(math.atan2(
                seg.inward_normal[1], seg.inward_normal[0],
            ))

            # Read existing scale
            xform = UsdGeom.Xformable(prim)
            existing_scale = Gf.Vec3f(1, 1, 1)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    existing_scale = op.Get()
                    break

            xform.ClearXformOpOrder()
            xform.AddScaleOp().Set(existing_scale)
            xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, yaw_deg))
            xform.AddTranslateOp().Set(Gf.Vec3d(wx, wy, wz))

            placed_xy.append((px, py))
            placed = True
            break

        if not placed:
            failed.append(prim)

    omni.kit.app.get_app().update()
    placed_count = len(assets) - len(failed)
    print(f"[sdg_utils] Wall-snapped {placed_count}/{len(assets)} assets "
          f"({len(failed)} failed -> fallback)")
    return failed


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def create_scene_lights(
    num_lights: int = 3,
    working_area_loc: tuple[float, float, float] = (0, 0, 0),
) -> list:
    """Create randomizable scene lights within the room."""
    stage = omni.usd.get_context().get_stage()
    lights = []

    x_range, y_range, floor_z = get_env_floor_bbox()
    # Lights should be near ceiling height (2.0-3.0m above floor)
    ceiling_z = floor_z + random.uniform(2.2, 2.8)

    for i in range(num_lights):
        light_path = f"/SceneLights/light_{i}"
        light = UsdLux.SphereLight.Define(stage, light_path)
        light.GetIntensityAttr().Set(random.uniform(500, 2000))
        light.GetRadiusAttr().Set(0.1)

        xform = UsdGeom.Xformable(light.GetPrim())
        xform.ClearXformOpOrder()
        pos = (random.uniform(*x_range), random.uniform(*y_range), ceiling_z)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

        lights.append(light.GetPrim())

    print(f"[sdg_utils] Created {num_lights} scene lights")
    return lights


def randomize_lights(
    lights: list,
    working_area_loc: tuple[float, float, float],
) -> None:
    """Randomize light positions and intensities."""
    x_range, y_range, floor_z = get_env_floor_bbox()

    for light_prim in lights:
        xform = UsdGeom.Xformable(light_prim)
        xform.ClearXformOpOrder()
        ceiling_z = floor_z + random.uniform(2.2, 2.8)
        pos = (random.uniform(*x_range), random.uniform(*y_range), ceiling_z)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

        light = UsdLux.SphereLight(light_prim)
        if light:
            light.GetIntensityAttr().Set(random.uniform(500, 3000))
            light.GetColorAttr().Set(Gf.Vec3f(
                random.uniform(0.8, 1.0),
                random.uniform(0.8, 1.0),
                random.uniform(0.8, 1.0),
            ))


_dome_light_prim = None


def register_dome_light_randomizer() -> None:
    """Create a dome light for ambient illumination."""
    global _dome_light_prim
    stage = omni.usd.get_context().get_stage()
    dome_path = "/DomeLight"
    _dome_light_prim = UsdLux.DomeLight.Define(stage, dome_path)
    _dome_light_prim.GetIntensityAttr().Set(500)


def randomize_dome_lights() -> None:
    """Randomize dome light intensity and color (called per-frame)."""
    global _dome_light_prim
    if _dome_light_prim:
        _dome_light_prim.GetIntensityAttr().Set(random.uniform(200, 1000))
        _dome_light_prim.GetColorAttr().Set(Gf.Vec3f(
            random.uniform(0.85, 1.0),
            random.uniform(0.85, 1.0),
            random.uniform(0.85, 1.0),
        ))


_shape_distractor_prims = []


def register_shape_distractors_color_randomizer(shape_distractors: list) -> None:
    """Store shape distractor references for per-frame color randomization."""
    global _shape_distractor_prims
    _shape_distractor_prims = shape_distractors


def randomize_shape_distractor_colors() -> None:
    """Randomize shape distractor colors (called per-frame)."""
    stage = omni.usd.get_context().get_stage()
    for prim in _shape_distractor_prims:
        # Find bound material and update diffuse color
        binding = UsdShade.MaterialBindingAPI(prim)
        material, _ = binding.ComputeBoundMaterial()
        if material:
            for shader_prim in Usd.PrimRange(material.GetPrim()):
                shader = UsdShade.Shader(shader_prim)
                if shader:
                    diffuse = shader.GetInput("diffuseColor")
                    if diffuse:
                        diffuse.Set(Gf.Vec3f(
                            random.random(), random.random(), random.random()
                        ))


# ---------------------------------------------------------------------------
# Physics simulation
# ---------------------------------------------------------------------------

def run_simulation(num_frames: int = 200, render: bool = False) -> None:
    """Advance the physics simulation for the given number of frames."""
    app = omni.kit.app.get_app()
    for _ in range(num_frames):
        app.update()
    print(f"[sdg_utils] Ran simulation for {num_frames} frames (render={render})")


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def randomize_camera_poses(
    cameras: list,
    targets: list,
    distance_range: list | tuple = (1.5, 3.0),
    polar_angle_range: tuple = (0, 75),
) -> None:
    """Point cameras at random target assets from random positions."""
    if not targets:
        return

    for cam_prim in cameras:
        # Pick a random target
        target_prim = random.choice(targets)
        target_xform = UsdGeom.Xformable(target_prim)
        world_xform = target_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        target_pos = world_xform.ExtractTranslation()

        # Random spherical coordinates around the target
        distance = random.uniform(*distance_range)
        polar = math.radians(random.uniform(*polar_angle_range))
        azimuth = math.radians(random.uniform(0, 360))

        cam_x = target_pos[0] + distance * math.sin(polar) * math.cos(azimuth)
        cam_y = target_pos[1] + distance * math.sin(polar) * math.sin(azimuth)
        cam_z = target_pos[2] + distance * math.cos(polar)

        # Set camera position
        cam_xform = UsdGeom.Xformable(cam_prim)
        cam_xform.ClearXformOpOrder()
        translate_op = cam_xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(cam_x, cam_y, cam_z))

        # Aim camera at target using a look-at rotation
        forward = Gf.Vec3d(target_pos[0] - cam_x, target_pos[1] - cam_y, target_pos[2] - cam_z)
        forward = forward.GetNormalized()

        # Compute rotation matrix (camera looks along -Z in USD)
        up = Gf.Vec3d(0, 0, 1)
        right = forward ^ up  # cross product
        if right.GetLength() < 1e-6:
            up = Gf.Vec3d(0, 1, 0)
            right = forward ^ up
        right = right.GetNormalized()
        new_up = right ^ forward
        new_up = new_up.GetNormalized()

        rot_matrix = Gf.Matrix4d()
        rot_matrix.SetIdentity()
        rot_matrix[0][0] = right[0];    rot_matrix[0][1] = right[1];    rot_matrix[0][2] = right[2]
        rot_matrix[1][0] = new_up[0];   rot_matrix[1][1] = new_up[1];   rot_matrix[1][2] = new_up[2]
        rot_matrix[2][0] = -forward[0]; rot_matrix[2][1] = -forward[1]; rot_matrix[2][2] = -forward[2]

        quat = rot_matrix.ExtractRotationQuat()

        rotate_op = cam_xform.AddOrientOp()
        rotate_op.Set(Gf.Quatf(
            float(quat.GetReal()),
            float(quat.GetImaginary()[0]),
            float(quat.GetImaginary()[1]),
            float(quat.GetImaginary()[2]),
        ))


# ---------------------------------------------------------------------------
# Writer setup
# ---------------------------------------------------------------------------

def setup_writer(writer_config: dict):
    """Create and configure a Replicator writer from config."""
    writer_type = writer_config.get("type", "BasicWriter")
    kwargs = writer_config.get("kwargs", {})

    try:
        writer = rep.WriterRegistry.get(writer_type)
        writer.initialize(**kwargs)
        return writer
    except Exception as e:
        print(f"[sdg_utils] WARNING: Failed to create writer '{writer_type}': {e}")
        return None


# ---------------------------------------------------------------------------
# Scene centering and export verification
# ---------------------------------------------------------------------------

def center_scene_at_origin(root_path: str = "/Environment") -> dict:
    """Translate the environment so the floor bounding box center sits at (0,0,0).

    This ensures exported scenes have a consistent coordinate origin for
    use in the training environment (each env clone's origin maps to the
    scene center).

    Returns a dict with the centered floor bounding box metadata:
        {"x_min", "x_max", "y_min", "y_max", "floor_z"}
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        print(f"[sdg_utils] WARNING: cannot center — {root_path} not found")
        return {"x_min": -3, "x_max": 3, "y_min": -3, "y_max": 3, "floor_z": 0.0}

    # Compute floor bounding box before centering
    x_range, y_range, floor_z = get_env_floor_bbox(root_path)
    cx = (x_range[0] + x_range[1]) / 2.0
    cy = (y_range[0] + y_range[1]) / 2.0

    # Apply centering offset to the root prim
    xform = UsdGeom.Xformable(root_prim)
    xform.AddTranslateOp(opSuffix="centering").Set(Gf.Vec3d(-cx, -cy, -floor_z))

    omni.kit.app.get_app().update()

    # Recompute the floor bbox in the new centered frame
    x_range_c, y_range_c, floor_z_c = get_env_floor_bbox(root_path)

    metadata = {
        "x_min": round(x_range_c[0], 3),
        "x_max": round(x_range_c[1], 3),
        "y_min": round(y_range_c[0], 3),
        "y_max": round(y_range_c[1], 3),
        "floor_z": round(floor_z_c, 3),
    }
    print(f"[sdg_utils] Centered scene at origin: {metadata}")
    return metadata


def verify_collision_in_exported_scene(scene_usd_path: str) -> bool:
    """Verify that an exported scene USDC has collision APIs on mesh prims.

    Opens the USDC as a standalone stage and checks that CollisionAPI is
    applied to mesh prims.  If fewer than 90% of meshes have collision,
    re-applies CollisionAPI + MeshCollisionAPI(meshSimplification) and
    saves the fixed file.

    Returns True if the scene already had collision (or was successfully fixed).
    """
    ref_stage = Usd.Stage.Open(scene_usd_path)
    if not ref_stage:
        print(f"[sdg_utils] WARNING: cannot open {scene_usd_path}")
        return False

    mesh_count = 0
    collision_count = 0
    for prim in Usd.PrimRange(ref_stage.GetPseudoRoot()):
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_count += 1

    if mesh_count == 0:
        print(f"[sdg_utils] WARNING: no meshes found in {scene_usd_path}")
        return False

    ratio = collision_count / mesh_count
    if ratio >= 0.9:
        print(f"[sdg_utils] Collision OK: {collision_count}/{mesh_count} meshes "
              f"in {Path(scene_usd_path).name}")
        return True

    # Fix: re-apply collision APIs
    print(f"[sdg_utils] Fixing collision: only {collision_count}/{mesh_count} meshes "
          f"had CollisionAPI in {Path(scene_usd_path).name}")
    for prim in Usd.PrimRange(ref_stage.GetPseudoRoot()):
        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_col.GetApproximationAttr().Set("meshSimplification")

    ref_stage.Save()
    print(f"[sdg_utils] Fixed and saved collision APIs in {Path(scene_usd_path).name}")
    return True
