"""One-way ceiling surface for the enriched procedural rooms.

The enclosure has to read as a solid ceiling to the robot's camera underneath it
and stay out of the way of an overhead debug/``--video`` camera above it. A
closed box cannot do both: whichever way it is culled, one of its two faces
still points at one of the two cameras. A surface with a single face can — point
that face down and the camera below sees it, the camera above sees through it.

RTX applies back-face culling only when ``/rtx/hydra/faceCulling/enabled`` is
set, and decides per prim from a *custom* ``singleSided`` bool attribute; the
``doubleSided`` schema attribute is ignored. That switch is global, so the
asymmetry has to live in the geometry rather than in a per-camera flag.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.sim import schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from pxr import Usd


def quad_points(
    size: tuple[float, float], z: float
) -> list[tuple[float, float, float]]:
    """The four corners of a horizontal quad wound so its one normal faces down.

    USD winds right-handed, so a face whose normal points along -Z runs
    clockwise when read from above.
    """
    half_x, half_y = size[0] / 2.0, size[1] / 2.0
    return [
        (-half_x, -half_y, z),
        (-half_x, half_y, z),
        (half_x, half_y, z),
        (half_x, -half_y, z),
    ]


@clone
def spawn_ceiling_surface(
    prim_path: str,
    cfg: CeilingSurfaceCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a single-quad ``UsdGeom.Mesh`` ceiling prim.

    Mirrors the shape spawners' layout — the mesh at ``{prim_path}/geometry/mesh``
    and the rigid-body properties on ``{prim_path}`` itself, so the entity poses
    like any other rigid object.

    Args:
        prim_path: The prim path or pattern to spawn the asset at.
        cfg: The configuration instance.
        translation: Translation w.r.t. the parent prim. Defaults to the origin.
        orientation: Orientation in (x, y, z, w) w.r.t. the parent prim.
            Defaults to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    from pxr import Sdf

    stage = get_current_stage()
    create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )

    mesh_prim_path = f"{prim_path}/geometry/mesh"
    mesh_prim = create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        attributes={
            "points": quad_points(cfg.size, cfg.surface_offset),
            "faceVertexIndices": [0, 1, 2, 3],
            "faceVertexCounts": [4],
            "subdivisionScheme": "none",
        },
        stage=stage,
    )
    mesh_prim.CreateAttribute(
        "singleSided",
        Sdf.ValueTypeNames.Bool,
        custom=True,
        variability=Sdf.VariabilityUniform,
    ).Set(True)

    if cfg.visual_material is not None:
        material_path = f"{prim_path}/geometry/{cfg.visual_material_path}"
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(mesh_prim_path, material_path, stage=stage)

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

    return stage.GetPrimAtPath(prim_path)


@configclass
class CeilingSurfaceCfg(RigidObjectSpawnerCfg):
    """Configuration parameters for a horizontal down-facing quad prim.

    See :meth:`spawn_ceiling_surface` for more information.
    """

    func: Callable = spawn_ceiling_surface

    size: tuple[float, float] = MISSING
    """Extent of the quad along x and y (in m)."""

    surface_offset: float = 0.0
    """Height of the surface above the prim's own origin (in m)."""

    visual_material_path: str = "material"
    """Path to the visual material, relative to the prim's path if not absolute."""

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties. Defaults to None, i.e. no material is added."""
