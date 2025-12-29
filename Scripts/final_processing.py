"""
Final processing for the mecanum robot USD:
  - Rotate robot -90 deg about X.
  - Translate robot +4.5 units along Y.
  - De-instance all prims (clear instance bodies) so the asset can be instanced cleanly.

Usage:
  python Scripts/final_processing.py --stage "<in.usd>" --out "<out.usd>" [--root "/World/mechanum_robot_v18"]
"""

import argparse
from typing import Optional

from pxr import Usd, UsdGeom, Gf


def deinstance_all(stage: Usd.Stage) -> int:
    """Clear instanceable flag on all prims; returns count changed."""
    changed = 0
    for prim in stage.Traverse():
        if prim.IsInstance() or prim.IsInstanceable():
            prim.SetInstanceable(False)
            changed += 1
    return changed


def apply_root_transform(root: Usd.Prim, rot_deg_x: float, translate_y: float) -> None:
    """Bake rotation about X and translation along Y onto the root's local transform."""
    xf = UsdGeom.Xformable(root)
    # Get existing local transform
    existing = xf.GetLocalTransformation(Usd.TimeCode.Default())
    existing_mat = existing[0] if isinstance(existing, tuple) else existing

    rot = Gf.Matrix4d(1.0)
    rot.SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), rot_deg_x))
    trans = Gf.Matrix4d(1.0)
    trans.SetTranslate(Gf.Vec3d(0.0, translate_y, 0.0))

    # Note: USD uses row-major composition; rot * trans applies rotation first, then translation in world axes.
    delta = rot * trans
    new_local = existing_mat * delta

    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(new_local)


def main():
    parser = argparse.ArgumentParser(description="Final processing: rotate, translate, and de-instance robot.")
    parser.add_argument("--stage", required=True, help="Input USD stage.")
    parser.add_argument("--out", required=True, help="Output USD stage.")
    parser.add_argument("--root", default="/World/mechanum_robot_v18", help="Root prim path for the robot.")
    parser.add_argument("--rotate-x", type=float, default=-90.0, help="Degrees about X to apply to root.")
    parser.add_argument("--translate-y", type=float, default=4.5, help="Translation along Y to apply to root.")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")

    root = stage.GetPrimAtPath(args.root)
    if not root:
        raise SystemExit(f"Root prim not found: {args.root}")
    if not root.IsA(UsdGeom.Xform):
        raise SystemExit(f"Root prim is not an Xform: {args.root}")

    apply_root_transform(root, args.rotate_x, args.translate_y)
    deinstanced = deinstance_all(stage)

    stage.GetRootLayer().Export(args.out)
    print(f"Applied rotation {args.rotate_x} deg about X and translation {args.translate_y} along Y to {args.root}")
    print(f"Cleared instanceable flag on {deinstanced} prim(s)")
    print(f"Exported to {args.out}")


if __name__ == "__main__":
    main()
