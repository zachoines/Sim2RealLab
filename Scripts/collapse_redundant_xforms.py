"""
Collapse redundant single-child Xform/Scope prims while preserving world transforms.

The operation mirrors manual "drag up and delete empty" cleanup:
  1) Compute world transform of the kept child.
  2) Move the child up to the redundant prim's parent (via a temp name).
  3) Bake the redundant parent's transform into the child so world pose stays fixed.
  4) Remove the empty prim and rename the child back to the original parent name.

Example:
  python Scripts/collapse_redundant_xforms.py --stage Assets/3209-0001-0006-v6/3209-0001-0006.usd \\
      --root /World/strafer --output ./collapse_log.txt --tree-output ./collapsed_tree.txt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pxr import Gf, Sdf, Usd, UsdGeom


PREDICATE = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
ALLOWED_NON_XFORM_ATTRS = {"visibility"}


def iter_tree_lines(stage: Usd.Stage, root_path: str, max_depth: Optional[int] = None):
    """Yield a compact tree view with type and simple physics flags."""
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise SystemExit(f"Root prim not found: {root_path}")
    base_depth = str(root_path).count("/")
    for prim in Usd.PrimRange(root, PREDICATE):
        path_str = str(prim.GetPath())
        depth = path_str.count("/") - base_depth
        if depth < 0 or (max_depth is not None and depth > max_depth):
            continue
        indent = "  " * depth
        type_name = prim.GetTypeName()
        flags = []
        if prim.IsInstance():
            flags.append("INST")
        if prim.HasAuthoredReferences():
            flags.append("REF")
        flag_str = ",".join(flags) if flags else "-"
        yield f"{indent}{path_str.split('/')[-1]} [{type_name}] ({flag_str})"


def _has_only_transform_attrs(prim: Usd.Prim) -> bool:
    """True when authored attrs/metadata are only xform ops or benign visibility."""
    for attr in prim.GetAuthoredAttributes():
        name = attr.GetName()
        if name == "xformOpOrder" or name.startswith("xformOp:"):
            continue
        if name in ALLOWED_NON_XFORM_ATTRS:
            continue
        return False
    if prim.GetAuthoredRelationships():
        return False
    if prim.HasAuthoredReferences() or prim.HasAuthoredInherits() or prim.HasAuthoredPayloads():
        return False
    if prim.GetVariantSets().GetNames():
        return False
    return True


def _is_collapsible_transform(prim: Usd.Prim) -> bool:
    """Heuristic: redundant if it only wraps one child xform and has no useful data."""
    if not prim or not prim.IsValid() or not prim.IsActive():
        return False
    if prim.IsInstance():
        return False
    parent = prim.GetParent()
    if parent is None:
        return False
    if prim.GetPath().pathString == "/":
        return False
    if prim.GetTypeName() not in {"Xform", "Scope"} and not prim.IsA(UsdGeom.Xformable):
        return False
    if not _has_only_transform_attrs(prim):
        return False
    xf = UsdGeom.Xformable(prim)
    if xf and xf.GetResetXformStack():
        return False
    children = [c for c in prim.GetChildren() if c.IsActive()]
    if len(children) != 1:
        return False
    child = children[0]
    if child.IsInstance():
        return False
    if not child.IsA(UsdGeom.Xformable):
        return False
    return True


def _unique_temp_path(stage: Usd.Stage, parent_path: Sdf.Path, base_name: str) -> Sdf.Path:
    """Return a non-colliding temp path under the given parent."""
    idx = 0
    while True:
        suffix = "" if idx == 0 else f"_{idx}"
        candidate = parent_path.AppendChild(f"{base_name}{suffix}")
        prim = stage.GetPrimAtPath(candidate)
        if not prim or not prim.IsValid():
            return candidate
        idx += 1


def _move_prim(stage: Usd.Stage, src: Sdf.Path, dst: Sdf.Path) -> bool:
    """Move by copying the spec in the current edit layer, then removing the source."""
    layer = stage.GetEditTarget().GetLayer()
    if layer.GetPrimAtPath(dst):
        return False
    if not Sdf.CopySpec(layer, src, layer, dst):
        return False
    stage.RemovePrim(src)
    return True


def _compute_new_local(child: Usd.Prim, target_parent: Optional[Usd.Prim]) -> Tuple[Gf.Matrix4d, bool]:
    """Compute child local transform that preserves world pose after reparent."""
    cache = UsdGeom.XformCache()
    child_world = cache.GetLocalToWorldTransform(child)
    child_xf = UsdGeom.Xformable(child)
    reset_child = bool(child_xf) and child_xf.GetResetXformStack()
    if reset_child:
        return child_world, reset_child
    parent_world = cache.GetLocalToWorldTransform(target_parent) if target_parent else Gf.Matrix4d(1)
    return child_world * parent_world.GetInverse(), reset_child


def _apply_new_transform(prim: Usd.Prim, matrix: Gf.Matrix4d, reset_stack: bool) -> None:
    """Replace xform op order with a single baked matrix op."""
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return
    xformable.SetXformOpOrder([])
    new_op = xformable.AddTransformOp(opSuffix="flatten")
    xformable.SetXformOpOrder([new_op])
    xformable.SetResetXformStack(reset_stack)
    new_op.Set(matrix)


def _format_matrix_summary(matrix: Gf.Matrix4d) -> str:
    """Human-readable xyz + scale summary for logs."""
    tfm = Gf.Transform(matrix)
    t = tfm.GetTranslation()
    s = tfm.GetScale()
    return f"t={tuple(round(v, 4) for v in t)}, s={tuple(round(v, 4) for v in s)}"


def collapse_redundant_transforms(
    stage: Usd.Stage,
    root_path: str,
    tmp_name: str,
    max_rounds: int = 8,
    dry_run: bool = False,
) -> List[Dict[str, str]]:
    """Iteratively collapse redundant xforms under root; returns log of actions."""
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise SystemExit(f"Root prim not found: {root_path}")
    actions: List[Dict[str, str]] = []

    for _ in range(max_rounds):
        changed = False
        prims = [p for p in Usd.PrimRange(root, PREDICATE)]
        prims.sort(key=lambda p: len(str(p.GetPath())), reverse=True)

        for prim in prims:
            if prim == root:
                continue
            if not _is_collapsible_transform(prim):
                continue
            parent = prim.GetParent()
            if parent is None:
                continue
            child = [c for c in prim.GetChildren() if c.IsActive()][0]
            temp_path = _unique_temp_path(stage, parent.GetPath(), tmp_name)
            final_path = parent.GetPath().AppendChild(prim.GetName())

            baked_matrix, child_reset = _compute_new_local(child, parent)
            matrix_summary = _format_matrix_summary(baked_matrix)

            if not dry_run:
                if not _move_prim(stage, child.GetPath(), temp_path):
                    print(f"Skip: move failed {child.GetPath()} -> {temp_path}")
                    continue
                moved = stage.GetPrimAtPath(temp_path)
                _apply_new_transform(moved, baked_matrix, child_reset)
                stage.RemovePrim(prim.GetPath())
                if not _move_prim(stage, temp_path, final_path):
                    print(f"Skip: rename failed {temp_path} -> {final_path}")
                    continue

            actions.append(
                {
                    "removed": str(prim.GetPath()),
                    "child": str(child.GetPath()),
                    "final": str(final_path),
                    "matrix": matrix_summary,
                    "reset": str(child_reset),
                }
            )
            changed = True

        if not changed:
            break

    return actions


def main():
    parser = argparse.ArgumentParser(description="Flatten redundant xform chains while preserving world transforms.")
    parser.add_argument("--stage", required=True, help="Path to USD stage to edit.")
    parser.add_argument("--root", default="/World", help="Root prim to process.")
    parser.add_argument("--tmp-name", default="tmp_flatten", help="Temporary name used during moves.")
    parser.add_argument("--output", help="Path to write a log of collapsed prims.")
    parser.add_argument("--output-usd", help="Optional path to save modified stage (input is left untouched).")
    parser.add_argument("--tree-output", help="Optional path to write the resulting prim tree.")
    parser.add_argument("--max-depth", type=int, default=6, help="Depth for the optional tree output.")
    parser.add_argument("--max-rounds", type=int, default=8, help="Safety limit on collapse passes.")
    parser.add_argument("--dry-run", action="store_true", help="Plan operations without writing USD changes.")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")
    stage.SetEditTarget(stage.GetRootLayer())

    actions = collapse_redundant_transforms(
        stage,
        root_path=args.root,
        tmp_name=args.tmp_name,
        max_rounds=args.max_rounds,
        dry_run=args.dry_run,
    )

    summary_lines = [
        f"Collapsed {len(actions)} redundant xform(s) under {args.root} (dry_run={args.dry_run})."
    ]
    for act in actions:
        summary_lines.append(
            f"- removed {act['removed']} | child {act['child']} -> {act['final']} | {act['matrix']} | reset={act['reset']}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(summary_lines), encoding="utf-8")
        print(f"Wrote collapse log to: {out_path.resolve()}")
    else:
        print("\n".join(summary_lines))

    if not args.dry_run:
        if args.output_usd:
            out_usd = Path(args.output_usd)
            out_usd.parent.mkdir(parents=True, exist_ok=True)
            stage.GetRootLayer().Export(str(out_usd))
            print(f"Saved modified stage to: {out_usd.resolve()}")
        else:
            stage.GetRootLayer().Save()

    if args.tree_output:
        tree_lines = list(iter_tree_lines(stage, args.root, args.max_depth))
        tree_path = Path(args.tree_output)
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_path.write_text("\n".join(tree_lines), encoding="utf-8")
        print(f"Wrote resulting tree to: {tree_path.resolve()}")


if __name__ == "__main__":
    main()
