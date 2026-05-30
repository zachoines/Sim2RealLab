# Scene-provider contract

The teleop harness is **scene-source-agnostic at runtime**. The
capture driver ([`teleop_capture.py`](../source/strafer_lab/scripts/teleop_capture.py)),
the mission picker
([`teleop_mission_picker.py`](../source/strafer_lab/strafer_lab/tools/teleop_mission_picker.py)),
and the dataset writer
([`lerobot_writer.py`](../source/strafer_lab/strafer_lab/tools/lerobot_writer.py))
never import Infinigen. They consume three on-disk artifacts. Any
source that produces those artifacts in the shape documented here is
consumed with **no code changes**.

This is the single source of truth for that shape. Infinigen is one
provider — its build-time scripts
([`prep_room_usds.py`](../source/strafer_lab/scripts/prep_room_usds.py),
[`extract_scene_metadata.py`](../source/strafer_lab/scripts/extract_scene_metadata.py),
[`generate_scenes_metadata.py`](../source/strafer_lab/scripts/generate_scenes_metadata.py),
[`postprocess_scene_usd.py`](../source/strafer_lab/scripts/postprocess_scene_usd.py))
produce conformant artifacts. A second source (a downloaded scene
pack, a hand-authored map, a ProcTHOR / Habitat / Cosmos export)
needs its own metadata authoring but reuses everything downstream.

**The boundary is artifact-based, not interface-based.** There is no
`SceneSourceAdapter` base class and no Python plugin system. You do
not subclass anything or register anything. You write files. The
contract is enforced by a round-trip test
([`test_scene_provider_contract.py`](../source/strafer_lab/tests/harness/test_scene_provider_contract.py)),
not by static typing.

---

## a. On-disk layout

Everything lives under the scenes root,
`Assets/generated/scenes/` (the runtime constant `SCENE_USD_DIR`).
One **scene name** — `<scene>` below — keys the whole bundle.

```
Assets/generated/scenes/
  <scene>/                                  scene directory (name == <scene>)
    scene_metadata.json                     per-scene labeled objects[] + rooms[]   (REQUIRED)
    scene_config.json                       provenance (the preset / source dict)   (optional)
    <anything the source needs>/            export trees, textures, .blend, ...     (source-specific)
  <scene>.usdc                              symlink at the scenes root → the real USDC (REQUIRED)
  scenes_metadata.json                      combined manifest across all scenes      (REQUIRED)
```

**The `<scene>` naming invariant.** One string is reused four ways
and they must all match:

1. the scene **directory** name — `Assets/generated/scenes/<scene>/`,
2. the top-level **symlink** stem — `<scene>.usdc`,
3. the **key** under `scenes_metadata.json` → `scenes.<scene>`,
4. the `scene_id` stamped on every captured episode (the capture
   driver passes `--scene <scene>` straight through to
   `StraferLeRobotWriter.begin_episode(scene_id=...)`).

The runtime discovers a scene by listing `*.usdc` files at the scenes
root, then filtering to stems present as keys in
`scenes_metadata.json`
([`strafer_env_cfg._get_scene_usd_paths`](../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)).
A scene whose stem is missing from the manifest is invisible to the
runtime, even if its USDC and sidecar exist.

**The symlink.** `<scene>.usdc` is a symlink so the textures /
export tree stay bundled inside the scene directory while the scenes
root holds only flat, discoverable `<scene>.usdc` entries. The link
target can live anywhere under the scene directory; for Infinigen it
points at
`<scene>/export/export_scene.blend/export_scene.usdc`. Resolution of
the sidecar from a USD path (the `--scene-usd` override case) is
handled by
[`scene_paths.resolve_scene_metadata_path`](../source/strafer_lab/strafer_lab/tools/scene_paths.py);
the default `--scene <scene>` path reads
`Assets/generated/scenes/<scene>/scene_metadata.json` directly.

---

## b. `scene_metadata.json` schema

Produced per scene by `extract_scene_metadata.py` (Infinigen) — the
one file a second source must author itself. Consumed by the picker
and (hashed) by the writer.

Top-level object:

| Field | Req? | Type | Semantics |
|---|---|---|---|
| `objects` | **required** | array of object entries (below) | The pickable targets. May be empty, but then the picker offers nothing. |
| `rooms` | optional | array of room entries (§d) | Room polygons + types. Empty `[]` is valid and common (USD-only extraction can't recover rooms). |
| `room_adjacency` | optional | array of `[i, j]` int pairs | Edge list over `rooms[]` indices. Empty `[]` when `rooms` is empty. Not read by the teleop picker today; reserved for room-graph consumers. |
| `source` | optional | string | Provenance tag, e.g. `"usd_prim_names"`. Informational only; no consumer branches on it. |

### `objects[]` entry

Each entry describes one labeled object. Field-by-field:

| Field | Req? | Type | Units | Semantic guarantee |
|---|---|---|---|---|
| `label` | **required** | string | — | Canonical object category, lowercase (`"chair"`, `"bottle"`, `"bowl"`). The picker normalizes with `strip().lower()` and renders `mission_text = "go to the {label}"`. An empty / missing label drops the entry. Coarse by design — many physical instances share one label (the disambiguation problem `mission-text-enrichment` addresses). |
| `instance_id` | **required** | int | — | A numeric identifier. **Not guaranteed unique per object.** For Infinigen it is the factory-class ID embedded in the prim name, so sibling instances of one factory share it; plain / architecture prims use `-1`. The picker's de-dup key is the `__spawn_asset_<N>_` token in `prim_path`, falling back to the tuple `(label, instance_id, prim_path)` when that token is absent. Stamped on episodes as `target_object_id = str(instance_id)`. Deterministic for a given seed; **not stable across scene seeds.** |
| `position_3d` | **required** | `[x, y, z]` float | meters | Object center in the **world frame** (XYZ). **`[0, 0, 0]` is the sentinel for *no valid position*.** The extractor's `_drop_origin_records` helper drops any entry within `1e-3` m of the origin at author time (Infinigen seeds un-placed creature prims and bbox-fallback rows there). Downstream consumers may assume every surviving entry has a valid, non-origin position. An entry shorter than 3 elements is dropped by the picker. |
| `prim_path` | recommended | string \| null | — | Absolute USD prim path, e.g. `/World/ChairFactory_991132__spawn_asset_0_`. Used for (1) picker de-dup via the `__spawn_asset_<N>_` token and (2) USD prim labelling. `null` is tolerated (de-dup falls back to the `(label, instance_id, prim_path)` tuple), but a source that omits it loses per-physical-instance de-dup. |
| `bbox_3d_min` | optional | `[x, y, z]` float | meters | World-frame, **axis-aligned** (AABB) minimum corner. **Not** an oriented bounding box (OBB). Defaults to `[0,0,0]` when unresolvable. |
| `bbox_3d_max` | optional | `[x, y, z]` float | meters | World-frame AABB maximum corner. Pairs with `bbox_3d_min`. |
| `room_idx` | optional | int \| null | — | Index into the top-level `rooms[]` array, or `null` when room membership is unknown (always `null` for USD-only scenes). The picker resolves it to a display-only `room_type` via `rooms[room_idx].room_type`. |
| `semantic_tags` | optional | array of string | — | Infinigen `Semantics` enum values / factory class names. Informational; not required by the picker. |
| `relations` | optional | array of `{type, target}` | — | Spatial relations (`SupportedBy`, `StableAgainst`, ...). `type` is the relation class name; `target` is the related object's name (or `null`). Empty `[]` for USD-only scenes. |
| `materials` | optional | array of string | — | Blender material slot names (e.g. `"VaseCeramic"`, `"shader_glass.001"`). Noisy — leaks the surface class. Empty `[]` for USD-only scenes. |

Minimum a second source must emit per object: `label`, `instance_id`,
`position_3d` (non-origin), and ideally `prim_path`. Everything else
degrades gracefully to its empty / null default.

---

## c. `scenes_metadata.json` schema

The combined manifest, one file at the scenes root, produced by
`generate_scenes_metadata.py` (Infinigen). Consumed by the env config
for scene discovery + robot spawn placement.

```json
{
  "scenes": {
    "<scene>": {
      "spawn_points_xy": [[x, y], ...],
      "floor_top_z": 0.148,
      "floor_bbox_xy": [[xmin, ymin], [xmax, ymax]],
      "source_usdc": "/abs/path/to/<scene>.usdc"
    }
  }
}
```

| Field | Req? | Type | Units | Semantic guarantee |
|---|---|---|---|---|
| `scenes` | **required** | object keyed by `<scene>` | — | Top-level map. Its **keys gate runtime discovery** — only stems present here are loaded (`_get_scene_usd_paths`). A key with an otherwise-empty value still makes the scene loadable, but spawn placement degrades (see below). |
| `scenes.<scene>.spawn_points_xy` | **core** | array of `[x, y]` float | meters | Interior floor positions clear of walls and obstacles. The env samples robot resets from these. An empty list falls back to `[[0, 0]]` (robot spawns at world origin — usually wrong, but non-fatal). |
| `scenes.<scene>.floor_top_z` | **core** | float | meters | Z of the floor's top surface. The ground collision plane is lifted to this so the robot rests on the floor instead of at world `z=0`. Omitting it makes the env fall back to its default floor height. |
| `scenes.<scene>.floor_bbox_xy` | informational | `[[xmin, ymin], [xmax, ymax]]` float | meters | Union XY bounds of all floor meshes. Produced for tooling / debugging; **not read by the runtime** today. |
| `scenes.<scene>.source_usdc` | informational | string (abs path) | — | Resolved absolute path of the scene USDC the manifest entry was derived from. Provenance only; **not read by the runtime** today. |

A second source whose floor prims are not named like Infinigen's
(`<room>_<i>_<j>_floor`) cannot use `generate_scenes_metadata.py`'s
auto-sampler — its floor-detection regex (`_FLOOR_NAME_RE`) and
obstacle heuristics are Infinigen-specific and **not** CLI-overridable
today (unlike the postprocess patterns in §f). Such a source must
hand-author its `scenes_metadata.json` entry (a list of clear interior
`[x, y]` points + the floor top Z). The worked example in §h shows
this.

---

## d. Extensibility — making the contract survive future briefs

The contract is designed to absorb new consumers without re-shipping.
Two consumers already exist on paper:
[`mission-text-enrichment`](tasks/parked/harness/mission-text-enrichment.md)
(parked, blocked on this brief) and
[`scene-metadata-in-usd`](tasks/parked/harness/scene-metadata-in-usd.md)
(parked, sibling). The rules below were written so they plug in
cleanly.

### Additive-fields policy

Producers **MAY** emit fields beyond the ones specified here — inside
`objects[]` entries, inside `rooms[]` entries, and at the
`scenes_metadata.json` top level. Consumers **MUST** treat any
unrecognized field as opaque: read past it, do not error, do not log
spam. This is binding on both sides. It is what lets a new descriptor
or annotation ride into the JSON without touching the picker or the
writer. The round-trip test pins this: a `scene_metadata.json`
carrying unknown fields must flow through the picker and writer
unchanged, and the unknown fields must not leak into the writer's
sidecar parquet.

### Reserved extension namespaces

A namespace is "reserved" when a brief has declared intent to emit it
and consumers agree on its shape in advance, so the producer and
consumer can land in either order.

**`objects[].descriptors`** — reserved by `mission-text-enrichment`
(parked). A per-object sub-dict of physical descriptors the
disambiguator will bind referring expressions to. Expected shape when
emitted:

| Key | Type | Domain |
|---|---|---|
| `descriptors.color_name` | string | one of the 11 basic colors: `red, orange, yellow, green, blue, purple, pink, brown, white, gray, black` |
| `descriptors.color_hsv` | `[h, s, v]` float | HSV, each component in `0..1` |
| `descriptors.material` | string | one of 8 categories: `wood, metal, ceramic, glass, plastic, fabric, plaster, marble` |
| `descriptors.material_subclass` | string | most-specific surface class verbatim, e.g. `"WhitePlywood"`, `"Marble"` |
| `descriptors.size_bucket` | string | `small` \| `medium` \| `large` |

**This reserves the name; it does not implement the fields.** No
producer emits `descriptors` today, and this brief does not add one.
The contract's promise is narrow: *if a source emits
`descriptors.color_name`, the future disambiguator will use it.*
Actually producing the fields is `mission-text-enrichment`'s v2 scope.

This is the **only** reserved namespace. Bias is toward *not*
over-reserving — a name reserved with no consumer is just clutter that
future authors have to reason around. A future brief that needs a new
namespace adds a row to the table in §d-last, in its own PR.

### `rooms[]` as a first-class but optional artifact

`rooms[]` is part of the schema but **optional**, and is empty (`[]`)
for every scene produced by USD-only extraction today (USD geometry
alone cannot recover the constraint solver's room polygons —
populating it is
[`infinigen-scene-corpus`](tasks/active/harness/infinigen-scene-corpus.md)'s
scope, not this brief's). Each entry, when present:

| Field | Type | Units | Semantics |
|---|---|---|---|
| `footprint_xy` | array of `[x, y]` float | meters | The room's floor polygon as an ordered vertex ring (the duplicated closing vertex is dropped). The parked briefs refer to this informally as the room "polygon" — same artifact, the on-disk key is `footprint_xy`. |
| `room_type` | string | — | e.g. `"kitchen"`, `"bathroom"`. Resolved by the picker for display via `objects[].room_idx`. |
| `area_m2` | float | m² | Floor area. |
| `story` | int | — | Floor/storey index (0 = ground). |

The room adjacency graph lives at the **top-level** `room_adjacency`
edge list (the parked briefs call this the "room_graph"), not inside
each `rooms[]` entry.

**Absence-handling contract.** Consumers MUST handle `rooms == []`
without crashing — fall back to single-room semantics. Concretely: the
picker shows no `room_type` suffix; the `mission-text-enrichment`
disambiguator's room-scoping qualifier **degrades to a no-op** (it
contributes nothing and the waterfall proceeds to the next qualifier).
A consumer that requires rooms to function is non-conformant.

### Versioning policy

**Soft-extension only.** Additive changes — new optional fields, new
reserved namespaces, new room fields — never require a version bump
and never break a consumer (per the additive-fields policy). Removing
or renaming a field requires a revision of *this document* (rare, and
a real interface break).

There is **deliberately no `scene_metadata_version` field** and no
versioned schema. Versioning is interface-talk; this contract is
artifact-based. A consumer that finds a field uses it; one that does
not, ignores it. That property — not a version handshake — is what
makes producers and consumers shippable in either order. (Note: the
sibling `scene-metadata-in-usd` brief muses about adding a
`strafer_scene_metadata_version` when it moves storage into USD
`customData`; that is a storage-layer concern for that brief to decide,
and does not change this artifact contract.)

### Known future extensions

Briefs that have reserved a namespace or field set. A future brief
that reserves something appends a row here in its own PR — this is the
ledger that prevents the "we forgot about future X" failure mode.

| Brief | Namespace / fields | Default when absent |
|---|---|---|
| `mission-text-enrichment` v2 | `objects[].descriptors.*` (`color_name`, `color_hsv`, `material`, `material_subclass`, `size_bucket`) | Disambiguator falls back to spatial + conjunctive language only |
| `mission-text-enrichment` v1 | `rooms[]` populated | Disambiguator skips the room-scoping qualifier (no-op) |

---

## e. USD prim expectations

What the runtime and the postprocess scripts traverse inside the
`<scene>.usdc`. A source whose prims differ overrides the patterns in
§f or authors the equivalent state directly.

- **Collision-API-applied meshes.** The runtime expects furniture /
  structural meshes to carry `UsdPhysics.CollisionAPI` +
  `MeshCollisionAPI` so the robot collides with them. `postprocess_scene_usd.py`
  attaches these once at build time (attaching them at Kit startup
  freezes the env for ~60 s on a multi-room house). Furniture gets
  `convexHull`; structural prims get `meshSimplification`.
- **Floor strip pattern.** Floor meshes are *excluded* from the
  collision pass — their tessellated triangle edges catch the mecanum
  rollers. Robot–floor collision is delegated to the clean
  `/World/ground` plane lifted to `floor_top_z`. Default floor match:
  `^/World/[^/]+_floor(?:/[^/]+_floor)?$` (the Infinigen
  Xform-with-same-named-Mesh-child pattern).
- **Structural prims for hybrid collider dispatch.** Walls, ceilings,
  roofs, attics, exteriors, and the two-prim door pair
  (frame + `__001` leaf) need `meshSimplification` rather than a
  convex hull, which would heal door / window cutouts and trap the
  robot in doorways. Default structural match:
  `^/World/[^/]+_(?:wall|ceiling|roof|attic|exterior)(?:_\d+)?(?:/.+)?$`
  plus the `(?:PanelDoor|LiteDoor|LouverDoor)Factory_\d+__spawn_asset_\d+_`
  door pattern.
- **Ceiling-light prims.** Infinigen exports the fixture mesh but no
  emitter, so interiors render black. `postprocess_scene_usd.py`
  authors a `UsdLux.SphereLight` under every prim whose **name** (the
  leaf, not the full path) matches `CeilingLightFactory_\d+__spawn_asset_\d+_`.
  Note the asymmetry: floor and structural patterns match the **full
  prim path**; the ceiling-light pattern matches the **prim name**.
- **Labelled prims for Replicator.** `extract_scene_metadata.py` can
  stamp `semanticLabel` / `instanceId` attributes on prims so
  Replicator's bbox annotators emit labeled boxes. Optional for
  teleop; required only for perception-side capture.

---

## f. Postprocess CLI surface

Every regex knob on `postprocess_scene_usd.py`, its default, what it
matches, and how a second source overrides it. All are matched with
`re.match` (anchored at the start).

| Flag | Default | What it matches | Override for a second source |
|---|---|---|---|
| `--floor-prim-pattern` | `^/World/[^/]+_floor(?:/[^/]+_floor)?$` | Infinigen floor Xform + same-named Mesh child (full path) | `^/World/floor.*$` (example) |
| `--structural-prim-pattern` | `^/World/[^/]+_(?:wall\|ceiling\|roof\|attic\|exterior)(?:_\d+)?(?:/.+)?$\|...Door...` | Infinigen walls / ceilings / door frames (full path) | `^/World/(?:wall\|ceiling).*$` (example). Pass an empty string to disable structural dispatch. |
| `--ceiling-light-prim-pattern` | `^CeilingLightFactory_\d+__spawn_asset_\d+_$` | Infinigen-named light fixture prims (**prim name**, not path) | `^MyCeilingLight_\d+$` (example) |
| `--collider-approximation` | `convexHull` | (not a regex) PhysX shape for furniture | one of `boundingCube, boundingSphere, convexHull, convexDecomposition, meshSimplification, none` |
| `--structural-approximation` | `meshSimplification` | (not a regex) PhysX shape for structural prims | same choice set |
| `--light-intensity` | `100000.0` | (not a regex) per-emitter intensity | tune up if rooms are dim, down if blown out |
| `--keep-floor-colliders` | off | (debug flag) keep floor colliders | reintroduces the wheel-catching behavior; debugging only |

Each of the three pattern functions is importable and can be called
directly by an adapter that needs finer control than the CLI:
`attach_mesh_colliders(stage, floor_pattern=..., structural_pattern=...)`,
`strip_floor_colliders(stage, floor_pattern)`, and
`inject_ceiling_light_emitters(stage, intensity, ceiling_light_pattern=...)`.
A source whose lights need a different emitter type (e.g. a
`UsdLux.DiskLight` at floor-up positions) can author them itself and
skip `inject_ceiling_light_emitters` entirely — nothing downstream
depends on it.

---

## g. Adapter-writer's checklist

To add a new scene source **X**, ship these:

1. **Stage the USDC.** Place `X`'s `.usdc` under
   `Assets/generated/scenes/<scene>/<...>/` and create the top-level
   symlink `Assets/generated/scenes/<scene>.usdc` pointing at it.
   (Reusable: `prep_room_usds.py ingest` copies an external scene
   directory tree into place — it does not run extract or postprocess.)
2. **Author `scene_metadata.json`** under
   `Assets/generated/scenes/<scene>/` with a conformant `objects[]`
   array (§b). This is the **only** source-specific code — either a
   new `prep_<src>_usds.py` / `extract_<src>_metadata.py`, or hand
   authoring for a one-off. Emit at minimum `label` + `instance_id` +
   non-origin `position_3d` (+ `prim_path`) per object.
3. **Postprocess the USDC** with `postprocess_scene_usd.py`, overriding
   the floor / structural / ceiling-light patterns (§f) to match `X`'s
   prim naming. Reusable as-is.
4. **Author the `scenes_metadata.json` entry** — run
   `generate_scenes_metadata.py` if `X`'s floors are named like
   Infinigen's, otherwise hand-write the `spawn_points_xy` +
   `floor_top_z` entry (§c). Manifest authoring is reusable; only the
   floor-detection step is Infinigen-coupled.
5. **Capture** with `Scripts/capture.py --driver teleop
   --mission-source scene-metadata --scene <scene>`. Reusable as-is.

Steps 1, 3, 4, 5 are source-agnostic. Only step 2 (and the
hand-authored half of step 4) is source-specific.

---

## h. Worked examples

### h.1 — Hand-authored single-USD ingest

An operator drops one USDC in, hand-writes the minimum metadata, and
captures. Literally runnable from the repo root.

```bash
SCENE=my_scene_alpha

# 1. Stage the USDC + symlink it at the scenes root.
mkdir -p "Assets/generated/scenes/$SCENE/import"
cp ~/Downloads/scene_alpha.usdc "Assets/generated/scenes/$SCENE/import/scene_alpha.usdc"
ln -sf "$(realpath "Assets/generated/scenes/$SCENE/import/scene_alpha.usdc")" \
    "Assets/generated/scenes/$SCENE.usdc"

# 2. Hand-author scene_metadata.json — minimum required fields per §b.
cat > "Assets/generated/scenes/$SCENE/scene_metadata.json" <<'EOF'
{
  "objects": [
    {"label": "chair", "instance_id": 1, "position_3d": [2.5, 1.2, 0.5],
     "prim_path": "/World/Chair_1"},
    {"label": "table", "instance_id": 2, "position_3d": [3.1, 0.4, 0.7],
     "prim_path": "/World/Table_2"}
  ],
  "rooms": [],
  "room_adjacency": []
}
EOF

# 3. Postprocess with CLI overrides matching this USD's prim naming.
$ISAACLAB -p source/strafer_lab/scripts/postprocess_scene_usd.py \
    --usdc "Assets/generated/scenes/$SCENE.usdc" \
    --floor-prim-pattern '^/World/floor.*$' \
    --structural-prim-pattern '^/World/(?:wall|ceiling).*$' \
    --ceiling-light-prim-pattern '^MyCeilingLight_\d+$'

# 4. Author the combined manifest entry.
#    If this USD's floor prims are NOT named <room>_<i>_<j>_floor, the
#    auto-sampler finds nothing — hand-write the entry instead:
python - "$SCENE" <<'EOF'
import json, sys
from pathlib import Path
scene = sys.argv[1]
root = Path("Assets/generated/scenes")
manifest = root / "scenes_metadata.json"
data = json.loads(manifest.read_text()) if manifest.is_file() else {"scenes": {}}
data["scenes"][scene] = {
    "spawn_points_xy": [[2.0, 1.0], [0.5, 0.5]],   # clear interior points
    "floor_top_z": 0.0,
    "source_usdc": str((root / f"{scene}.usdc").resolve()),
}
manifest.write_text(json.dumps(data, indent=2))
print("wrote", manifest)
EOF
# (If the floors ARE Infinigen-named, replace step 4 with:
#  $ISAACLAB -p source/strafer_lab/scripts/generate_scenes_metadata.py \
#      --scenes-dir Assets/generated/scenes)

# 5. Capture.
RUN_ID=$(date +%Y%m%dT%H%M%S)
$ISAACLAB -p Scripts/capture.py \
    --driver teleop --mission-source scene-metadata \
    --scene "$SCENE" \
    --output "data/sim_in_the_loop/${SCENE}_${RUN_ID}"
```

### h.2 — Programmatic adapter sketch

What a `prep_<src>_usds.py` looks like for a second source. This is a
**skeleton**, not a complete second source — only the
source-specific extraction is new; everything downstream is reused.

```python
# source/strafer_lab/scripts/prep_acme_usds.py  (sketch)
import json
from pathlib import Path

SCENES_DIR = Path("Assets/generated/scenes")

def extract_acme_objects(acme_scene) -> list[dict]:
    """SOURCE-SPECIFIC: turn one ACME scene into conformant objects[].

    The only part you write. Map ACME's native object records to the
    scene_metadata.json objects[] shape (§b). Drop any object whose
    position is the world origin — the harness treats (0,0,0) as
    'no valid position'.
    """
    out = []
    for i, obj in enumerate(acme_scene.iter_objects()):
        x, y, z = obj.world_center          # meters, world frame
        if abs(x) < 1e-3 and abs(y) < 1e-3 and abs(z) < 1e-3:
            continue
        out.append({
            "label": obj.category.lower(),  # canonical, lowercase
            "instance_id": i,               # unique here, since ACME has no factory IDs
            "position_3d": [x, y, z],
            "prim_path": obj.usd_path,
            "bbox_3d_min": list(obj.aabb_min),
            "bbox_3d_max": list(obj.aabb_max),
        })
    return out

def prep_scene(acme_scene, scene_name: str) -> None:
    scene_dir = SCENES_DIR / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    # 1. (stage the USDC + symlink — omitted; see prep_room_usds.ingest)
    # 2. author the sidecar (the source-specific step)
    metadata = {"objects": extract_acme_objects(acme_scene),
                "rooms": [], "room_adjacency": []}
    (scene_dir / "scene_metadata.json").write_text(json.dumps(metadata, indent=2))
    # 3. postprocess + 4. manifest + 5. capture are reused as-is:
    #    postprocess_scene_usd.main(["--usdc", str(SCENES_DIR / f"{scene_name}.usdc"),
    #                                "--floor-prim-pattern", r"^/World/Floor.*$", ...])
    #    generate_scenes_metadata.main(["--scenes-dir", str(SCENES_DIR)])
```

---

## Related

- [`docs/HARNESS_DATA_CAPTURE.md`](HARNESS_DATA_CAPTURE.md) — operator
  workflow for capturing against a conformant scene.
- [`scene-metadata-in-usd`](tasks/parked/harness/scene-metadata-in-usd.md)
  — moves the `scene_metadata.json` payload into USD `customData`; same
  artifact contract, different storage backend. When it lands, this doc
  gains a reader-fallback-order note (sidecar JSON ↔ USD customData).
- [`mission-text-enrichment`](tasks/parked/harness/mission-text-enrichment.md)
  — the first consumer of the `descriptors` namespace + populated
  `rooms[]` reserved here.
</content>
</invoke>
