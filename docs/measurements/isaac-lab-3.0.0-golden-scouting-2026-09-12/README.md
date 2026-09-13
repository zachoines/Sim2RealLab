# Isaac Lab 3.0.0 — what the composition goldens do under the retarget — 2026-09-12

The boot-stall investigation ended with Isaac Sim 6.1.0.0 identified as the pin that fixes the
stall, and with a scoping estimate for moving the migration's target there. That estimate
turned on one unanswered question: whether the frozen composition goldens move the way they
moved for Stages 1 and 2 — additively, so that a named drop-set reproduces them — or whether
they move because the contract itself changed.

This prices that question. It does not start the leg, and nothing here recommends starting it.
The probe is Kit-free, needs no build and no GPU, and takes minutes.

The answer is worse than the Stage 1/2 precedent. Eight of the twenty-two variants do not
move at all — they **fail to construct**, on an Isaac Lab API that no longer exists. Of the
fourteen that do move, the movement includes removed field names and changed values at
unchanged paths, neither of which a drop-set can subtract. The existing drop-set does not
apply: none of the fourteen lands on the hash it lands on at the candidate pin.

## Verdicts

| item | result |
|---|---|
| Probe validated against the current pair | 22 of 22 frozen contract goldens reproduce |
| Probe validated against the candidate pin | 22 of 22 move, reproducing the movement count recorded for Stages 1 and 2 |
| `release/3.0.0` — variants that fail to construct | **8 of 22**, all the enrichment variants, on `SimulationCfg.render` |
| `release/3.0.0` — variants whose golden moves | 14 of 22 |
| Is the movement additive, as in Stages 1 and 2? | **No.** 2388 removed field paths and 534 changed values, alongside 3224 added |
| Does the Stage 1/2 drop-set apply? | **No.** 0 of the 14 movers hash as they do at the candidate pin |
| Cost implication for the retarget leg | The golden work is a contract migration, not a drop-set exercise |

## The probe, and why its two controls matter

The probe constructs each composed RL variant and recomputes its canonical contract hash. It
imports the serializer, the field lists and the variant table **from the contract test itself**
rather than reimplementing them, so the probe and the gate cannot drift apart. For every
variant it also keeps the full canonical preimage, which is what makes a moved hash
diagnosable rather than merely visible.

Two controls establish that the instrument reads correctly before it is pointed at anything new:

- Against the pair the tooling currently selects, **all 22 goldens reproduce**. An instrument
  that cannot reproduce the frozen values cannot be trusted to interpret new ones.
- Against the migration's candidate pin, **22 of 22 move** — independently reproducing the
  movement count already recorded for Stages 1 and 2 from a different direction.

The first run of this probe was mistakenly pointed at the candidate pin and reported all 22
moved, which looked like a broken harness and was in fact the harness correctly reporting a
known result. The two controls exist so that confusion is resolved by the data.

## Eight variants do not move — they do not build

```
RLDepthEnriched_Real                 AttributeError: 'SimulationCfg' object has no attribute 'render'
RLDepthEnriched_Real_PLAY            (same)
RLDepthEnriched_Robust               (same)
RLDepthEnriched_Robust_PLAY          (same)
RLDepthSubgoalEnriched_Real          (same)
RLDepthSubgoalEnriched_Real_PLAY     (same)
RLDepthSubgoalEnriched_Robust        (same)
RLDepthSubgoalEnriched_Robust_PLAY   (same)
```

Every enrichment variant. `RenderCfg` is gone upstream and `SimulationCfg` no longer carries a
`render` field, so the three shipped fixes that reach the renderer through it — the RTX
histogram auto-exposure merge, the ceiling back-face-culling merge, and the lightest-scene
bind — have lost their mechanism, and the cfgs that use them cannot be instantiated at all.
This is not a golden that needs re-freezing. It is code that needs migrating before the
question of goldens can even be asked, and it lands on fixes that were each measured and
validated separately.

A caution carried from the earlier follow-up record applies directly here: a settings path that
imports is not a settings path that takes effect. Whatever replaces `RenderCfg` has to be shown
to reach the same lifecycle point, by read-back, not by the absence of an import error.

## The fourteen that move, move for reasons a drop-set cannot absorb

Field-level classification of the 14 movers against the reproducing baseline, counting
distinct canonical paths:

| kind of difference | occurrences | distinct paths | subtractable by a drop-set? |
|---|---|---|---|
| field names **added** | 3224 | 2 (`commands`, `events`) | yes — this is the Stage 1/2 shape |
| field names **removed** | 2388 | 1 (`commands`) | **no** |
| values **changed** at an unchanged path | 534 | 2 (`commands`, `events`) | **no** |

Stages 1 and 2 succeeded because the movement was purely additive: a named drop-set
(`cmd_kind`, `element_names`) removed the added names and the goldens recomputed
byte-identically. That does not hold here. Removals and changed values both survive any
drop-set, because a drop-set can only decline to look at a name that is present.

The command manager carries nearly all of it — 2996 added paths, 2388 removed, 306 changed —
so the bulk of the work is one subsystem's contract, not a diffuse sweep.

And the movement is **new**, not the movement already characterised: none of the 14 lands on
the hash it lands on at the candidate pin. Whatever is learned about the candidate pin's
drop-set does not transfer.

## What this prices

The golden half of a retarget leg is not "re-freeze 22 hashes after subtracting two names."
It is: migrate the code behind eight variants that no longer build, then establish a new
contract baseline for a command manager that has been restructured, then decide — per variant —
whether a moved hash represents a policy-visible change or a serialization artefact. The last
of those is the expensive part, because a contract golden exists precisely to make that
distinction impossible to wave through, and a trained checkpoint depends on the answer.

Nothing here argues for or against the retarget. It replaces an unknown with a shape.

## Scope and limits

The `release/3.0.0` arm paired that worktree's Isaac Lab sources with the candidate pin's
interpreter rather than with Isaac Sim 6.1.0.0, because the 6.1.0.0 probe environment has no
pytest and installing into it mid-measurement would have disturbed a boot-ordering run in
progress. For this question that is the right isolation — a composition golden is a property of
the cfg tree, and the eight construction failures are an Isaac Lab API change that no isaacsim
version affects — but it does mean this record does not speak to anything that requires Isaac
Sim 6.1.0.0 to be the interpreter. Two consequences worth stating: a variant that builds here
is not thereby shown to build under 6.1.0.0, and the 14 hashes are not the values a future
re-freeze should adopt.

The probe covers the 22 composition contract goldens. It does not cover the layout goldens, the
palette signature, or the depth observation golden, each of which is pinned separately.

## Evidence

One deposit, in the private repository
`https://github.com/zachoines/Sim2RealLab-Artifacts`.

### `isaac-lab-3.0.0-golden-scouting-2026-09-12/` — commit `c67cd00d7d8930df7953ed3132f676d592e9e76f`

It holds the probe, the three runs — the reproducing baseline, the candidate pin, and the
`release/3.0.0` worktree at `3cd1279e63effc1b82b51f70ec3250cec10d21f5` — and the three-way
field-level classification printed from them. Each run carries the complete canonical preimage
of every variant, so any moved hash can be re-diffed without re-running anything; the three are
gzipped for size and `gunzip -c` restores the bytes the digests cover. The deposit's own
`DEPOSIT.md` says which is which. This record was README-only from the start and its evidence
never lived in the record directory, so there is no `record-files/` tree.

sha256 of every file, paths relative to that deposit directory:

```
afd6f5fe7ba9c6d04176babbd50b24698674091849fa50a66103bb609c40d691  probe/classification.txt
3a97c9206be15fda4bc01d9587e8a94e201f2d38b9e21c6ba315b88dee86bd28  probe/control-6.0.1.0.json.gz
82ec68a9ae99c6abbfc74d90e87782915fe7a5747da2fcbc2817489a99872cd8  probe/control-current-pair.json.gz
bf9bce6aa9650f5997c5b719575824d711f67962ea268e3b02792879e9418f8c  probe/golden_probe.py
8f51768ff796603a702c24926813d5bbbd4ab8b65d61f3b93c04c2ecc393468b  probe/rel300-via-beta2.json.gz
```
