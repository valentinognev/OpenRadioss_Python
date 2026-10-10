# Port extensions — code with no upstream counterpart

*This is the single registry of code in `pyradioss/` that has **no** open-source
OpenRadioss Fortran counterpart to transcribe from.* Anything listed here is
**not a port**: it was written in this repository, from the element's or
keyword's definition rather than from a Fortran routine, and it carries no
parity claim against `$OR_SRC`.

The inverse is what matters as much as the entry: **if a module is not in this
file, it is a port and it must cite a Fortran file that exists.** A citation
naming a file absent from `$OR_SRC` is worse than no citation at all — it lends
the authority of a transcription nobody performed to a routine that could never
be compared. `tests/test_p2_tria3_provenance.py` enforces that inverse across
the whole `pyradioss/` tree.

## How to read an entry

| Field | Meaning |
|---|---|
| **Status** | `declared port extension` — code written here; or `refused` — input accepted and explicitly rejected, with no kernel; or `ported-unreachable` — a real transcription of an upstream file that no deck `pyradioss` reads can reach (the one entry of this kind is the **exception** to the registry's rule above: it *is* a port, and it is listed here because "ported" would over-claim). |
| **Upstream search** | The commands run against `$OR_SRC`, and what they returned. Dates are the day they were run. |
| **Honest citation** | What may be written in the module docstring instead of a file path. |
| **Kept honest by** | The test that fails if this entry is deleted or the module drifts back to claiming a port. |

---

## `pyradioss/elements/solid_tria3.py` — /TRIA3 2-D solid triangle

**Status:** declared port extension.

**Upstream search** (2026-10-09, against `$OR_SRC`, read-only):

```console
$ ls engine/source/elements/solid_2d/
quad  quad4
```

The plan's 2026-10-02 note — that `engine/source/elements/solid_2d/` contains
only `quad/` and `quad4/` — is confirmed. There is **no** engine-side `tria/`
directory, and no routine named `t3forc2.F`, `t3deri2.F`, `t3rota2.F`,
`t3dlen2.F` or `t3mass2.F` anywhere in the tree.

The one place the name `tria` does occur is worth stating precisely, because it
is the tempting false positive:

```console
$ ls starter/source/elements/solid_2d/tria/
t3grhead.F  t3grtails.F
```

That directory exists, and it holds **only the two input-deck mesh readers**.
`starter/source/elements/solid_2d/tria/t3grhead.F:22-30` is called by
`starter/source/starter/lectur.F` and reads element groups out of the deck —
`t3grtails.F` likewise. Neither computes area, stress, forces or a time step.
There is no mass routine there either, so the `t3mass2.F` the module used to
cite was never real in either tree.

So /TRIA3 is a **declared element of the input format whose kernel ships only
in the closed-source tree**: the starter can read such a group, and the
open-source engine has no routine to advance it. That is the plan's Step 3
decision order, branch 1 — the element exists in closed-source Simcenter
Radioss — so the module is a declared port extension and the honest citation is
*Simcenter Radioss (closed source)*.

**Honest citation:** `Simcenter Radioss (closed source)`. The module docstring
now says this, says the module is a port extension, and records the upstream
search above. No `t3*.F` path appears in it any more.

**What was removed.** The fabricated citations, all of which pointed at files
that do not exist:

- the module docstring's `engine/source/elements/solid_2d/tria/` origin block,
  naming `t3forc2.F`, `t3deri2.F`, `t3rota2.F` and `t3dlen2.F`;
- the section header above `_edofs` (driver / derivatives / mass);
- the `Fortran origin:` lines in `tangent`, `kgeo` and `consistent_mass`, the
  last of which named `starter/source/elements/solid_2d/tria/t3mass2.F`.

Each now reads *Port extension — no upstream counterpart*.

**What did not change.** No physics. The kinematics, the axisymmetric hoop
terms, the bulk viscosity, the Courant time step and the implicit matrices are
byte-identical; this task only relabels provenance. Nothing here may be
presented as Fortran-faithful, and no parity run against the oracle is possible
or claimed for /TRIA3 — there is no oracle code to compare against.

**Kept honest by:** `tests/test_p2_tria3_provenance.py` —
`test_no_new_fabricated_citations` (no citation anywhere in `pyradioss/` names
an absent file), `test_solid_tria3_cites_no_absent_file` and
`test_solid_tria3_cites_no_tria_kernel` (no `t3*.F` comes back),
`test_port_extensions_registry_declares_tria3` (this entry stays listed) and
`test_solid_tria3_docstring_says_it_is_an_extension`.

---

## `pyradioss/elements/solid_q1np.py` — Q1NP hexahedra on a B-spline surface

**Status:** `ported-unreachable` (for `q1np_forc3.F90`). This is **not** a port
extension: every routine in the module is transcribed from a named upstream
file. It is registered here because the plan's Step 3 asks for the reason to be
recorded and because `ported` would claim something false.

**What is ported, and what that does and does not mean.**

| Upstream file | Census status | Python |
|---|---|---|
| `engine/source/elements/solid/solid_q1np/q1np_nurbs_surface_eval_mod.F90` | not promoted (stays `missing` in the allowlist; no parity run exists) | `evaluate_surface`, `evaluate_top_surface_point[_and_derivs]`, `evaluate_shape_values` — plain functions, tested directly |
| `engine/source/elements/solid/solid_q1np/q1np_forc3.F90` | **`ported-unreachable`** (`tools/validation_data/port_status.json`) | `forces` and the Gauss-point helpers; **no caller anywhere in `pyradioss/`** |
| `common_source/modules/q1np_geom_mod.F90`, `q1np_restart_mod.F90` | not promoted | the basis / shape-function / Gauss-scheme routines the two files above call |
| `engine/source/elements/solid/solid_q1np/q1np_dump_hist_state.F90` | `missing` | not ported — a debug CSV dump of node coordinates |

**Why the force path is unreachable — checked against `$OR_SRC`, 2026-10-10.**
The plan's wording ("the open-source deck format has no syntax for a NURBS
surface", "the projection path (`q1np_forc3.F90`)") is half right, and the
half that is wrong matters:

1. *`q1np_forc3.F90` contains no projection.* `rg -i "project|newton"` over
   `engine/source/elements/solid/solid_q1np/` matches only a *comment* in
   `q1np_nurbs_surface_eval_mod.F90` ("Used by Newton projection"). The file is
   the element force driver: Gauss loop, Jacobian, strain rate, material call,
   nodal force accumulation. The Newton projection is
   `q1np_contact_project_point_newton` in
   `engine/source/interfaces/ists_q1np/q1np_contact_algorithms.F90`, which is
   **not** ported by this task.
2. *A deck can reach the Fortran kernel; it cannot reach this port.* There is
   indeed no NURBS keyword. But `starter/source/starter/lectur.F` (≈ line 3328)
   calls `Q1NP_GENERATE_MAIN`
   (`starter/source/elements/solid/solid_q1np/q1np_generate_main.F90`), which
   walks the `/INTER` cards, reads the `Ists` flag and the two `/SURF` ids they
   name, fits a B-spline top surface to the mesh surface
   (`q1np_genelements.F90`, `q1np_cholesky.F90`, `q1np_setupnurbs.F90`) and
   fills `KQ1NP_TAB` / `Q1NP_CPTAB` / `Q1NP_KTAB`, which the Engine reads. So
   the surface is *derived from a mesh `/SURF`*, not specified. `pyradioss`
   ports none of that Starter chain (≈ 6,000 lines under
   `starter/source/elements/solid/solid_q1np/` plus the `/INTER` `Ists` reader)
   and never builds a Q1NP element, so **no `pyradioss` deck produces a call to
   `forces`.** The prototype generator also hard-codes surface ids (`lectur.F`:
   "Prototype mode: generate Q1NP for external /SURF IDs 100 and 200").

**What was deliberately not done.** No deck, no keyword reader, no `/INTER`
`Ists` hook and no material lookup was invented to make `forces` look
exercised. `forces` requires a caller-supplied `stress_fn`; upstream's stress
comes from `MMAIN` over the element buffer (`Q1NP_GP_MAT`), which is not
ported. Its unit tests check the *transcription* (volume, equilibrium, the
isoparametric identity `sum_k F_k x_k^T = -V sigma`) on a hand-built element;
they say nothing about reachability and the test file says so.

**Not ported from `q1np_forc3.F90`:** `Q1NP_GP_MAT` (`SROTA3`, `SRHO3`,
`MMAIN`, `SSTRA3`), `Q1NP_INIT_NODE_MAP` / `Q1NP_GET_BULK_NODE_IDS` /
`Q1NP_REBUILD_BULK_*` (need `KQ1NP_TAB`), `SMALLA3` / `SMALLB3`,
`Q1NP_AVG_SIG_BILAN` (and its disabled `IGE3DBILAN`), and the
`IDTMIN(101)` `MASS_ELEM` accumulation, which upstream writes and never reads.
`ported-unreachable` is therefore also *partial*: the row means "the
geometry-and-force core is transcribed; nothing can call it".

**Numerical notes recorded so nobody "fixes" them.** The upstream surface is a
plain B-spline — `Q1NP_WTAB` exists but no evaluated routine uses a weight;
`Q1NP_CHAR_LEN` scales by `0.2 × (sum of four edges)`, not the mean `0.25`,
with `0.2` a default-real literal (the module keeps the single-precision
value); `Q1NP_GAUSS_1D` falls back to a uniform grid above five points, which
is not Gauss-Legendre.

**Parity:** none, and none is possible from a deck. The oracle's Q1NP path is
only entered through the `/INTER` `Ists` prototype above.

**To make it reachable** (a later task, not this one): port
`q1np_generate_main` and the fit chain, the `Ists` `/INTER` reader, the
`KQ1NP_TAB` family, and wire `forces` into the engine's group loop with the real
material call. Only then may the row move to `ported`, and only with a parity
run against an `Ists` deck on the oracle.

**Kept honest by:** `tests/test_p2_solid_q1np.py` —
`test_nothing_in_pyradioss_references_solid_q1np`,
`test_forc3_is_marked_ported_unreachable_in_the_allowlist_and_the_census`,
`test_port_extensions_registry_records_q1np_unreachability`,
`test_forces_has_no_default_material`.

---

## Known misses outside this task's scope

Generalising the citation scan from `pyradioss/elements/` to the whole
`pyradioss/` tree — which the plan's Step 4 asks for — found **56 citations
naming absent files across 23 further modules**, plus the one settled above.
They are enumerated in that test's `KNOWN_MISSES` set and deliberately **not**
fixed here: each needs its own decision, because in many cases the routine does
exist under a *different* path and the citation is merely spelled wrong. That
distinction is the whole question — a misspelling is a one-line fix, a genuinely
absent kernel is a `docs/PORT_EXTENSIONS.md` entry — and it is not this task's
to make for 23 modules.

Two sub-classes are worth recording, both checked against `$OR_SRC`:

- **Misspelled path; the routine exists elsewhere.** `shell/ccoor3.F`,
  `shell/cdefo3.F`, `shell/chvis3.F`, `shell/czforc3.F` are all under
  `engine/source/elements/shell/coque/`; `solid/sdefo3.F` is under
  `engine/source/elements/solid/solide/` (and in the starter);
  `spring/rmass.F` is under `starter/source/elements/spring/`.
- **Unverified.** The remainder — `beam/pmass3.F`, `beam/bsigini.F`,
  `ige3d/ig3dinit3.F`, `reader/hm_read_node.F`, `reader/hm_read_rivet.F`,
  `solid/sinit3.F`, `solid/sbulk3.F`, `solid/sdlen3.F`, `solid/sfint3.F`,
  `solid/shour3.F`, `solid/srcoor3.F`, `solid/srota3.F`, `solid/solide/smass3.F`,
  `solid/tetra10/t10*.F`, `spring/r3buf3.F`, `spring/r4buf3.F`,
  `spring/redef3.F`, `spring/rinit3.F`, `spring/rini35.F`, `spring/rmass3.F`,
  `spring/preload_axial.F`, `truss/tmass3.F`, `truss/tsigini.F`,
  `xelem/xini28.F`, `joint/rjoint/rini33.F`, `joint/rjoint/rini45.F`,
  `initia/lec_inistate.F`, `initia/hm_read_inistate_d00.F`,
  `inibri_eref.F`, `shell/coque/corthdir.F`, `shell/coque/lcgeo19.F`,
  `sh3n/coquedk6/cdk6mass3.F`, `solid_2d/quad4/q4mass2.F` — one per path, some
  cited from both the engine and the starter tree.

**This list is ratcheted, not waived.** A *new* fabricated citation still turns
the suite red (`test_no_new_fabricated_citations`), and repairing an entry
without removing it from `KNOWN_MISSES` also fails
(`test_known_misses_are_still_absent_upstream`) — so the allowlist cannot be
left to rot into a permanent exemption.

Affected modules, for whoever picks this up:
`pyradioss/accel/gpu_kernels/__init__.py`, `elements/beam_fiber.py`,
`elements/iga3d.py`, `elements/nstrand.py`, `elements/rivet.py`,
`elements/shell_dkt6.py`, `elements/shell_ortho.py`,
`elements/solid_hexa8z.py`, `elements/solid_pyra5.py`,
`elements/solid_quad4_full.py`, `elements/solid_tshell8.py`,
`elements/spring_advanced.py`, `elements/spring_beam.py`,
`elements/spring_mat.py`, `elements/thickshell_composite.py`,
`elements/thickshell_wedge6.py`, `elements/truss.py`, `engine/bolt_preload.py`,
`engine/kjoint.py`, `input/keywords/geometry.py`, `model/entities/misc.py`,
`model/stack.py`, `starter/inista.py`, `starter/initemp.py`.