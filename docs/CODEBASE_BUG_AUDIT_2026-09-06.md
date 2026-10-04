# Whole-codebase bug audit — 2026-09-06

Audited snapshot: `aa868fe1f5c28bd2aca19ed574ccac1e671ce8d8`
(`M488: contact penalty stiffness, gaps & Istf combination unit tests`).

## Result

This audit found **29 confirmed bug clusters**: **22 P1** defects that can
crash a claimed workflow or silently change a simulation, and **7 P2** defects
in optional output, restart, diagnostics, process handling, or project status.
Several clusters contain more than one affected keyword or execution path.

“Whole-codebase” here means every package module was included in the import,
compile, state-consumer, undefined-name, duplicate-test, and targeted source
passes. It cannot prove that no further numerical defect exists. The list below
contains the failures that could be tied to a concrete trigger and effect; weak
static-analysis leads were excluded.

Severity used in this report:

- **P1:** primary workflow crash, silently ignored simulation control, or
  materially wrong physics/result.
- **P2:** optional workflow failure, misleading output/status, indefinite wait,
  or a serious test/documentation gap.

## Verification performed (2026-09-06, this audit's snapshot)

- `.venv\Scripts\python.exe -m pytest -q -m "not slow"`:
  **5,498 passed, 4 skipped, 19 deselected** in 615.75 s.
- `.venv\Scripts\python.exe -m pytest -q -m slow`:
  **18 passed, 1 skipped, 5,502 deselected** in 1,255.89 s.
- `.venv\Scripts\python.exe -m compileall -q pyradioss tools`: passed.
- Import sweep: **142 package modules imported, 0 failed** in the fully
  provisioned project environment.
- Targeted checks covered the failing AMS, LAW36, shell, high-order topology,
  surface, VTK, parser, energy, restart, modal, random-response, and implicit
  dynamics branches.
- A clean collection found **5,502 selected / 5,521 total tests**. The current
  documented baseline still says 1,105 / 1,121.

The green suite and the findings are consistent. Most failures sit in branches
that current tests only parse, never execute, or do not cover at all.

## Confirmed findings

### AUD-001 — P1 — Many Engine controls are parsed and then ignored

`EngineControls` defines the fields in
[`model.py:153`](../pyradioss/model/model.py#L153), and the parser fills them in
[`engine_keywords.py:110`](../pyradioss/input/engine_keywords.py#L110). The
runtime loop, however, ends only on `/RUN`'s `t_end`
([`engine.py:484`](../pyradioss/engine/engine.py#L484)) and has no consumer for
many of those values.

Concrete no-op families include:

- `/STOP/NSTEP`, `/STOP/TSTOP`, and `/STOP/TIMET`, parsed at
  [`engine_keywords.py:1164`](../pyradioss/input/engine_keywords.py#L1164).
- Element `/DT/BRICK`, `/DT/SHELL`, `/DT/QUAD`, `/DT/TETRA10`, and `/DT/INTER`
  action controls, stored at
  [`engine_keywords.py:214`](../pyradioss/input/engine_keywords.py#L214).
- `/BCS/ON|OFF`, `/RBODY/ON|OFF`, `/INTER/ON|OFF`, and `/DEL/*`, stored at
  [`engine_keywords.py:1217`](../pyradioss/input/engine_keywords.py#L1217),
  [`engine_keywords.py:1231`](../pyradioss/input/engine_keywords.py#L1231),
  [`engine_keywords.py:1332`](../pyradioss/input/engine_keywords.py#L1332), and
  [`engine_keywords.py:1354`](../pyradioss/input/engine_keywords.py#L1354).
- `/DAMP`, `/MASS/RESET`, `/SENSOR/RESET`, and `/NEGVOL/STOP|DEL`, stored at
  [`engine_keywords.py:1464`](../pyradioss/input/engine_keywords.py#L1464),
  [`engine_keywords.py:1476`](../pyradioss/input/engine_keywords.py#L1476),
  [`engine_keywords.py:1482`](../pyradioss/input/engine_keywords.py#L1482), and
  [`engine_keywords.py:1547`](../pyradioss/input/engine_keywords.py#L1547).
- `/H3D`, `/NOIS`, `/FLOW`, Engine `/TH`, and `/ANIM/*/TENS` request controls,
  stored at [`engine_keywords.py:153`](../pyradioss/input/engine_keywords.py#L153),
  [`engine_keywords.py:1260`](../pyradioss/input/engine_keywords.py#L1260),
  [`engine_keywords.py:1281`](../pyradioss/input/engine_keywords.py#L1281),
  [`engine_keywords.py:1290`](../pyradioss/input/engine_keywords.py#L1290), and
  [`engine_keywords.py:1572`](../pyradioss/input/engine_keywords.py#L1572).

A conservative runtime-reference scan found **73 of 214** `EngineControls`
fields with no read outside parsing/model declaration. The full candidate list
is in Appendix A. A deck therefore appears accepted while its requested stop,
activation, deletion, damping, or output behavior never occurs.

### AUD-002 — P1 — `/DT/AMS` is unusable and contains three independent faults

The Engine calls `ams.tag_nodes(claims)` at
[`engine.py:408`](../pyradioss/engine/engine.py#L408) and again at
[`engine.py:568`](../pyradioss/engine/engine.py#L568), but
[`AMSManager`](../pyradioss/engine/ams.py#L9) has no `tag_nodes` method. Every
active `/DT/AMS` run reaches an `AttributeError` before the AMS solve.

Two additional faults are visible behind that first crash:

- Part groups resolve into `part_ids_resolved`, while AMS iterates
  `grpart.members` at [`ams.py:24`](../pyradioss/engine/ams.py#L24).
- The local `d_add` array is accumulated against `np.arange(n)` at
  [`ams.py:71`](../pyradioss/engine/ams.py#L71). When the AMS group omits the
  highest global node, its length is below `n`; a two-node group in a five-node
  model reproduced a broadcast `ValueError`.

### AUD-003 — P1 — The declared base installation cannot import Starter or Engine

[`pyproject.toml:12`](../pyproject.toml#L12) declares only NumPy as a base
dependency and presents SciPy and Numba as optional extras. Core imports are
eager: [`shell_thick16.py:13`](../pyradioss/elements/shell_thick16.py#L13) and
[`coloring.py:2`](../pyradioss/engine/coloring.py#L2) import Numba, while
[`ams.py:7`](../pyradioss/engine/ams.py#L7) and
[`lagmul.py:18`](../pyradioss/engine/lagmul.py#L18) import SciPy.

Fresh-process probes reproduced both failures:

- Blocking `numba` and importing `pyradioss.starter.starter` raises
  `ModuleNotFoundError`.
- Blocking `scipy` and importing `pyradioss.engine.engine` raises
  `ModuleNotFoundError` for `scipy.sparse`.

Users following the base dependency metadata cannot launch the explicit solver.

### AUD-004 — P1 — LAW36 shell filtering or kinematic hardening raises `NameError`

[`law36_tabulated.shell_update`](../pyradioss/materials/law36_tabulated.py#L307)
has no `extra` parameter, yet reads `extra` in the `f_cut` and `c_hard`
branches at lines
[`330`](../pyradioss/materials/law36_tabulated.py#L330) and
[`337`](../pyradioss/materials/law36_tabulated.py#L337). The material dispatcher
also discards its own `extra` argument for LAW36 at
[`materials/__init__.py:225`](../pyradioss/materials/__init__.py#L225).

Any shell LAW36 material with `f_cut > 0` or `c_hard > 0` crashes on the first
constitutive update.

### AUD-005 — P1 — LAW36 solid elastic rows lose their existing backstress

The solid update subtracts backstress from every row at
[`law36_tabulated.py:272`](../pyradioss/materials/law36_tabulated.py#L272), then
adds it back only for the plastic subset at
[`law36_tabulated.py:287`](../pyradioss/materials/law36_tabulated.py#L287).
Elastic unloading rows therefore return stress in the shifted frame and erase
the physical effect of their stored backstress.

A load/unload probe with `E=210`, `nu=0.3`, `c_hard=0.5`, and
`sigma_y=0.4+10 eps_p` produced `sigma_xx=0.31044024`; restoring the unchanged
backstress gives `0.36915607`. The `0.05871583` difference is exactly the prior
backstress, while equivalent plastic strain remains unchanged. The upstream
`sigeps36.F` keeps the backstress as a separate state and adds it to returned
stress (reference lines 1401–1414).

### AUD-006 — P1 — BT shell `Ishell=3` and `Ishell=4` force branches crash

[`shell_bt4.forces`](../pyradioss/elements/shell_bt4.py#L683) unpacks `_pre`
without `xl` at line 696. The IHBE 2/3/4 branches later reference `xl` at
[`shell_bt4.py:773`](../pyradioss/elements/shell_bt4.py#L773),
[`shell_bt4.py:790`](../pyradioss/elements/shell_bt4.py#L790), and
[`shell_bt4.py:814`](../pyradioss/elements/shell_bt4.py#L814).

A one-element `Ishell=3` force evaluation reproduced `NameError: name 'xl' is
not defined`. The six tests in
[`test_m41_bt_rotation.py`](../tests/test_m41_bt_rotation.py) execute forces only
for `Ishell=1`; cards 3/4 are checked only through their formulation mapping.

### AUD-007 — P1 — `/BRIC20` cannot complete Starter initialization

The builder maps BRIC20 into `model.bric20s` at
[`initialization.py:35`](../pyradioss/starter/initialization.py#L35), and
`Model.element_groups()` yields it at
[`model.py:3663`](../pyradioss/model/model.py#L3663). The kernel registry at
[`elements/__init__.py:34`](../pyradioss/elements/__init__.py#L34) has no
`bric20s` entry. Starter indexes `KERNELS[name]` unconditionally at
[`initialization.py:1329`](../pyradioss/starter/initialization.py#L1329), so a
valid BRIC20 deck raises `KeyError: 'bric20s'`. Existing milestone tests stop
before full Starter initialization.

### AUD-008 — P1 — SHEL16 has invalid mass, topology, material-time, and stable-step handling

The problems share the incomplete kernel in
[`shell_thick16.py`](../pyradioss/elements/shell_thick16.py):

- Initialization indexes `model.x0[conn]` at line
  [`591`](../pyradioss/elements/shell_thick16.py#L591), so absent midside nodes
  encoded as `-1` read the last unrelated global node.
- It sends `vol=np.ones(n)` into the mass routine at line
  [`600`](../pyradioss/elements/shell_thick16.py#L600). Element geometry does
  not contribute to initial mass.
- Both initial element-step arrays and every force return use `EP30`
  ([lines 601–602](../pyradioss/elements/shell_thick16.py#L601) and
  [line 791](../pyradioss/elements/shell_thick16.py#L791)). An all-SHEL16 model
  is clamped directly to the run end instead of receiving a stability limit.
- The force loop again indexes `x[conn]` at line
  [`746`](../pyradioss/elements/shell_thick16.py#L746), so missing midside
  coordinates corrupt strain and force before `-1` entries are filtered only
  during scatter.
- LAW2 receives `np.ones(n_sl)` instead of the actual scalar time step at line
  [`775`](../pyradioss/elements/shell_thick16.py#L775). LAW2 calls Python
  `max(dt, 1e-30)` at
  [`law02_johnson_cook.py:193`](../pyradioss/materials/law02_johnson_cook.py#L193),
  so a slice with more than one element raises an ambiguous-truth `ValueError`;
  a single element silently uses `dt=1` and gets the wrong strain-rate effect.
- Only LAW1 and LAW2 have branches. Any other accepted material law leaves
  stress unchanged without a warning or exception.

### AUD-009 — P1 — Slaved and mixed TETRA10 groups corrupt geometry and mass

Slaved quadratic tetrahedra use `-1` for virtual midside nodes. Initialization
immediately evaluates `model.x0[group.conn]` at
[`solid_tetra10.py:103`](../pyradioss/elements/solid_tetra10.py#L103), so the
last global node becomes every absent midpoint. A reference tetrahedron probe
changed volume magnitude from `0.166666664` with reconstructed midpoints to
`0.0298142404` when an unrelated last node was selected.

Mass is then repeated over all ten connectivity slots and scattered using the
raw connectivity at
[`solid_tetra10.py:128`](../pyradioss/elements/solid_tetra10.py#L128). For six
virtual midsides, 60% of the element mass lands on the final global node.

The force path decides whether the entire group is slaved by checking only
`conn[0, 4]` at
[`solid_tetra10.py:253`](../pyradioss/elements/solid_tetra10.py#L253). Mixed
real/slaved groups therefore either overwrite real midpoint kinematics or
scatter virtual-node force incorrectly, depending on row order.

### AUD-010 — P1 — Newer element kernels silently bypass `/FAIL`

TETRA10 explicitly sets `chk_fail=False` at
[`solid_tetra10.py:133`](../pyradioss/elements/solid_tetra10.py#L133), as does
QUAD at [`solid_quad.py:72`](../pyradioss/elements/solid_quad.py#L72). Neither
force path calls the failure handler. DKT18 imports `_layer_failure` at
[`shell_dkt18.py:73`](../pyradioss/elements/shell_dkt18.py#L73) but never calls
it. SHEL16 imports and allocates failure/material state at
[`shell_thick16.py:15`](../pyradioss/elements/shell_thick16.py#L15) and
[`shell_thick16.py:647`](../pyradioss/elements/shell_thick16.py#L647), but never
evaluates failure.

Failure cards attached to QUAD, TETRA10, DKT18, or SHEL16 therefore do not
damage/delete the element or update contact eligibility.

### AUD-011 — P2 — DKT18 and LAW27 print solver internals, while the DKT18 test is a duplicate

Every DKT18 force call prints the global and element coordinate maxima at
[`shell_dkt18.py:83`](../pyradioss/elements/shell_dkt18.py#L83). Degenerate
elements dump area, full coordinates, and connectivity at lines
[`103–105`](../pyradioss/elements/shell_dkt18.py#L103). LAW27 prints complete
broken-element arrays at
[`law27_brittle.py:132`](../pyradioss/materials/law27_brittle.py#L132). Long runs
can flood stdout and spend substantial time formatting arrays.

The supposed DKT18 milestone test
[`test_m47_dkt18.py`](../tests/test_m47_dkt18.py) is byte-for-byte identical to
[`test_m39_shell_fidelity.py`](../tests/test_m39_shell_fidelity.py), with SHA256
`D8FCC67E...D98EF`. It contains no DKT18 execution, so this kernel has no direct
physics regression test despite adding a full test file to the count.

### AUD-012 — P2 — The advertised QEPH Numba path is disconnected and malformed

QEPH requests accelerated blocks at
[`shell_qeph.py:950`](../pyradioss/elements/shell_qeph.py#L950) and
[`shell_qeph.py:1010`](../pyradioss/elements/shell_qeph.py#L1010). The active JIT
registry ends by exporting QBAT only at
[`jit_kernels/__init__.py:1185`](../pyradioss/accel/jit_kernels/__init__.py#L1185),
so `accel.get("qeph_pre")` and `accel.get("qeph_post")` always return `None`
and silently select NumPy.

The separate implementation is not usable if registered:
[`shells_qeph.py:419`](../pyradioss/accel/jit_kernels/shells_qeph.py#L419)
references undefined `d`, line 445 references undefined `G`, and
[`qeph_post`](../pyradioss/accel/jit_kernels/shells_qeph.py#L554) references
many undefined constants and arrays. Selecting the Numba backend therefore
does not accelerate QEPH, contrary to the milestone claim.

### AUD-013 — P1 — `/MONVOL/AIRBAG1` crashes with the valid default `Iequi=0`

[`airbag.update_airbag`](../pyradioss/engine/airbag.py#L4) does nothing when
`iequil == 0` at line 5, then continues with zero initial gas mass and divides
`right / left` at
[`airbag.py:40`](../pyradioss/engine/airbag.py#L40). The entity and parser both
default the field to zero at
[`entities.py:1210`](../pyradioss/model/entities.py#L1210) and
[`starter_keywords.py:19844`](../pyradioss/input/starter_keywords.py#L19844).

A default AIRBAG1 update reproduced `ZeroDivisionError`. The official
`airbag1.cfg` defines zero as deriving initial gas mass from the time-zero
volume; current tests use `Iequi=1` and miss the default path.

### AUD-014 — P1 — Transformations targeting derived groups execute before those groups exist

Starter applies `/TRANSFORM` at
[`starter.py:143`](../pyradioss/starter/starter.py#L143). Target resolution calls
`resolve_single_node_group` at
[`starter.py:89`](../pyradioss/starter/starter.py#L89), but element groups,
entity groups, surfaces, and final node groups are built only from
[`starter.py:315`](../pyradioss/starter/starter.py#L315) onward.

Direct node-list groups work. `/GRNOD/PART`, `/GRNOD/SURF`, and `/GRNOD/GR*`
targets resolve empty at transform time, so the requested motion is warned and
skipped.

### AUD-015 — P1 — Virtual node index `-1` leaks from elements into derived node groups

`_nodes_of_parts` appends `np.unique(group.conn[mask])` without filtering at
[`initialization.py:565`](../pyradioss/starter/initialization.py#L565), and
element-group references do the same at
[`initialization.py:817`](../pyradioss/starter/initialization.py#L817).

Slaved TETRA10 and optional-midpoint SHEL16 connectivity contains `-1`.
Consequently `/GRNOD/PART` and `/GRNOD/GRBRIC|GRSHEL` can include `-1`; any
load, constraint, transform, damping, or other nodal operation then applies to
the final unrelated node through NumPy negative indexing. The prior contact
tracking fix for `-1` does not sanitize these group builders.

### AUD-016 — P1 — High-order and QUAD surfaces are empty or malformed

`resolve_surfaces` handles explicit segments, low-order shell families, BRICK8,
and TETRA4 in
[`initialization.py:944`](../pyradioss/starter/initialization.py#L944). Its
`/SURF/PART` path omits QUAD, TETRA10, SHEL16, and BRIC20. Constructed
QUAD/TETRA10/SHEL16 part surfaces each resolved to shape `(0, 4)` with a
“has no segments” warning.

The element-group family table at
[`initialization.py:580`](../pyradioss/starter/initialization.py#L580) also omits
SHEL16 and BRIC20. For `/SURF/GRBRIC` referencing TETRA10, the generic branch at
[`initialization.py:1016`](../pyradioss/starter/initialization.py#L1016) forwards
the entire 10-node row as one segment. Contact code then interprets the first
four indices as a quadrilateral, which is neither a real tetrahedron face nor
the complete boundary.

### AUD-017 — P1 — Parsed analytical and selector `/SURF` definitions never resolve

The parser stores `/SURF/MAT`, `/SURF/PROP`, `/SURF/BOX`, `/SURF/PLANE`,
`/SURF/ELLIPSE`, `/SURF/CYL`, and `/SURF/SPHER` data at
[`starter_keywords.py:9169`](../pyradioss/input/starter_keywords.py#L9169)
through [`starter_keywords.py:9259`](../pyradioss/input/starter_keywords.py#L9259).
The entity exposes those fields at
[`entities.py:330`](../pyradioss/model/entities.py#L330).

`resolve_surfaces`, however, consumes only explicit segments, part IDs, element
group references, and referenced surfaces. No runtime path reads the material,
property, box, plane, ellipse, cylinder, or sphere selector fields. These
surfaces resolve empty, silently disabling contact, pressure loads, airbag
boundaries, and any other feature that uses them.

### AUD-018 — P1 — VTK animation crashes when a newer element group is present

The topology map at
[`anim_vtk.py:53`](../pyradioss/output/anim_vtk.py#L53) omits `quads`,
`tetra10s`, `shel16s`, and `bric20s`. The writer indexes the map for every
element group at
[`anim_vtk.py:189`](../pyradioss/output/anim_vtk.py#L189). A model containing a
QUAD reproduced `KeyError: 'quads'`; the other omitted groups fail the same way.
The stress-family logic at
[`anim_vtk.py:150`](../pyradioss/output/anim_vtk.py#L150) also omits their
integration-point state shapes.

### AUD-019 — P1 — Most requested T01 entity channels are plausible zero data

[`TimeHistory`](../pyradioss/output/time_history.py#L36) implements NODE, PART,
and SECT requests. Every other kind is placed in `_other_req` at
[`time_history.py:58`](../pyradioss/output/time_history.py#L58), and every such
column is written as `0.0` at
[`time_history.py:120`](../pyradioss/output/time_history.py#L120).

Thus `/TH/RBODY`, `/TH/SHEL`, `/TH/SH3N`, `/TH/SPRING`, `/TH/BRIC`,
`/TH/RWALL`, `/TH/INTER`, and the many later accepted kinds produce valid-looking
but fabricated zero histories. Nodal `AX/AY/AZ` also always take the fallback
zero branch at
[`time_history.py:103`](../pyradioss/output/time_history.py#L103), despite the
module documentation saying acceleration is approximated from force/mass.
Part KE at lines
[`84–91`](../pyradioss/output/time_history.py#L84) includes translation only.

### AUD-020 — P1 — Rotational kinetic energy is absent from global energy accounting

The Engine computes KE only as `0.5*m*v^2` at
[`engine.py:162`](../pyradioss/engine/engine.py#L162), and constructs initial E0
the same way at
[`engine.py:384`](../pyradioss/engine/engine.py#L384). No
`0.5*inertia*vr^2` term is included, although rotational equations, mass
scaling, imposed motion, and damping all use `model.inertia` and `model.vr`.

A node with `mass=2`, `inertia=4`, and `vr=(3,0,0)` reports KE `0`; its
rotational KE is `18`. This corrupts the T01 KE, energy error, peak/reference
energy, and energy-error termination for beams, shells, rigid bodies, or other
rotational DOFs.

### AUD-021 — P2 — Sensor latch status is lost across restart

Sensors store both `fire_time` and boolean `status` at
[`sensors.py:43`](../pyradioss/engine/sensors.py#L43), and `active()` reads only
`status` at [`sensors.py:159`](../pyradioss/engine/sensors.py#L159). Restart
serialization saves only `fire_time` at
[`engine.py:449`](../pyradioss/engine/engine.py#L449), and resume restores only
that dictionary at
[`engine.py:307`](../pyradioss/engine/engine.py#L307).

Restoring `{sensor_id: fire_time}` into a fresh board reproduces
`active(sensor_id) == False`. A latched DISP/VEL sensor can therefore become
inactive after restart until its threshold happens to trigger again, shifting
or suppressing dependent loads and outputs.

### AUD-022 — P2 — The GUI subprocess timeout cannot interrupt a quiet or stuck child

[`_stream_subprocess`](../pyradioss/gui/postproc.py#L485) performs a blocking
`for raw in proc.stdout` at line 505 and calls `proc.wait(timeout=timeout)` only
after stdout reaches EOF at line 516. A child that keeps stdout open never
reaches the timed wait. The GUI worker can hang indefinitely despite its
documented timeout.

### AUD-023 — P2 — Starter reports a missing input file with process exit code 0

[`run_starter`](../pyradioss/starter/starter.py#L112) catches
`FileNotFoundError` and returns integer `1` at line 132 even though its contract
returns `Model`. The CLI ignores that return value and always returns zero at
[`starter/__main__.py:39`](../pyradioss/starter/__main__.py#L39). A direct
missing-file invocation logged an error and exited successfully, so scripts and
CI cannot detect the failure.

### AUD-024 — P2 — Engine divergence and time-step failure also exit with code 0

Energy, NaN, minimum-step, and collapsed-step guards set `state.stop_reason`
at [`engine.py:783`](../pyradioss/engine/engine.py#L783) through line 821. Final
handling only logs `ENGINE TERMINATION : ERROR` at
[`engine.py:843`](../pyradioss/engine/engine.py#L843) and returns a model. The CLI
then unconditionally returns zero at
[`engine/__main__.py:58`](../pyradioss/engine/__main__.py#L58). Batch validation
can accept a diverged or prematurely stopped calculation as successful.

### AUD-025 — P1 — Three advertised Starter cards escape with `NameError`

The generic dispatcher catches only `ValueError`, `IndexError`, and `KeyError`
at [`starter_keywords.py:90525`](../pyradioss/input/starter_keywords.py#L90525),
so these undefined names terminate parsing:

- `/BCS/TEMP` and `/BCS/FLUX` instantiate undefined `ThermalBcs` and assign to
  nonexistent `model.thermal_bcs` at
  [`starter_keywords.py:9559`](../pyradioss/input/starter_keywords.py#L9559).
  The real types are `HeatBcs` and `model.heat_bcs`.
- `/INIVEL/ROTVEL` calls nonexistent `read_inirotvel` at
  [`starter_keywords.py:10156`](../pyradioss/input/starter_keywords.py#L10156).
- A one-card fixed-format `/HOOKE_JOINT` reads undefined `LAYOUTS` at
  [`starter_keywords.py:49911`](../pyradioss/input/starter_keywords.py#L49911).
  This module imports `CARD_LAYOUTS` and later aliases a separate `_LAYOUTS`.

Each branch was reached with a minimal valid `KeywordBlock` and raised the
named exception.

### AUD-026 — P1 — Consistent modal mass ignores `/ADMAS`

Starter correctly adds `/ADMAS` to the explicit lumped nodal mass at
[`initialization.py:1341`](../pyradioss/starter/initialization.py#L1341).
The implicit consistent-mass assembler iterates only element
`consistent_mass()` matrices at
[`assembly.py:129`](../pyradioss/implicit/assembly.py#L129) and never reads
`model.admas` or the nodal mass delta. Modal analysis uses that matrix at
[`modal.py:135`](../pyradioss/implicit/modal.py#L135).

In a bar probe, adding `0.0026` tip mass changed `model.mass` but left the first
frequency exactly `14.30352584 Hz`; including the point mass gives about
`4.31267531 Hz`. Modal response and spectral-fatigue results inherit the wrong
mass model.

### AUD-027 — P1 — Shell and high-order stress recovery crashes random response

`stress_channels()` treats only `ndim == 2` stress arrays as multi-component at
[`random_response.py:583`](../pyradioss/implicit/random_response.py#L583).
Anything else becomes one scalar channel. `stress_modes()` then assigns the
selected entry to a scalar at
[`random_response.py:623`](../pyradioss/implicit/random_response.py#L623).

BT/QBAT/QEPH/DKT/SHEL16 shell state and TETRA10 state are three-dimensional
`(element, integration_point, component)` arrays. A shell recovery probe
reproduced `ValueError: setting an array element with a sequence`. Random
response and all spectral-fatigue workflows built on this recovery fail for
those element families.

### AUD-028 — P1 — Rotational `/IMPDISP` crashes implicit dynamics

The Newmark step iterates all imposed DOFs but always writes into translational
`u[idx, d]` at
[`dynamics.py:822`](../pyradioss/implicit/dynamics.py#L822). `u` has three
columns; rotational directions map to `d=3..5`. `/IMPDISP ... ZZ` reproduced an
`IndexError` on the first step. Constraint-reaction work repeats the same
translational indexing at
[`dynamics.py:620`](../pyradioss/implicit/dynamics.py#L620), so merely changing
the assignment would still leave rotational work broken.

### AUD-029 — P2 — Project status and test structure overstate runtime coverage

[`docs/STATE.md:30`](STATE.md#L30) reported 1,121 collected / 1,105 fast tests
(as of 2026-09-06, this audit's snapshot);
the audited snapshot collects 5,521 / 5,502. Its heading still says
“M1 → M251” at [`STATE.md:43`](STATE.md#L43), while the table reaches M488, and
line 46 simultaneously says the “real history” has 41 milestones.

More materially, many roadmap entries use “implement” for parser/dataclass
storage even when the Engine, element kernels, or output writers never consume
the stored state. A conservative AST census found **2,220 attributes** assigned
in `Model.__init__`; **1,814** had no direct or literal dynamic read in runtime
modules after excluding input parsers and deck serialization. Largest prefixes
were `fail` 547, `lagmul` 335, `eng` 284, `sensor` 209, `mat` 132, and `prop` 41.

That census is an inventory signal rather than 1,814 individual confirmed
defects: aliases, metadata, and indirect serialization can be legitimate. In
combination with the directly reproduced no-ops above, it shows that parser
coverage and milestone counts are not a reliable measure of solver support.

## Recommended triage order

1. Make accepted-but-unused physics/control cards fail loudly or mark them as
   parser-only. Fix `/STOP`, activation/deletion, and Engine `/DT` handling
   first because they can invalidate a run without any visible error.
2. Repair the immediately crashing primary paths: AMS, LAW36 shell, BT
   `Ishell=3/4`, AIRBAG1 default, BRIC20, random-response stress recovery, and
   rotational `/IMPDISP`.
3. Treat QUAD/TETRA10/SHEL16/BRIC20 as one high-order integration milestone:
   topology reconstruction, mass, stable time step, surfaces/contact, failure,
   output, and dedicated end-to-end tests must land together.
4. Correct energy, restart, and exit-status contracts so validation tooling can
   trust a completed process and its T01 output.
5. Replace parser-only milestone tests with at least one Starter-to-Engine or
   implicit execution test for every feature advertised as implemented.

## Appendix A — `EngineControls` fields with no runtime reference

This conservative list excludes parser/model declaration reads and includes
literal-string dynamic access. Some fields may be intentional metadata, but the
simulation-affecting fields need either an implementation or an explicit
unsupported diagnostic.

`abf_dt`, `abf_dt_write`, `ale_active`, `anim_tens`, `bcs_active`,
`checksum_mode`, `damp_alpha`, `damp_beta`, `damp_dt`, `damp_grpart`,
`damp_tstart`, `damp_tstop`, `debug_acc_freq`, `debug_acc_start`, `debug_flags`,
`del_elements`, `dli7_controls`, `dt1tet10`, `dt_controls`, `dtix_tini`,
`dtix_tmax`, `dttsh`, `dynain_dt`, `dynain_tstart`, `dyrel_active`,
`dyrel_beta`, `dyrel_istatg`, `dyrel_period`, `eig_off`, `flow_dt`,
`fvbag_modif`, `fvbag_remesh`, `h3d_dt`, `h3d_requests`, `heat_active`,
`heat_flag`, `impl_dt_fixp`, `impl_fatig_copula`, `impl_fatig_copula_params`,
`inivel_engine`, `inter_active`, `inter_windows`, `kerel_active`,
`kerel_istatg`, `kerel_tstart`, `kerel_tstop`, `madymo_mode`, `mass_reset`,
`negvol_action`, `noise_dt`, `noise_flags`, `noise_tstart`, `parith`,
`perf_sort`, `python_functions`, `rad2r_active`, `rbody_active`, `report_dt`,
`report_freq`, `run_name`, `sensor_reset`, `stop_nstep`, `stop_timet`,
`stop_tstop`, `th_records`, `thermal_acc_fact`, `thermal_dt`, `thermal_tstart`,
`upwind_active`, `upwind_mass_eng`, `upwind_mom`, `upwind_wet_surf`,
`viper_active`.

## Exclusions checked

- The typo-shaped expressions `epssp if 'epssp' in locals() else epsp` in
  `/INITRU` and `yssp if 'yssp' in locals() else ysp` in LAW122 are dead
  fallbacks, but their guards are always false and the parsed values are
  preserved. Direct fixed/free probes and existing targeted tests passed.
- The UTF-8 BOM in `shells_qbat.py` makes a naive `ast.parse()` of decoded text
  complain, but Python's source loader handles it correctly; the module imports
  and its tests pass.
- The older [`BUG_REPORT_2026-09-05.md`](BUG_REPORT_2026-09-05.md) describes
  defects fixed before this audited commit. They were not carried forward as
  open findings.
