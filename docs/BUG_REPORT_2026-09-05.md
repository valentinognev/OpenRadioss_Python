# Bug report — 5 September 2026

## Scope and verification (2026-09-05, revision `f0263ad`)

- Reviewed revision: `f0263ad` (`feat/vtk-full-tensors`, M483).
- Review focus: M476–M479 and the runtime paths reached from those features. M480–M483 were not audited.
- Status: all findings below were open at the reviewed revision. Antigravity is continuing development; recheck each finding before fixing it.
- The review changed no solver code. This report records the findings previously delivered in the Codex conversation.
- Verification: 78 focused tests passed, alongside the targeted reproductions described below. Passing unit tests did not establish correct parser-to-engine integration.
- Upstream comparison was by reading the installed Fortran source. No full Fortran differential run was performed for this report.

The focused test command was:

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_m477_iform2_friction.py tests\test_m478_damp_rayleigh.py tests\test_m479_contact_tracking.py tests\test_m476_lad_dccfr_serre_spinor_spatial_suite.py tests\test_m4_contact.py::test_type7_forces_vanish_on_deleted_segments
```

Observed result: `78 passed in 3.52s`.

## Findings

P1 means a high-priority simulation correctness problem. P2 means a narrower correctness problem. These are current defects, not a claim that every defect was introduced by M476–M479.

### BUG-01 — P1: Iform=2 parsing disables friction

Location: [starter_keywords.py:14847](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/input/starter_keywords.py:14847).

Trigger: a fixed-format TYPE7 interface with `Iform=2` and nonzero friction.

The reader adds 10 to IFQ before computing XFILTR. The subsequent mapping handles only IFQ 1, 2 and 3, leaving XFILTR at zero for modes 10–13. The engine uses this zero as the multiplier on the incremental tangential stiffness. With an initially empty force history, friction therefore stays zero.

Observed parser results:

| Input Ifiltr | Input Xfreq | Runtime IFQ | Runtime XFILTR |
| --- | --- | --- | --- |
| 0 | 0 | 10 | 0.0 |
| 1 | 0.25 | 11 | 0.0 |
| 2 | 100 | 12 | 0.0 |
| 3 | 20 | 13 | 0.0 |

Upstream [hm_read_inter_type07.F:623](C:/OpenRadioss/source/OpenRadioss-latest-20260520/starter/source/interfaces/int07/hm_read_inter_type07.F:623) assigns XFILTR=1 for IFQ=10 and maps the other modes through `MOD(IFQ,10)`.

Acceptance check: parse all four fixed-format cases and exercise the resulting ContactType7 objects. Check both the mapped coefficients and nonzero resisting tangential forces.

### BUG-02 — P1: Incremental friction accelerates the sliding node

Locations: [inter_type7.py:554](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/contact/inter_type7.py:554), [friction.py:256](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/contact/friction.py:256).

Trigger: an incremental-friction interface with a nonzero XFILTR, such as one constructed through the Python API. BUG-01 currently masks this in the reproduced deck-driven cases.

The relative velocity is secondary minus main. The helper accumulates a force in that direction, and ContactType7 adds the returned force to the secondary node. This produces positive tangential power during initial sliding. The Fortran force convention cannot be copied without also translating its scatter sign: upstream [i7for3.F:1511](C:/OpenRadioss/source/OpenRadioss-latest-20260520/engine/source/interfaces/int07/i7for3.F:1511) subtracts its force from the secondary node.

A one-node/one-quad ContactType7 reproduction used stiffness 1000, friction coefficient 0.5, gap 0.1, separation 0.05, timestep 0.001 and secondary velocity `[1,0,0]`:

| Mode | Secondary force | Tangential power |
| --- | --- | --- |
| IFQ=0 | `[-22.7272727, 0, 50]` | -22.7272727 |
| IFQ=10, XFILTR=0 | `[0, 0, 50]` | 0 |
| IFQ=10, XFILTR=1 | `[1, 0, 50]` | +1 |

Acceptance check: verify the assembled secondary and main forces, not only the helper's magnitude. Initial sliding from zero tangential history must be resisted, and the two sides must receive equal and opposite forces.

### BUG-03 — P1: Lagrange-multiplier constraints never enter the solve

Location: [lagmul.py:53](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/engine/lagmul.py:53).

Trigger: a model with a registered LAGMUL interface.

`nc` starts at zero. `__len__()` returns `nc`, so the engine's `if len(lagmul) > 0` condition skips the first force transfer. Calling `transfer_forces()` directly also returns when `nc == 0`, before any handler can generate rows and update the count. The constraint system can never become active through the normal lifecycle.

Focused reproduction: one registered interface, public length zero, zero calls to its row builder. Constraint forces and velocity corrections are never applied.

Acceptance check: construct a supported interface through normal initialization, run the first force stage, and assert both row generation and the resulting constraint response. Do not seed `nc` manually in the regression test.

### BUG-04 — P1: Damping rigid-body slaves books phantom dissipation

Locations: [damping.py:81](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/engine/damping.py:81), [engine.py:608](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/engine/engine.py:608), [rigid_body.py:526](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/engine/rigid_body.py:526).

Trigger: a /DAMP node group overlaps moving /RBODY or /RBE2 slave nodes.

Dampers includes those slaves, rescales their trial velocities and books the kinetic-energy reduction in DE. RigidBodyEngine.advance subsequently overwrites these velocities with the rigid field. The damping booking remains even though the final rigid-body motion did not lose that energy.

For mass 2, velocity 3 and `alpha*dt=1`, the focused damping/restore reproduction booked 7.78198245087 energy units while the final kinetic-energy loss was zero. This numeric reproduction illustrates the engine ordering; it was not a complete deck run.

Upstream [damping.F:150](C:/OpenRadioss/source/OpenRadioss-latest-20260520/engine/source/assembly/damping.F:150) and its rotational branch explicitly exclude nodes with a rigid-body slave tag.

Acceptance check: run a translating rigid body with an overlapping damping group and verify final motion and all energy bookings against the upstream treatment.

### BUG-05 — P1: Contact deletion runs even with Idel=0

Location: [inter_type7.py:423](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/contact/inter_type7.py:423).

Trigger: a TYPE7 interface with default `Idel=0` whose main-surface parent elements can fail.

Once the material-based `deletable` flag is true, the force routine masks failed segments without consulting Idel. The existing deletion test uses a compact TYPE7 whose parsed Idel remains zero and nevertheless expects the contact to vanish.

Upstream [chkstfn3.F:1352](C:/OpenRadioss/source/OpenRadioss-latest-20260520/engine/source/interfaces/interf/chkstfn3.F:1352) gates this processing on IDEL>=1. It also distinguishes the enabled deletion modes. The Python behavior therefore changes contact forces for decks that leave deletion disabled.

Acceptance check: test disabled and enabled deletion separately with the same failing geometry. Determine each enabled mode's behavior from the upstream implementation.

### BUG-06 — P1: Failure confined to secondary elements never enables tracking

Location: [inter_type7.py:308](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/contact/inter_type7.py:308). The same pattern occurs in [inter_type10.py:62](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/contact/inter_type10.py:62) and [inter_type24.py:300](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/contact/inter_type24.py:300).

Trigger: secondary contact nodes belong to failure-capable elements, while the main surface has explicit or non-failing provenance.

`any_deletable()` examines only main-segment provenance. It returns false in this case, so the handler never builds reference counts or refreshes the secondary-node mask. After all structural elements attached to a secondary node die, that node can keep transmitting contact force indefinitely.

Focused gate reproduction: main provenance `[""]` with a failure-capable secondary group returned false. Source tracing confirms that the secondary-node refresh is behind the same gate.

Acceptance check: enable the intended deletion policy, fail only the secondary elements, advance beyond a broad-phase refresh, and assert that orphaned nodes cease transmitting force.

### BUG-07 — P2: Missing connectivity creates false ownership of the last node

Location: [tracking.py:76](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/contact/tracking.py:76).

Trigger: a model contains supported slaved TETRA10 connectivity with `-1` placeholders, and its final real node becomes structurally orphaned.

The reference-count scatter passes every connectivity value to NumPy, where index -1 means the final array entry. A slaved TETRA10 therefore creates six nonexistent references to the last node. This can keep that node active in contact after its actual attached elements fail.

Reproduction: a live TETRA10 with connectivity `[0,1,2,3,-1,-1,-1,-1,-1,-1]` and a deleted element attached to node 4 produced alive counts `[1,1,1,1,6]`. The orphaned node remained tracked.

Acceptance check: exclude missing-node markers from ownership counts and test both live and deleted groups with placeholder connectivity.

### BUG-08 — P2: Numeric titles shift fixed-format failure-card fields

Location: [starter_keywords.py:56657](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/input/starter_keywords.py:56657).

Trigger: a fixed-format /FAIL/LAD_DYNAMIC_CORE_CRUSHING_FAILURE_RATE block has a numeric title, for example `123`.

The reader uses `_title_and_data`, whose free-format heuristic interprets a numeric first card as data. Fixed-format titled cards must consume that first card as the title regardless of its contents.

Reproduction: title `123`, thresholds 120/380 and deletion flags 1/2 were read as an empty title, thresholds 123/1 and flags 1/120, with no parse errors. `_fixed_data` already provides the appropriate fixed-format title handling elsewhere.

Acceptance check: compare identical numeric cards under alphabetic and numeric fixed-format titles; the parsed physical parameters must be identical.

## M476 integration gaps — separate from the defects above

These observations identify unfinished integration. They do not establish that the named physics has been implemented or that every missing feature is a regression.

- **Failure model:** [starter_keywords.py:56701](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/input/starter_keywords.py:56701) stores the specialized object and may append to `mat.fail_models`, but does not add a generic failure model to `raw_fails`. After material resolution, `mat.fail` remains None. Element failure paths consume `mat.fail`, not this new list. Placing the FAIL card before MAT also loses that direct attachment.
- **Energy directive:** [starter_keywords.py:56709](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/input/starter_keywords.py:56709) registers the new resonance-energy reader in Starter. The corresponding Engine-deck reproduction warned that the keyword was not ported and ignored it. No runtime output consumer was found.
- **Linkage joint:** [starter_keywords.py:56776](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/input/starter_keywords.py:56776) stores the Serre linkage object, but no engine or implicit-solver consumer was found for that collection.
- **Torsional-crackle sensor:** [starter_keywords.py:62162](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/input/starter_keywords.py:62162) creates a generic sensor kind absent from [Sensors](C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/engine/sensors.py:45). The reproduction parsed one sensor but produced an empty runtime definition table, so it cannot fire.

Searches did not find the exact new M476 keyword names in the configured upstream source or hm_cfg_files tree. Establish their upstream provenance before implementing additional physics. Parser acceptance and alias tests alone are not evidence of solver support.

## Suggested repair order

1. Address BUG-01 and BUG-02 together; correcting the coefficient exposes the force-sign defect.
2. Address LAGMUL startup and damping energy accounting.
3. Correct contact deletion policy, tracking activation and connectivity counting together, with distinct regression cases for each.
4. Correct the fixed-format title handling and explicitly track M476 runtime support separately from parser coverage.

Physics repairs require targeted tests and the repository's Fortran parity validation. This review does not supply that post-fix evidence.
