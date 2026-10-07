# OpenRadioss_Python (`pyradioss`)

[![CI](https://github.com/pmquang87/OpenRadioss_Python/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pmquang87/OpenRadioss_Python/actions/workflows/ci.yml)

A **Python port of [OpenRadioss](https://github.com/OpenRadioss/OpenRadioss)**, the
open-source explicit finite-element solver for crash, impact and highly non-linear
transient dynamics.

> **Goal of this repository.** Reproduce the *functional architecture* of
> OpenRadioss — the Starter / Engine split, the Radioss input-deck format, the
> explicit central-difference solver, the element/material/contact libraries and
> the output files — in pure, heavily documented Python. The point is **not**
> speed (Python is orders of magnitude slower than the original Fortran); the
> point is a code base you can *read, understand and modify*. Every module
> carries long explanatory comments and a reference to the original Fortran
> source it was ported from.

---

## How OpenRadioss works (and how this port mirrors it)

OpenRadioss is split into two executables, and so is this port:

| OpenRadioss (Fortran) | This port (Python) | Role |
|---|---|---|
| `starter/` → `starter_linux64_gf` | `pyradioss/starter/` → `pyradioss-starter` | Reads the input deck `<RunName>_0000.rad`, checks the model, builds/initializes all data structures (masses, element volumes, contact surfaces…), writes the `<RunName>_0000.out` listing and a *restart* file for the Engine. |
| `engine/` → `engine_linux64_gf` | `pyradioss/engine/` → `pyradioss-engine` | Reads the engine control file `<RunName>_0001.rad` **and** the Starter restart file, runs the explicit time-integration loop, writes the `<RunName>_0001.out` listing, the time-history file (T01) and animation states (A-files). |

The original build chain (`build_script.sh`, gfortran, MPI…) described in the
upstream [HOWTO.md](https://github.com/OpenRadioss/OpenRadioss/blob/main/HOWTO.md)
is replaced by a plain Python package — see [HOWTO.md](HOWTO.md) in this repo
for the equivalent instructions.

## Installation

```bash
git clone https://github.com/pmquang87/OpenRadioss_Python.git
cd OpenRadioss_Python
pip install -e .          # installs `pyradioss` + the two console scripts
```

The only runtime dependency is **NumPy** (all element/material kernels are
vectorized over elements — this matches how the Fortran loops over element
*groups*). Since M7 there is also an **optional numba backend** for the
hottest kernels (1.5–2.9× on the bundled examples, identical results), and
since M40 it is the **default when it pays off**: if numba is installed the
engine turns it on automatically for models large enough to amortise its
one-time JIT warm-up (≥ 32 elements, a threshold derived from the M39 speed
sweep) and stays on NumPy for small models and whenever numba is absent — so
the base install keeps working unchanged. The engine listing names the
chosen backend (`COMPUTE BACKEND . . . : numba (auto: 200 elements >= 32)`).

```bash
pip install -e ".[accel]"                          # adds numba -> auto-enabled
pyradioss-engine -i MYRUN_0001.rad                 # auto: numba when it helps
pyradioss-engine -i MYRUN_0001.rad -backend numpy  # pin (or PYRADIOSS_BACKEND=numpy)
```

Pin either backend with `-backend numpy|numba` or `PYRADIOSS_BACKEND` to
override the auto default. See `pyradioss/accel/__init__.py` for the backend
architecture, the auto-threshold derivation and the parity contract, and
`tools/benchmark.py` to measure both backends on the examples.

## Running a simulation

Exactly like the original solver — two steps, same file naming convention:

```bash
cd examples/tensile_bar

# 1. Starter: reads TENSILE_0000.rad, writes TENSILE_0000.out + TENSILE_0000.rst
pyradioss-starter -i TENSILE_0000.rad

# 2. Engine: reads TENSILE_0001.rad + TENSILE_0000.rst, runs the simulation
pyradioss-engine  -i TENSILE_0001.rad
```

(You can also run them without installing: `python -m pyradioss.starter -i …`
and `python -m pyradioss.engine -i …`.)

### Domain decomposition (`-np N`, the SPMD port)

The original solver's SPMD mode — `starter -np N` cuts the model into N
domains, `mpirun -np N engine` runs one MPI process per domain — is ported
in `pyradioss/spmd/` (Fortran origin: `starter/source/spmd/` and
`engine/source/mpi/`):

```bash
# 1. Starter -np N: weighted recursive-bisection decomposition (initwg.F /
#    domdec1.F / domdec2.F), frontier nodes shared between domains, and ONE
#    restart per domain, TENSILE_0000_0001.rst ... TENSILE_0000_000N.rst
#    (ddsplit.F naming), plus the usual global TENSILE_0000.rst
pyradioss-starter -i TENSILE_0000.rad -np 4

# 2a. Engine -np N without MPI: the N domains run as N threads of one
#     process (pyradioss.spmd.ThreadComm) — functional, not a speed-up
pyradioss-engine -i TENSILE_0001.rad -np 4

# 2b. Engine under MPI (needs mpi4py + an MPI library): one process per
#     domain (pyradioss.spmd.Mpi4pyComm); N must equal the Starter's -np,
#     else the inipar.F "REQUIRED (number of .rst files) NSPMD" error
mpirun -np 4 python -m pyradioss.engine -i TENSILE_0001.rad -np 4
```

Each cycle the domains exchange and sum the frontier-node forces and
stiffnesses (`spmd_exch_a.F`), agree on the global time step
(`spmd_glob_min5.F`), and reduce the energy/momentum ledgers with the
`WEIGHT` convention so shared nodes are booked once (`ecrit.F`). Domain 0
gathers the global view and writes the ONLY listing, T01 and ANIM files
(`spmd_chkw.F`: only P0 prints; set `PYRADIOSS_SPMD_LOG_ALL=1` for
per-domain `<Run>_0001_000p.out` listings). The results are the serial
results up to floating-point summation order.

Outputs:

| File | Content |
|---|---|
| `<Run>_0000.out` | Starter listing: model summary, checks, initial mass/inertia. |
| `<Run>_0001.out` | Engine listing: cycle table (time, time step, energies, momentum) and final energy balance. |
| `<Run>T01.csv` | Time history (global energies + requested `/TH` groups). The original binary T01 format is replaced by documented ASCII/CSV. |
| `<Run>A000.vtk`, `A001.vtk`, … | Animation states. The proprietary ANIM format is replaced by **legacy VTK**, directly readable in [ParaView](https://www.paraview.org/) — the tool the OpenRadioss HOWTO itself recommends for post-processing. |

## GUI (`pyradioss-gui`)

A small **run-and-monitor** desktop GUI wraps the two console scripts. It is
built on Tkinter (Python standard library, no new hard dependency);
`matplotlib` is optional — when present the Results tab embeds live T01
plots, otherwise it shows a textual channel table.

```bash
pyradioss-gui                       # empty, then Browse… to a *_0000.rad deck
pyradioss-gui TENSILE_0000.rad      # or pre-select a deck
python -m pyradioss.gui             # without installing
```

- **Job panel** — pick a `*_0000.rad` starter deck (the last directory is
  remembered in `~/.pyradioss_gui/config.json`), the `*_0001.rad` engine deck
  is auto-derived, choose the backend (auto/numpy/numba), the **CPUs** to use
  (the `-np N` SPMD domain count handed to both the Starter and the Engine —
  1 = serial, bounded by the CPUs the machine offers, remembered in the
  config) and the thread count (`-nt`), then **Run** (Starter then Engine as
  subprocesses, stdout streamed into the log) / **Stop**.
- **Progress** — the Engine's cycle listing is parsed into a live status bar
  and a progress bar against the `/RUN` end time; the `ENGINE TERMINATION`
  banner is shown prominently (green NORMAL / red ERROR).
- **Results tab** — load and plot the `<Run>T01.csv` channels (IE/KE/EW/error).
- **Deck info tab** — the port's read-only deck reader's model summary
  (node/element counts, materials, properties, parts, contacts, keywords).
- **Post-processing tab** — convert a finished run's artifacts for downstream
  viewers, streaming progress into the Log tab. Pick a run directory (it is
  auto-filled with the last job's directory) and **Detect artifacts**; the
  three converter buttons enable/disable to match what is present:
  - **anim → d3plot** — the open-source [Vortex-Radioss](https://github.com/Vortex-CAE/Vortex-Radioss)
    library reads the `<Run>A001`, `A002`, … animation files and writes an
    LS-Dyna `.d3plot` family so results open in LS-PrePost. **Effective plastic
    strain is preserved** (the pin, `v1.021`, is chosen for exactly this — older
    versions omit or misassign it; see the `postproc` extra comment in
    `pyproject.toml`). Enabled only when Vortex-Radioss is installed; otherwise
    the tab shows the install command.
  - **anim → VTK** — OpenRadioss `anim_to_vtk_win64.exe`, one `.vtk` per
    animation file.
  - **TH → CSV** — OpenRadioss `th_to_csv_win64.exe` on a *Fortran* binary
    time-history `<Run>T01` (pyradioss runs already emit the T01 as CSV
    natively, so this is greyed for a port-native run).

  The Fortran converter directory is **resolved, never hardcoded**: the tab asks
  `pyradioss/paths.py` — the single external-path resolver — for your
  OpenRadioss install prefix and looks for the exes in its `bin/` and `exec/`
  subdirectories, so on Linux `export OR_ROOT=<your OpenRadioss install prefix>`
  is all that is needed; the Windows compatibility path `C:\OpenRadioss\exec`
  (the pre-cmake Windows install) is still tried last. With nothing configured
  and nothing found, the tab names every location it tried and what to export.
  The entry stays editable in the tab (persisted to the config). The **After
  run:** toggles in the Job panel (`d3plot` / `VTK` / `TH→CSV`) run the selected
  conversions automatically after a clean, NORMAL Engine run. The d3plot bridge
  is an optional extra (it is not part of the base install):

  ```bash
  pip install "pyradioss[postproc]"   # lasso-python + Vortex-Radioss @ v1.021
  ```

## What is implemented so far

> **Current status lives in [docs/STATE.md](docs/STATE.md)** — the milestone
> history (M1–M41: explicit + implicit solvers, the spectral-fatigue tower,
> QBAT/QEPH shells, differential validation against the real Fortran solver,
> the GUI), the known-good test baseline, and the roadmap. See
> [PORTING_GUIDE.md](PORTING_GUIDE.md) (large; grep, don't read linearly) for
> the detailed feature matrix and the map from every Python module to the
> original Fortran directory. The narrative below describes the foundation
> laid in Milestones 1–11:

- **Input**: the Radioss block-keyword deck format (`/NODE`, `/BRICK`,
  `/TETRA4`, `/SHELL`, `/SH3N`, `/TRUSS`, `/SPRING`, `/BEAM`, `/PART`,
  `/MAT/LAW1/2/27/36/42`, `/FAIL/JOHNSON`, `/FAIL/BIQUAD`,
  `/PROP/TYPE1/2/3/4/14`,
  `/BCS`, `/INIVEL`, `/GRAV`, `/CLOAD`, `/PLOAD`, `/IMPVEL`, `/IMPDISP`,
  `/ADMAS`, `/FUNCT`, `/GRNOD`, `/RWALL`, `/RBODY`, `/RBE2`, `/RBE3`,
  `/SECT`, `/INTER/TYPE2/7/11`, `/SURF`, `/LINE`, `/TH`, `/EOS`,
  `/DAMP`, `/SENSOR`, `/MPC`, …) with `#include` support.
- **Elements**: 8-node solid (one-point integration + Flanagan–Belytschko
  hourglass control) with degenerated-brick→tetra conversion, 4-node
  constant-strain tetra, 4-node Belytschko–Tsay shell (membrane + bending +
  transverse shear, through-thickness integration, BLT84 stiffness hourglass
  control), 3-node C0 triangle shell, 2-node corotational Timoshenko beam
  (elastic or global-plasticity), 2-node truss, 2-node spring.
- **Materials**: LAW1 (linear elastic, hypoelastic Jaumann update), LAW2
  (Johnson–Cook elasto-plasticity with strain-rate hardening, 3D and
  plane-stress variants), LAW36 (tabulated plasticity from /FUNCT hardening
  curves with a strain-rate curve family), LAW27 (brittle tensile cracking
  for shells, fixed crack direction + unilateral damage), LAW42
  (Ogden/Mooney-Rivlin hyperelasticity for solids, with the nonlinear
  tangent sound speed feeding the stable time step).
- **Failure**: /FAIL/JOHNSON and /FAIL/BIQUAD damage criteria with full
  element deletion (per-layer bookkeeping for shells, deleted elements
  flagged in the VTK output as `OFF`), plus the eps_p_max thresholds of
  LAW2/LAW36.
- **Contact** (M4): /INTER/TYPE7 penalty node-to-surface contact with the
  Istf stiffness variants, constant/variable gap (Igap), self-impact and a
  voxel (bucket) broad phase; /INTER/TYPE2 tied contact (kinematic
  secondary-to-main gluing with co-rotating offsets — does no work by
  construction); /INTER/TYPE11 edge-to-edge penalty contact on /LINE edge
  sets. Contact fully respects /FAIL element deletion: segments and edges
  of deleted elements drop out and orphaned nodes stop being tracked, so
  crack faces behave physically.
- **Constraints & loads** (M5): /RBODY rigid bodies (starter-assembled
  mass/COG/inertia tensor; stable 6-DOF Newton–Euler update that
  conserves angular momentum by construction, exponential-map finite
  rotation; coexists with contact) and /RBE2 rigid links (pivot mode
  from the master's /BCS — physical pendulums), /RBE3 interpolation
  constraints (force distribution without stiffening), rigid walls with
  plane/sphere/cylinder geometry that can move (free with a mass —
  momentum-exact impulse exchange — or velocity-driven), /SECT
  section-force output, /PLOAD follower pressure, /IMPDISP imposed
  displacement, /ADMAS added mass, /INIVEL/AXIS initial spin.
- **Engine niceties** (M6): /DT/NODA/CST mass scaling (the run holds its
  target time step by adding mass at the critical nodes — the added
  mass, momentum and energy are tracked, reported and kept in the
  balance), restart chaining (`RunName_0002.rad` resumes the previous
  run's `.rst` and reproduces the unchained run exactly), /STATE/DT
  restart snapshots, /DAMP mass damping with exactly-booked
  dissipation, /SENSOR/TIME + /SENSOR/DISP gating loads and contact
  interfaces, /MPC general linear multi-point constraints, /EOS
  (polynomial and ideal-gas equations of state with implicit
  energy–pressure coupling), the LAW2 adiabatic thermal terms and
  /FAIL/JOHNSON's D5.
- **Engine**: explicit central-difference integration, element/nodal stable
  time step (including the accumulated contact-spring stiffness), boundary
  conditions, initial/imposed velocities, gravity, concentrated loads,
  kinematic rigid walls, full energy-balance bookkeeping (contact work
  booked at the leapfrog-consistent midstep velocity; since M6 the
  numerical dissipation of the element dampers is measured exactly and
  reported as its own EN ledger).
- **Performance** (M7, M39, M40): profiled cycle path with pure-NumPy fast
  paths (`pyradioss/common/fastmath.py`) and a numba backend
  (`pyradioss/accel`) mirroring the measured hotspots — solid/shell kernels,
  the TYPE7 contact narrow phase and the force scatter — with a tested
  parity contract: both backends produce the same results, and the
  restart-chaining bit-match holds under numba. Since M40 the backend
  defaults to `auto` — numba once installed and the model is large enough to
  amortise JIT warm-up (a size threshold derived from the M39 speed sweep),
  NumPy otherwise — with `-backend`/`PYRADIOSS_BACKEND` still pinning either.
- **Implicit solver** (M8–M11, `pyradioss/implicit/`, needs SciPy):
  a parallel Newton–Raphson branch reusing the explicit force kernels for
  the residual — statics with load stepping and /IMPDISP displacement
  control (M8), nonlinear geometry with the updated-Lagrangian frame,
  geometric stiffness and Riks/Crisfield arc-length continuation through
  limit points (M9, `/IMPL/NONLIN`, `/IMPL/ARCL`), Newmark-β/HHT-α
  implicit dynamics on the lumped mass (M10, `/IMPL/DYNA`), and — M11 —
  element-tangent COMPLETENESS (every element family, mixed models
  welcome), LAW2 consistent tangents for shells and trusses, Rayleigh
  damping with an exact dissipation ledger (`/IMPL/DYNA/DAMP`), automatic
  step cut/growth control (imp_dt.F) and the `/IMPL/BUCKL` linearized
  buckling card.
- **Output**: listings, CSV time history (incl. /TH/SECT section
  resultants), VTK animation states.

## Repository layout

```
pyradioss/
├── common/       # shared low-level services (≈ OpenRadioss common_source/)
├── input/        # deck reader + keyword parsers (≈ starter/source/reader)
├── model/        # in-memory model (≈ Fortran modules/derived types)
├── starter/      # the Starter program        (≈ starter/source)
├── engine/       # the Engine program         (≈ engine/source)
├── elements/     # element kernels            (≈ engine/source/elements)
├── materials/    # material laws              (≈ engine/source/materials/mat)
├── failure/      # /FAIL damage criteria      (≈ engine/source/materials/fail)
├── contact/      # contact interfaces         (≈ engine/source/interfaces)
└── output/       # listings, TH, ANIM         (≈ engine/source/output)
paths.py         # the single external-path resolver (OR_SRC, OR_ROOT,
                  # hm_cfg_files, the RD-* deck corpus, the oracle binaries)
examples/         # ready-to-run input decks
tests/            # pytest suite incl. analytic validations
```

## License

**AGPL-3.0-or-later** (see [LICENSE](LICENSE)) — see `docs/LICENSING.md`
§Decision for the maintainer decision and its reasoning.

pyradioss is a literal transcription of OpenRadioss, which is
AGPL-3.0-or-later, © 2026 Siemens (Simcenter Radioss). Because the port copies
AGPL-covered expression, AGPL section 5(c) requires the whole work — this
repository as a whole, however it is packaged — to be licensed under the
AGPL, which it is. Copyright of the port itself is © 2026 Minh Quang Pham.

This is an independent port. It is not affiliated with, endorsed by, or
supported by Siemens; the upstream commercial alternative is Simcenter
Radioss.

## Reading order for agents

1. Read this `README.md` (mandatory if present).
2. `UPDATES.md` (mandatory) for the change history and current state before
   working.
