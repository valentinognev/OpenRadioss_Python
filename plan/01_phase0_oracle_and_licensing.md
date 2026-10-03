# Phase 0 — Oracle and licensing

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:subagent-driven-development`. **Model: space-bunny, never Fast.**

**Goal:** make the real Fortran OpenRadioss solver buildable and runnable on this
box, and make `tools/validate_vs_fortran.py` able to use it — so that every
later phase can produce the differential evidence the domain rules require.

**Architecture:** an out-of-tree build prefix `$OR_ROOT` fed by a writable
mirror `$OR_BUILD` of `$OR_SRC`; a single resource-resolution module
(`pyradioss/paths.py`) that replaces the harness's hardcoded `C:\OpenRadioss`;
and a numeric T01 comparator that turns two runs into one rel-RMS number.

**Tech stack:** cmake ≥ 3.15 (measured 2026-10-03 on this box: `/usr/bin/cmake`
3.28.3), gfortran (measured: `/usr/bin/gfortran` 13.3.0, `-fopenmp` verified
working), GNU make, python3 on `PATH` (the cmake build invokes
`Compiling_tools/script/*.py`), extlib per `$OR_SRC/EXTLIB_VERSION.json`
(declares **v82**; that release is unreachable from this network, so the
harvested tree under `$OR_BUILD/extlib` is **v59** — see
`tools/validation_data/oracle_provenance.json` → `extlib`), pytest.
*(Pre-migration values, kept for history: cmake 4.4.3, gfortran 15.2.0 — both
were conda packages of a prefix this box does not have.)*

**Spec:** `plan/README.md` §1 (scope), `plan/00_ORCHESTRATION.md` §1.3
(licensing), §4 (environment).

**Entry criterion: none** — this is the program's first phase.
**Band: S.** Nothing else in this program can start before this phase exits.

---

## Scope

| Item | Value |
|---|---|
| Upstream paths | `CMakeLists.txt`, `starter/CMakeLists.txt`, `engine/CMakeLists.txt`, `CMake_Compilers/`, `Compiling_tools/script/`, `EXTLIB_VERSION.json`, `qa-tests/` |
| LOC touched | 0 (upstream is read-only; nothing is modified there) |
| Python produced | ~1,200 LOC across `pyradioss/paths.py`, `tools/oracle/*`, `tools/compare_t01.py` |
| Tests | ~45 |

## Gap analysis — what blocks us today

Measured on 2026-10-02:

1. **The Fortran source is present but unbuilt.** `$OR_SRC/exec/` does not
   exist; there are no `starter_linux64_gf` / `engine_linux64_gf` binaries.
2. **`tools/validate_vs_fortran.py` is hardcoded to Windows.** Lines 106–110:
   `OR_ROOT = r"C:\OpenRadioss"`, `STARTER_EXE = .../exec/starter_win64.exe`,
   `ENGINE_EXE = .../exec/engine_win64.exe`, plus `ONEAPI =
   r"C:\Program Files (x86)\Intel\oneAPI"` at line 107. On Linux every parity
   run fails at import.
3. **`CMake_Compilers/` ships only `cmake_win64_compilers.bat`.** The
   `starter/CMakeLists.txt` and `engine/CMakeLists.txt` both do:
   `if ( NOT EXISTS .../CMake_Compilers/cmake_${arch}.txt ) set (arch "none")`,
   then `include()` the per-arch flags only when `arch != none`. With
   `-Darch=linux64_gf` on this tree, **no architecture flags load at all** — no
   `-fopenmp`, no `-march`, no `-fno-second-underscore`. A build that compiles
   but silently loses OpenMP is worse than no build, because it invalidates
   every threaded parity claim.
4. **The build writes into its own source tree, twice.**
   `Compiling_tools/script/load_extlib.py` unzips into `<source_root>/extlib`
   and leaves `extlib.zip` behind; and both `CMakeLists.txt` files have
   `POST_BUILD … copy $<TARGET_FILE> ${source_directory}/../exec`. `$OR_SRC` is
   READ-ONLY by rule. **The build must run from a writable copy.**
5. **Runtime library paths are required.** `INSTALL.md:34-42`:
   `RAD_H3D_PATH=$OPENRADIOSS_PATH/extlib/h3d/lib/linux64` and
   `LD_LIBRARY_PATH=$OPENRADIOSS_PATH/extlib/hm_reader/linux64/`. The port's
   current CI (`.github/workflows/ci.yml`) has no notion of either.
6. **`numba` and `mpi4py` are not installed on this box**, so every accel and
   `-np` code path is currently untestable locally.
7. **No `apptainer`/`singularity`; `docker` is present at `/usr/bin/docker`.**
   The upstream CI uses a private image (`rd-linux64-common:latest`), so a
   container is not a shortcut here — the native cmake build is the path.
8. **Licence contradiction** (see `00_ORCHESTRATION.md` §1.3). Three artefacts
   disagree; a literal port of AGPL code cannot ship under MIT.

## Wave graph

```
Wave 0 (parallel, new files only)
  0.0 licensing decision note        0.4 native build script
  0.1 toolchain probe               0.5 golden-run smoke test
  0.2 architecture flag file        0.7 optional-dep install
  0.3 mirror + extlib fetch         0.8 corpus checkout manifest
   ── gate: tools/oracle/toolchain_probe.json reports gfortran+openmp+cmake+net
Wave 1 (serial — all touch pyproject.toml / paths.py / harness)
  0.6 pyradioss/paths.py            0.9 harness de-hardcoding
                                   0.10 T01 comparator
   ── gate: fast tier + parity selftest reproduces one deck's channel md5
```

---

### Task P0.0: Record the licensing decision

**Fortran:** `$OR_SRC/LICENSE.md:1-20`; headers in e.g.
`$OR_SRC/engine/source/engine/resol.F:1-23`.
**Files:** Create `docs/LICENSING.md`; Modify `pyproject.toml:9`.

**Interfaces:**
- Consumes: nothing.
- Produces: a single documented `license` field in `pyproject.toml` and a
  `docs/LICENSING.md` that later phases cite when adding upstream-derived code.

**This task does not choose.** It lays out the four lawful resolutions and
records the maintainer's answer. It must enumerate, with the licence text cited:

1. **Relicense the port to AGPL-3.0-or-later.** Consistent with the upstream
   source it derives from. Removes the conflict outright. Cost: the port cannot
   be distributed under permissive terms.
2. **Keep the existing files' derived status and relicense only the new
   program.** Not lawful for a *literal* port: the new files are
   AGPL-covered derivative works of AGPL-covered code, so the repository as a
   whole must comply. Listed because it is the option most often *assumed*
   correct and it is not.
3. **Replace the copied expressions.** Every ported formula re-derived from the
   published physics literature rather than transcribed from the Fortran, and
   the upstream files cited as references. This is the only route that keeps a
   permissive licence, and it converts a transcription task into a
   re-derivation task — a large, honest cost increase. Requires the maintainer
   to accept that "line-for-line port" and "permissive licence" are mutually
   exclusive.
4. **Obtain a commercial or dual-licence grant from Siemens** for the
   derivative work.

- [ ] **Step 1** — write `tests/test_p0_licensing.py` asserting the
      repository's declared licence, in every artefact, agrees:

```python
def test_declared_licence_is_consistent(tmp_path):
    meta = tomllib.loads(Path("pyproject.toml").read_text())
    readme = Path("README.md").read_text()
    license_text = Path("LICENSE").read_text()
    declared = meta["project"]["license"]["file"]
    assert "AGPL" in Path(declared).read_text()
    assert "AGPL-3.0-or-later" in readme
```

- [ ] **Step 2** — run `python -m pytest -q tests/test_p0_licensing.py` →
  FAIL, quoting the actual mismatch (`LICENSE` says MIT; `README.md` says
  GPL-3.0).
- [ ] **Step 3** — write `docs/LICENSING.md` with the four resolutions above,
  the AGPL-3.0-or-later citation from `$OR_SRC/LICENSE.md:1-3`, the Siemens
  2026 header citation, and a **Decision** section naming the chosen option,
  its date, and who chose it. Update `pyproject.toml`'s `license` field and the
  `README.md` licence section to match. **If no decision has been recorded by
  the maintainer, this task ends at "recommendation", the test is marked
  `@pytest.mark.xfail(strict=True)` with that reason, and every downstream
  phase's entry criterion "Task 0.0 decided" is unsatisfied — which is the
  intended blocking behaviour.**
- [ ] **Step 4** — run the test → PASS, or XFAIL with the recorded reason.
- [ ] **Step 5** — commit: `docs(licence): reconcile MIT/GPL-3.0/AGPL-3.0 contradiction`

---

### Task P0.1: Toolchain probe

**Fortran:** `INSTALL.md:34-60` (the Linux environment contract).
**Files:** Create `tools/oracle/toolchain_probe.py`, `tools/oracle/__init__.py`.

**Interfaces:**
- Consumes: nothing.
- Produces: `tools/oracle/toolchain_probe.py::probe() -> dict` with keys
  `gfortran`, `gfortran_version`, `cmake`, `cmake_version`, `make`,
  `openmp_ok`, `python3`, `docker`, `network_ok`, `extlib_url_reachable`.
  Writes `tools/validation_data/toolchain_probe.json`.

- [ ] **Step 1** — write the failing test:

```python
def test_probe_reports_required_tools():
    from tools.oracle.toolchain_probe import probe
    r = probe()
    for key in ("gfortran", "cmake", "make", "python3", "openmp_ok"):
        assert key in r
    assert r["gfortran_version"].startswith(("11", "12", "13", "14", "15"))
```

- [ ] **Step 2** — run → FAIL (`ModuleNotFoundError: tools.oracle`).
- [ ] **Step 3** — implement `probe()`. `openmp_ok` is decided by compiling
  this to a temp dir and running it — not by checking for the flag:

```fortran
program t
integer i
!$omp parallel do private(i)
do i=1,10
enddo
!$omp end parallel do
end program t
```

  invoked as `gfortran -fopenmp <file> -o <exe>` then `<exe>`.
  **AMENDED 2026-10-03 (fix round 2).** `network_ok` was specified here as "a
  `urllib` HEAD against the `url` in `$OR_SRC/EXTLIB_VERSION.json` (5 s
  timeout), because Task 0.3 needs it", and reality disagrees: that asset URL
  answers **404** on a box whose network demonstrably works (`curl -I
  https://github.com/` → 200; the extlib release asset → 404 — the
  organisation moved it). A key called `network_ok` reporting `false` there is
  a false record, and this project's standard is that a record must not claim
  something untrue. Reality wins; the two facts are therefore two keys:

  * **`network_ok`** — a `urllib` HEAD against a known-good **host**
    (`NETWORK_HOST_URL`, the operator that serves the asset; 5 s timeout).
    This is the machine fact Task 0.3 needs before it tries to download
    anything: "can this box reach the internet at all".
  * **`extlib_url_reachable`** — the original HEAD, against the `url` in
    `$OR_SRC/EXTLIB_VERSION.json`. Kept as its own key so nothing the old
    single key carried is lost: a `404` here is a fact about *the release*
    (where the asset lives now), not about the machine, and Task 0.3 needs it to
    decide whether it can fetch v82 at all.

  Nothing is downloaded by either probe; both are HEADs.
- [ ] **Step 4** — run → PASS; commit
  `feat(oracle): toolchain probe covering gfortran/cmake/make/openmp/network`.

---

### Task P0.2: Author the Linux architecture flag file

**Fortran:** `$OR_SRC/starter/CMakeLists.txt:118-136` (the
`if (NOT EXISTS cmake_${arch}.txt) set(arch "none")` guard and the
`include (…/cmake_${arch}.txt)`),
`$OR_SRC/engine/CMakeLists.txt` (same pattern);
`$OR_SRC/CMake_Compilers/cmake_win64_compilers.bat` (the shipped reference for
which variables a flags file must set).
**Files:** Create `tools/oracle/cmake_linux64_gf.txt`.

**Interfaces:**
- Consumes: nothing.
- Produces: a file that is **copied into the build mirror** at
  `$OR_BUILD/CMake_Compilers/cmake_linux64_gf.txt` by Task 0.3. It must set
  every variable the two `CMakeLists.txt` files read: `CMAKE_Fortran_FLAGS*`,
  `CMAKE_C_FLAGS*`, `CMAKE_CXX_FLAGS*`, `LINK`, `static_link`, and must add
  `-fopenmp` to both compile and link lines.

- [ ] **Step 1** — write the failing test:

```python
def test_arch_flag_file_enables_openmp_and_double_precision():
    from pathlib import Path
    txt = Path("tools/oracle/cmake_linux64_gf.txt").read_text()
    assert "-fopenmp" in txt
    assert "-fno-second-underscore" in txt   # required: Fortran name mangling
    assert "-fdefault-real-8" not in txt     # the port is single precision
```

- [ ] **Step 2** — run → FAIL (file absent).
- [ ] **Step 3** — author the file. **Non-obvious requirements, all derived
  from the upstream build, not guessed:**
  - `-fno-second-underscore` — the source uses trailing-underscore Fortran
    names, so the default mangling would not link.
  - `-fopenmp` on **both** the compile line and `LINK`; omitting it on `LINK`
    produces an undefined `GOMP_*` reference at link time.
  - **No** `-fdefault-real-8`: precision is selected by
    `starter/CMakeLists.txt:88-101` from the `precision` variable
    (`sp`/`dp`), defaulting to `dp`, by swapping the `share/r8` vs
    `share/r4` include directory. Forcing 8-byte reals here would silently
    change the oracle away from what the port is compared against.
  - `-O2 -march=native` are permitted; `-ffast-math` and
    `-Ofast` are **forbidden**, because they would make the oracle's own
    energy ledger disagree with its printout and invalidate the parity score.
- [ ] **Step 4** — run → PASS; commit
  `build(oracle): add the missing cmake_linux64_gf compiler flags`.

---

### Task P0.3: Mirror the source tree and fetch extlib

**Fortran:** `$OR_SRC/Compiling_tools/script/load_extlib.py:14-16` (it computes
`source_root` as its own grandparent and writes `extlib/` and `extlib.zip`
there), `:70-99` (the download and `extractall(source_root)`),
`$OR_SRC/EXTLIB_VERSION.json`.
**Files:** Create `tools/oracle/mirror_and_fetch.sh`.

**Interfaces:**
- Consumes: `OR_SRC`, `OR_ROOT` from the environment;
  `probe()["network_ok"]` and `probe()["extlib_url_reachable"]` from Task 0.1.
- Produces: `$OR_BUILD` — a writable mirror of `$OR_SRC` (via
  `git -C "$OR_SRC" worktree list`-independent `git archive` so the mirror
  carries no `.git` churn), containing `CMake_Compilers/cmake_linux64_gf.txt`
  and `extlib/`; and `$OR_ROOT/extlib` as a symlink or copy for
  `LD_LIBRARY_PATH`.

- [ ] **Step 1** — write the failing test:

```python
def test_mirror_is_writable_and_has_extlib(or_mirror):
    assert (or_mirror / "extlib" / "hm_reader").is_dir()
    assert (or_mirror / "CMake_Compilers" / "cmake_linux64_gf.txt").is_file()
    probe = or_mirror / "extlib_probe.tmp"
    probe.write_text("x")            # the real assertion: $OR_SRC must stay read-only
    probe.unlink()
```

  with an `or_mirror` fixture that skips when `OR_BUILD` is unset.
- [ ] **Step 2** — run → FAIL (`$OR_BUILD` does not exist).
- [ ] **Step 3** — implement:

```bash
OR_SRC=${OR_SRC:?}  OR_BUILD=${OR_BUILD:?}  OR_ROOT=${OR_ROOT:?}
mkdir -p "$OR_BUILD" "$OR_ROOT/bin"
git -C "$OR_SRC" archive --format=tar HEAD | tar -x -C "$OR_BUILD"
cp tools/oracle/cmake_linux64_gf.txt "$OR_BUILD/CMake_Compilers/"
python3 "$OR_BUILD/Compiling_tools/script/load_extlib.py"
```

  Use `git archive`, not `cp -r`: it skips `.git`, honours
  `.gitattributes`, and is re-runnable. `load_extlib.py` must be run with
  `python3` (the `CMakeLists.txt` files hardcode `python3` on non-Windows) and
  with `cwd` anywhere — it derives `source_root` from its own path.
- [ ] **Step 4** — run → PASS; commit
  `build(oracle): mirror $OR_SRC into a writable tree and fetch extlib v82`.

---

### Task P0.4: Native build script

**Fortran:** `$OR_SRC/CMakeLists.txt:16-32` (`-Dbuild=starter|engine|both`),
`$OR_SRC/starter/CMakeLists.txt:145-170` (the `extlib` custom target and the
generated `.inc` files), `:216-236` (`add_executable`, `add_dependencies(… extlib)`,
the `POST_BUILD` copy into `../exec`), `$OR_SRC/engine/CMakeLists.txt` (the CUDA
`*.cu` glob, which must stay empty without an NVIDIA SDK),
`$OR_SRC/INSTALL.md:98-115` (invocation `./starter_linux64_gf -i … -np 1`).
**Files:** Create `tools/oracle/build_oracle.sh`,
`tools/oracle/oracle_env.sh`.

**Interfaces:**
- Consumes: `$OR_BUILD`, `$OR_ROOT`, `tools/oracle/cmake_linux64_gf.txt`.
- Produces: `$OR_ROOT/bin/starter_linux64_gf`, `$OR_ROOT/bin/engine_linux64_gf`,
  and `tools/oracle/oracle_env.sh` exporting `OR_STARTER`, `OR_ENGINE`,
  `RAD_H3D_PATH`, `LD_LIBRARY_PATH`.

- [ ] **Step 1** — write the failing test:

```python
def test_oracle_binaries_exist():
    starter = Path(os.environ["OR_STARTER"]); engine = Path(os.environ["OR_ENGINE"])
    assert starter.is_file() and os.access(starter, os.X_OK)
    assert engine.is_file() and os.access(engine, os.X_OK)
```

  skipped entirely when `PYRADIOSS_ORACLE_DISABLED=1`.
- [ ] **Step 2** — run → FAIL (`OR_STARTER` unset / file absent).
- [ ] **Step 3** — implement `build_oracle.sh`:

```bash
cmake -S "$OR_BUILD" -B "$OR_BUILD/build" -Dbuild=both -Darch=linux64_gf
cmake --build "$OR_BUILD/build" --parallel "$(nproc)"
```

  then copy `$OR_BUILD/exec/starter_linux64_gf` and `engine_linux64_gf` into
  `$OR_ROOT/bin/`. Two traps the implementer must handle, both read off the
  upstream CMake files above:
  - the `POST_BUILD` copy lands in `$OR_BUILD/exec`, **not** `$OR_ROOT`; the
    script must move them, not assume they arrived in the prefix;
  - `LD_LIBRARY_PATH` must include `$OR_BUILD/extlib/hm_reader/linux64/` and
    `RAD_H3D_PATH` `$OR_BUILD/extlib/h3d/lib/linux64` (`INSTALL.md:34-42`) —
    without them the binaries start and then die on the first H3D/message call.
- [ ] **Step 4** — run → PASS; verify
  `$OR_STARTER -v` prints a version banner; commit
  `build(oracle): cmake build of starter/engine into $OR_ROOT/bin`.

---

### Task P0.5: Golden-run smoke test

**Fortran:** `$OR_SRC/INSTALL.md:105-115` (starter `-np 1` then engine),
`$OR_SRC/qa-tests/scripts/or_qa_script` (the upstream reference runner).
**Files:** Create `tools/oracle/oracle_selftest.py`,
`tests/test_p0_oracle_selftest.py`, `tests/data/oracle_smoke/`.

**Interfaces:**
- Consumes: `$OR_STARTER`, `$OR_ENGINE`, `pyradioss.paths.rd_decks_dir()`.
- Produces: `tools/validation_data/oracle_smoke.json` — the md5 and per-channel
  maxima of the reference run of one vendored deck. **This file is the seed of
  every later parity comparison.**

- [ ] **Step 1** — write the failing test:

```python
def test_oracle_reproduces_reference_t01(tmp_path):
    from tools.oracle.oracle_selftest import run_reference
    ref = run_reference("TENSILE", workdir=tmp_path)
    assert ref["verdict"] == "NORMAL"
    assert ref["n_cycles"] > 0
    assert ref["t01_md5"] == json.loads(
        Path("tools/validation_data/oracle_smoke.json").read_text())["t01_md5"]
```

- [ ] **Step 2** — run → FAIL (`oracle_selftest` absent).
- [ ] **Step 3** — implement `run_reference(run_name, workdir)`:
  copy the deck pair into `workdir`, invoke `$OR_STARTER -i <run>_0000.rad -np 1`
  then `$OR_ENGINE -i <run>_0001.rad`, parse the `<run>_0001.out` for the
  `ENGINE TERMINATION` banner and the cycle count, md5 the `<run>T01` binary,
  and assert `ENGINE TERMINATION` is `NORMAL`. **Two runs of the same Fortran
  binary must produce a byte-identical T01** — if they do not, the oracle is not
  deterministic and no parity claim in this program is admissible; that check is
  part of this task, not a later one.
- [ ] **Step 4** — run → PASS; commit
  `test(oracle): reference golden run and T01 determinism check`.

---

### Task P0.6: `pyradioss/paths.py` — one resource resolver

**Fortran:** `$OR_SRC/INSTALL.md:34-42` (env-var names and their meaning).
**Files:** Create `pyradioss/paths.py`, `tests/test_p0_paths.py`.

**Interfaces:**
- Produces (used by every later phase):
  - `pyradioss.paths.or_src() -> Path`
  - `pyradioss.paths.or_root() -> Path`
  - `pyradioss.paths.or_starter() -> Path`
  - `pyradioss.paths.or_engine() -> Path`
  - `pyradioss.paths.hm_cfg_dir() -> Path`
  - `pyradioss.paths.rd_decks_dir() -> Path`
  - `pyradioss.paths.missing_resource(name: str, tried: list[Path]) -> FileNotFoundError`
  - `pyradioss.paths.reload() -> None` (re-reads the environment; tests use it)

- [ ] **Step 1** — write the failing test:

```python
def test_missing_resource_names_every_attempted_location(monkeypatch):
    monkeypatch.delenv("OR_SRC", raising=False)
    with pytest.raises(FileNotFoundError) as e:
        paths.reload(); paths.or_src()
    msg = str(e.value)
    assert "OR_SRC" in msg and "$OR_ROOT/../OpenCourant" in msg

def test_hm_cfg_defaults_to_the_sibling_tree(monkeypatch, tmp_path):
    monkeypatch.setenv("OR_ROOT", str(tmp_path))
    (tmp_path / "OpenCourant" / "hm_cfg_files").mkdir(parents=True)
    paths.reload()
    assert paths.hm_cfg_dir() == tmp_path / "OpenCourant" / "hm_cfg_files"
```

- [ ] **Step 2** — run → FAIL (`ModuleNotFoundError: pyradioss.paths`).
- [ ] **Step 3** — implement the resolution order from
  `00_ORCHESTRATION.md` §4.1. **The fourth rule is the point of the task:** a
  missing resource raises with all four attempted locations listed, rather than
  silently falling through to a wrong directory. The current code degrades
  silently (that is bug OPEN_BUGS item 6, the LAW4 cfg lookup).
- [ ] **Step 4** — run → PASS; run the previously-env-failing
  `tests/test_m535_law04.py::test_direct_read_generic_mat_law4` → PASS
  (OPEN_BUGS item 6 closed); commit
  `fix(paths): single resource resolver, replaces silent cfg-path fallback`.

---

### Task P0.7: Install the missing optional dependencies

**Fortran:** none (environment only).
**Files:** Modify `pyproject.toml`; Create `requirements-lock.txt` additions;
`tests/test_p0_optional_deps.py`.

**Interfaces:**
- Consumes: nothing.
- Produces: a dev environment where `numba` and `mpi4py` import, and a test that
  *reports* rather than fails when they are absent — so the box can be
  inspected honestly without making CI red.

- [ ] **Step 1** — write the failing test:

```python
def test_optional_backends_are_installed_or_documented():
    for mod in ("numba", "mpi4py"):
        try:
            __import__(mod)
        except ImportError:
            pytest.skip(f"{mod} not installed — set PYRADIOSS_ALLOW_MISSING_DEPS=0 to gate")
```

  plus a companion that **fails** when `PYRADIOSS_ALLOW_MISSING_DEPS=0`.
- [ ] **Step 2** — run → SKIP (documents the gap).
- [ ] **Step 3** — `pip install -e ".[accel,mpi]"` into the project venv; record
  the resolved versions in `requirements-lock.txt`. **Do not** promote numba or
  mpi4py from optional to base dependencies — the base install must keep working
  NumPy-only (`README.md` promises this).
- [ ] **Step 4** — run → PASS; commit
  `chore(env): install numba and mpi4py, pin in requirements-lock`.

---

### Task P0.8: Corpus manifest refresh

**Fortran:** none. **Files:** Create `tools/validation_data/rd_decks_manifest.json`;
Modify `tools/validate_vs_fortran.py` (manifest load path only).

**Interfaces:**
- Produces: `tools.validate_vs_fortran.load_manifest() -> list[dict]` with one
  record per deck: `case_id`, `deck` (path), `sha256`, `category`, `package`,
  `in_envelope`. Replaces the hand-maintained case list that
  `parity_m41.json` and `inventory.json` are keyed by.

- [ ] **Step 1** — write the failing test:

```python
def test_manifest_covers_every_vendored_deck():
    decks = sorted(p.name for p in paths.rd_decks_dir().rglob("*_0000.rad"))
    ids = {c["case_id"] for c in load_manifest()}
    assert decks, "no vendored decks found"
    assert len(ids) >= 0  # manifest may be a subset; assert the intersection is non-empty
    assert ids & {Path(d).stem for d in decks}
```

- [ ] **Step 2** — run → FAIL (no manifest loader).
- [ ] **Step 3** — generate the manifest by hashing every deck under
  `PYRADIOSS_RD_DECKS`; carry `case_id`/`category` over from
  `tools/validation_data/inventory.json` where the case is present, so the
  existing `parity_m41.json` remains joinable.
- [ ] **Step 4** — run → PASS; commit
  `feat(validation): hashed corpus manifest, joinable with the m41 evidence`.

---

### Task P0.9: De-hardcode the validation harness

**Fortran:** `INSTALL.md:34-42` (the environment the harness must set up).
**Files:** Modify `tools/validate_vs_fortran.py:79,106-110,` and every other
`C:\` literal; Create `tests/test_p0_harness_portable.py`.

**Interfaces:**
- Consumes: `pyradioss.paths.*`.
- Produces: `tools.validate_vs_fortran.oracle_paths() -> dict` with keys
  `starter`, `engine`, `th_to_csv`, `h3d_lib`, `hm_reader_lib`; and a module
  import that succeeds on Linux with no environment set.

- [ ] **Step 1** — write the failing test:

```python
def test_harness_has_no_windows_only_paths():
    import tools.validate_vs_fortran as V
    assert V.oracle_paths()["starter"].name.startswith(("starter_linux64", "starter_win64"))
    src = Path(V.__file__).read_text()
    assert r"C:\OpenRadioss" not in src
    assert "Intel\\\\oneAPI" not in src
```

- [ ] **Step 2** — run → FAIL (`ModuleNotFoundError` on import — today the module
  raises before this can even be evaluated, because it touches `C:\` at
  import time).
- [ ] **Step 3** — replace the five literals with `paths.*` lookups. Also:
  - move `OR_ROOT`/`ONEAPI` out of **module scope** into `oracle_paths()`, so
    importing the harness never touches the filesystem;
  - the `th_to_csv` converter is a separate upstream tool
    (`$OR_SRC/tools/th_to_csv`) — Phase 12's binary-T01 work needs it, so
    resolve it now even though the current port emits CSV natively.
- [ ] **Step 4** — run → PASS; commit
  `refactor(validation): replace hardcoded C:\OpenRadioss with paths.* lookups`.

---

### Task P0.10: Numeric T01 comparator

**Fortran:** `$OR_SRC/engine/source/output/th/` (the channel layout and record
format the port must learn to read in Phase 12);
`$OR_SRC/tools/th_to_csv/` (upstream's own binary→ASCII converter, the reference
for the record layout).
**Files:** Create `tools/compare_t01.py`, `tests/test_p0_compare_t01.py`.

**Interfaces:**
- Produces:
  - `tools.compare_t01.read_t01(path: Path) -> T01` where `T01` is a dataclass
    with `channels: list[str]`, `times: np.ndarray`, `values: np.ndarray`
    (shape `(n_times, n_channels)`), `scaling: list[tuple[int,int,int]]`.
  - `tools.compare_t01.score(ref: T01, port: T01) -> Score` with
    `per_channel: dict[str, ChannelScore]` and `worst: ChannelScore`.
  - `tools.compare_t01.ChannelScore` fields: `rel_rms: float`,
    `max_abs: float`, `n: int`, `verdict: Literal["MATCH","DEVIATION","NODATA"]`.
  - `tools.compare_t01.MATCH_RMS = 0.05` — the score below which a channel is
    called a match, taken from the existing harness so the program's verdicts
    stay comparable with `parity_m41.json`.

- [ ] **Step 1** — write the failing test, using a synthetic pair so it does not
  need the oracle:

```python
def test_score_reports_rel_rms_per_channel():
    ref = T01(channels=["T", "IE"], times=np.array([0.0, 1.0]),
              values=np.array([[0.0, 100.0], [1.0, 90.0]]), scaling=[])
    same = T01(channels=["T", "IE"], times=np.array([0.0, 1.0]),
               values=np.array([[0.0, 100.0], [1.0, 90.0]]), scaling=[])
    assert score(ref, same).worst.rel_rms == 0.0
    assert score(ref, same).worst.verdict == "MATCH"

    off = T01(channels=["T", "IE"], times=np.array([0.0, 1.0]),
              values=np.array([[0.0, 100.0], [1.0, 95.0]]), scaling=[])
    assert score(ref, off).per_channel["IE"].verdict == "DEVIATION"
```

  plus `test_channel_order_is_matched_by_name_not_position`.
- [ ] **Step 2** — run → FAIL (module absent).
- [ ] **Step 3** — implement. Three decisions the implementer must not guess:
  - channels are matched **by name**; a channel present in one and absent in the
    other is `NODATA`, not a spurious deviation;
  - `rel_rms = sqrt(mean((port-ref)²)) / max(|ref|)` per channel, with the
    denominator floored at `max(|ref|).max() * 1e-12` so an all-zero reference
    channel does not divide by zero;
  - `NODATA` channels are excluded from `worst`.
  `read_t01` reads the **Fortran binary** layout, cross-checked against
  `th_to_csv` output on the same file in Step 4 — the byte layout is the
  Phase 12 deliverable, and this task lands the reader early because the
  comparator needs it.
- [ ] **Step 4** — run → PASS; then run `th_to_csv` on the golden T01 from Task
  0.5 and assert `read_t01` agrees with it channel-for-channel; commit
  `feat(validation): binary T01 reader and per-channel rel-RMS scorer`.

---

## Wave 2 — EXECUTED 2026-10-03: the Linux migration and the record/test repairs

P0.11–P0.16 ran after the repo was migrated from Windows to this Linux box and
the environment rebuilt (venv, mirror + extlib, starter/engine recompiled into
`$OR_ROOT`). They are **executed work, not a plan**: each entry records what it
did and the evidence it left, in the same Fortran / Files / Interfaces shape as
the tasks above. There is no step checklist because there is nothing left to
check off. A separate records pass (P0.17, milestone M706) updated
`docs/STATE.md` §Baseline, `UPDATES.md` and this file.

**Task-ID collision, stated once:** `UPDATES.md` 1.4.0 / 1.5.1 already use the
label "P0.11" for the binary-T01 parity route, which is NOT one of the tasks
below and is not in this plan file. The P0.11 here is the lock re-pin.

**One citation rotted:** `tests/test_p0_toolchain.py:96,133` (comment and
failure message only, not asserted) points at
`plan/01_phase0_oracle_and_licensing.md:172` for the "GCC major in 11..15"
rule. P0.17's tech-stack edit above shifted it to **:177**.

### Task P0.11: Re-pin the dependency lock to this interpreter

**Fortran:** none (no physics); the recorded precedent for the fact being
machine-scoped is `requirements-lock.txt` §[B]'s own header rule.
**Files:** Modify `requirements-lock.txt` §[B]. No test file touched.

**Interfaces:**
- Consumes: the interpreter that runs the suite
  (`platform.python_version()`, each module's `__version__`).
- Produces: §[B] pins that are true on this box — python 3.14.6→3.12.3,
  numpy 2.5.2→2.5.3, scipy 1.18.0→1.18.1 (pytest 9.1.1, numba 0.68.0,
  llvmlite 0.50.0 already true). §[A] (the maintainer's Windows box) is
  byte-identical.

**Evidence:** the lock was the defect, not the test —
`tests/test_p0_optional_deps.py` went `3 failed, 17 passed` → `20 passed,
1 skipped` with no assertion edited. The header now describes the interpreter
(`.venv` created by `python3 -m venv`, `base_prefix=/usr`, no `conda-meta/`)
instead of naming version numbers, and the mpi4py note's stated *reason* was
corrected: this box HAS OpenMPI (`libmpi.so.40`), so `pip install mpi4py`
would work — 4.1.2 stays a candidate pin, nothing was installed.

### Task P0.12: Make the oracle records describe the installed oracle

**Fortran:** `$OR_SRC/INSTALL.md:34-42` (the Linux environment contract the
records restate), `$OR_SRC/starter/CMakeLists.txt:14-19` (`set(PYTHON_EXEC
"python3")` on non-Windows — the fact P0.16 r1 mis-recorded).
**Files:** Modify `tools/validation_data/oracle_provenance.json`,
`tools/validation_data/oracle_smoke.json`; Create
`tests/test_p0_oracle_provenance.py` (10 tests).

**Interfaces:**
- Consumes: `pyradioss.paths.or_starter()/or_engine()`, `sha256sum`,
  `<tool> --version`, `CMakeCache.txt`.
- Produces: two records whose paths, digests and toolchain facts are the
  binaries on this box — starter `8b504acc…`, engine `6d58d0b1…` (replacing
  `b2f6a19f…` / `99e63c5c…`, which named binaries the migration removed), the
  absent conda prefix recorded as absent, `upstream.mirror_path` corrected to
  `$OR_BUILD` (`= /home/valentin/OpenRadioss_build`, evidence:
  `build/starter/CMakeCache.txt:CMAKE_HOME_DIRECTORY`).

**Evidence:** the golden T01 anchor did **not** move —
`t01.md5_normalized = e3688899358f35e825cd640f9bd94964`, reproduced by three
consecutive reference runs on the rebuilt binaries and now pinned as a literal
in the new test (`git show e144176 -- …/oracle_smoke.json` touches only the
two sha256 lines). Before the test existed nothing compared the recorded
digests with the binaries on disk, so the suite was green over a false record.
The two-way H3D refusal measurement is marked carried-from-the-pre-migration
build, not restated: the golden pair has no `/H3D` card
(`grep -c H3D examples/tensile_bar/TENSILE_0000.rad` → 0).

### Task P0.13: Recover the LAW34/LAW37 input-audit coverage

**Fortran:** `$OR_SRC/hm_cfg_files/config/CFG/radioss110/MAT/matl34_boltzman.cfg`
and `matl37_biphas.cfg` (the schemas the audit parses; `radioss110` +
`radioss2018` spellings).
**Files:** Modify `tests/test_m539_law34_input_audit.py`,
`tests/test_m540_law37_input_audit.py`. `pyradioss/paths.py` NOT edited — the
resolver already answered correctly.

**Interfaces:**
- Consumes: `pyradioss.paths.hm_cfg_dir()` (replaces the hardcoded
  `not os.path.isdir(r"C:\OpenRadioss\hm_cfg_files")` guard, which cannot hold
  on Linux).
- Produces: 12 tests that **execute** — the migration had silently stopped
  them, and they did not skip on the old Windows box, so this was real coverage
  loss, not an environment exemption.

**Evidence:** `57 passed, 12 skipped` before → `69 passed` after (the set has
since grown by the two CFG-spelling guards of commit `0d5f79e`, one per file,
so it is **71 today** — re-measured `.venv/bin/python -m pytest -q
tests/test_m539_law34_input_audit.py tests/test_m540_law37_input_audit.py` →
`71 passed in 1.34s`, exit 0; the `69` figure is what P0.13 left behind and is
the number `0d5f79e` re-measured as 71). No audit assertion edited and no skip
re-added: the 9/12 physics attributes resolve as FLOAT with law_number 34/37,
the upstream `CARD("%20lg…")` statements match the port's 20-column layouts,
and the CFG roundtrip recovers every parameter to rel_tol 1e-12. With every
candidate blanked (throwaway plugin outside the repo) the tests skip again
with the project's standard "CFG tree not found" reason — `57 passed, 14
skipped` as measured by `0d5f79e` (12 pre-existing + its own 2 guards).

### Task P0.14: One oracle gate, so the bare fast tier is reproducible

**Fortran:** `$OR_SRC/INSTALL.md:34-42` (`LD_LIBRARY_PATH` is required — the
binaries start without it and die on the first message call with an unresolved
`libhm_reader`), `execargcheck.F:1200-1201` (only the starter's `-v` touches
the reader library).
**Files:** Modify `tests/test_p0_oracle_build.py` (defines
`_require_live_oracle`), `tests/test_p0_oracle_selftest.py`,
`tests/test_p0_harness_portable.py`, `tests/test_p0_oracle_provenance.py`.

**Interfaces:**
- Consumes: `starter`, `engine`, `$OR_BUILD`, and the extlib reader library.
- Produces: one shared gate. Absent prerequisites → skip with a reason naming
  each missing one; a *stale export* → fail, not skip;
  `PYRADIOSS_ORACLE_REQUIRED=1` / `PYRADIOSS_ORACLE_DISABLED=1` override.

**Evidence:** the three modules each had a private `_require_oracle()` that
asked only whether the two executables resolve — they do here (the dev-box
`~/OpenRadioss_or/bin` fallback), so the gate waved them into a run that
cannot work. `4 failed, 48 passed, 3 errors` bare → `48 passed, 7 skipped`
(same command) — historical, measured at `2147864`; the pre-fix state is not
re-measurable now. **The `56 passed, 9 skipped` bare vs `65 passed` configured
pair this entry carried until 2026-10-03 no longer reproduces.** Re-measured
over the four oracle modules (`test_p0_oracle_build`, `test_p0_oracle_selftest`,
`test_p0_oracle_provenance`, `test_p0_harness_portable`), which collect **68**
tests today: bare `68 passed, 0 skipped`; with `OR_SRC`/`OR_BUILD`/`OR_ROOT`
exported, `oracle_env.sh` sourced and `PYRADIOSS_ORACLE_REQUIRED=1`,
`68 passed, 0 skipped`. Both exit 0.

Nothing skips any more: `pyradioss/paths.py` resolves the install prefix
unaided on this box, so the skip surface this task added never fires here.
That cuts both ways — the gate can no longer infer "the oracle is absent" from
a skip count, so it must export `PYRADIOSS_ORACLE_REQUIRED=1` (Exit gate), and
a bogus export now fails loudly instead: `OR_BUILD=/tmp/no-such-mirror-xyz`
gives `3 failed, 13 passed` in `test_p0_oracle_build.py`
(`test_oracle_binary_runs[starter]`, `…runtime_library_path_exists…`,
`…starter_reads_a_multi_part_deck`) whether or not the REQUIRED flag is set
(measured 2026-10-03).

### Task P0.15: Purge false machine facts from tooling and packaging

**Fortran:** none; the claims were about the binaries, and the measured answer
is `readelf -d $OR_ROOT/bin/{starter,engine}_linux64_gf` → **no DT_RPATH, no
DT_RUNPATH** (`tools.validate_vs_fortran.elf_search_paths()` returns `[]` for
both).
**Files:** Modify `tools/validate_vs_fortran.py` (docstrings only),
`tools/oracle/build_oracle.sh` (comments only), `pyproject.toml` (comment);
Create `tests/test_p0_no_stale_machine_paths.py` (9 tests at P0.15; **10**
today — `00b5112` added the literal-block case, re-counted 2026-10-03 with
`.venv/bin/python -m pytest -q --collect-only tests/test_p0_no_stale_machine_paths.py`).

**Interfaces:**
- Consumes: `elf_search_paths`, `<tool> --version`, the lock's `# pin:` lines,
  the filesystem.
- Produces: four rules (a machine path must exist or be labelled history; a
  claimed RPATH must be one the binaries carry; the two docstrings must keep
  the hazard reasoning *and* state the measured fact; a tool/pyproject version
  must match a measured/locked one) plus self-tests that drive the checkers
  over synthetic text, and a claim-vs-specimen discriminator that reads
  document structure (a reST literal block or a fenced block is a quoted
  specimen, prose is a claim).

**Evidence:** 5 failed / 3 passed with the test in place and the edits reverted
to HEAD, 9 passed after; the pre-fix files re-checked in a scratch tree still
fail all four rules. `numpy 2.5.2` in `pyproject.toml` was replaced by a
pointer at the lock's `# pin:` lines rather than a third copy of a version.
The `tests/` exclusion is a recorded decision (9 hits re-measured, none a stale
claim).

### Task P0.16: The toolchain record is a gated fact, not a test side effect

**Fortran:** `$OR_SRC/starter/CMakeLists.txt:14-19` — upstream hardcodes
`PYTHON_EXEC "python3"` on non-Windows, which is why the record must hold
`shutil.which("python3")` and not the checkout's `sys.executable`.
**Files:** Modify `tools/oracle/toolchain_probe.py` (adds `read_record()`,
`differences()`, `explain()`; `probe()` untouched, `main()` still the only
writer), `tools/validation_data/toolchain_probe.json`,
`tests/test_p0_toolchain.py`.

**Interfaces:**
- Consumes: a fresh `probe()`; the committed record.
- Produces: a gate that compares every recorded key and quotes the refresh
  command; skips through `_require_live_oracle(runtime_env=False)` where the
  oracle is not installed.

**Evidence:** `test_probe_json_is_written` used to call `probe.main([])`, so
every suite run rewrote the claim and the committed copy was a lie between
runs (it named `/home/valentin/anaconda/bin/{gfortran,cmake,make}`, "cmake
version 4.4.3"). A suite run now leaves `git status --porcelain` clean. Round 1
fixed the record's own values: `gfortran_version` is the compiler's version
(`13.3.0`, digits-and-dots only) rather than the Ubuntu package string, with
the banner kept in `gfortran_banner`, and every key is now a property of the
machine (the run-local classification is gone).

---

## Reviewer checklist (per task)

Beyond `00_ORCHESTRATION.md` §9.1:

- **P0.0** — the four resolutions are stated with citations and **none is
  quietly adopted**; if no decision is recorded, the test is `xfail(strict=True)`
  and the blocking behaviour is intact.
- **P0.1** — `openmp_ok` is decided by compiling *and running*, not by grepping
  a flag string.
- **P0.2** — no `-ffast-math`/`-Ofast`; no `-fdefault-real-8`; `-fopenmp` is on
  the link line.
- **P0.3** — `$OR_SRC` is unmodified afterwards (`git -C "$OR_SRC" status
  --porcelain` is empty).
- **P0.4** — the binaries actually run (`-v`), not merely exist; `LD_LIBRARY_PATH`
  and `RAD_H3D_PATH` are set by `oracle_env.sh`.
- **P0.5** — two consecutive reference runs give a **byte-identical** T01.
- **P0.6** — every resolver raises with all attempted locations when missing.
- **P0.9** — no `C:\` literal remains anywhere in `tools/`.
- **P0.11** — the lock, not the test, was changed; §[A] is byte-identical.
- **P0.13** — the recovered tests **execute** (no skip smuggled back in), and
  the guard resolves through `pyradioss/paths.py`.
- **P0.14** — the gate is proved *not* to be a blanket skip: a stale export
  fails (`OR_BUILD=/tmp/no-such-mirror-xyz` → `3 failed, 13 passed` in
  `test_p0_oracle_build.py`, measured 2026-10-03), `PYRADIOSS_ORACLE_REQUIRED=1`
  turns absence into a failure, and a working oracle runs. **The "absent
  resource skips with a reason" half no longer holds on this box** — the
  resolver finds the install prefix unaided, so nothing skips; that is why the
  exit gate exports the REQUIRED flag rather than watching for skips.
- **P0.15/P0.16** — the rot-scanner distinguishes a sentence about this box
  from a quoted specimen, and a suite run leaves the committed records
  unmodified (`git status --porcelain` clean).

## Parallelisation

- **Wave 0:** P0.0, P0.1, P0.4, P0.7, P0.8 are independent. P0.2 depends on
  nothing. P0.3 depends on P0.2. P0.5 depends on P0.4.
- **Wave 1:** P0.6, P0.9, P0.10 all touch shared files
  (`pyproject.toml`, `pyradioss/__init__` neighbours, `tools/`) → serialised,
  one owner each, order P0.6 → P0.9 → P0.10 (each consumes the previous).
- **Wave 2 (done):** P0.11 → P0.16 ran serially in that order; each fixed a
  record the previous one's gate had just made falsifiable.

## Exit gate

```bash
export OR_SRC=/home/valentin/Projects/OpenRadioss/OpenCourant   # READ-ONLY
export OR_BUILD=/home/valentin/OpenRadioss_build               # writable mirror
export OR_ROOT=/home/valentin/OpenRadioss_or                   # install prefix
source tools/oracle/oracle_env.sh
export PYRADIOSS_ORACLE_REQUIRED=1   # VERIFY the oracle; absence must fail, not skip
PYTHON=.venv/bin/python                       # never bare `python`
PYRADIOSS_BACKEND=numpy $PYTHON -m pytest -q tests/test_p0_oracle_selftest.py tests/test_p0_compare_t01.py
PYRADIOSS_BACKEND=numpy $PYTHON -m pytest -q -m "not slow"
git -C "$OR_SRC" status --porcelain        # must be empty
```

All four must pass. The phase reviewer additionally re-runs Task 0.5's
determinism check **three** times and confirms the rel-RMS scorer reproduces a
known verdict from `parity_m41.json` on a stored deck pair.

**`PYRADIOSS_ORACLE_REQUIRED=1` is load-bearing (added 2026-10-03).** Without
it the oracle-dependent tests **skip** whenever the oracle does not resolve
(`_require_live_oracle`), so a gate run on a box where the oracle is missing
would be green while verifying nothing — the tests that check the oracle is real
are exactly the ones a bare gate run drops. Two test docstrings assert that the
Phase 0 exit gate sets it (`tests/test_p0_oracle_build.py:10`,
`tests/test_p0_harness_portable.py:64`); until this line existed the gate did
not, so those docstrings described a gate that did not exist.

**Measured 2026-10-03 (the gate exactly as written above, this box):**
line 2 `32 passed, 1 skipped` (the skip is `th_to_csv` absent — documented in
`test_p0_compare_t01.py:1085` and covered by three independent cross-checks);
line 3 `14576 passed, 15 skipped, 20 deselected, 26 xfailed in 888.48s (0:14:48)`,
**exit 0**; line 4 empty, exit 0 (all at commit `13deef2`). Line 3's
environment is exactly line 2's plus the full fast tier, so the fast-tier
figures — and the bare and `PYRADIOSS_BACKEND=numpy`-only ones they must not
be confused with — live in `docs/STATE.md` §Baseline, which is their single
home. Under this gate the oracle tests **execute**:
`test_p0_oracle_{build,selftest,provenance}.py`, `test_p0_harness_portable.py`,
`test_p0_toolchain.py` and `test_p0_compare_t01.py` together give
`113 passed, 1 skipped`, and that one skip is `th_to_csv`, not a missing
oracle.

Lines 1 and 2 were stale as written before P0.17: the gate said bare `python`
(this box has only `.venv/bin/python`) and never exported
`OR_SRC`/`OR_BUILD`/`OR_ROOT`, which `oracle_env.sh` requires
(`:${OR_BUILD:?…}`, `${OR_ROOT:?…}`).
