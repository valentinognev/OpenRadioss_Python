"""Gate for P0.4 -- the native cmake build of the reference (Fortran) OpenRadioss.

Three layers, in decreasing order of "always applies":

1. Binary assertions (``test_oracle_binaries_exist``, ``test_oracle_binary_runs``)
   SKIP with an explicit reason while the oracle has not been built, and FAIL
   when ``PYRADIOSS_ORACLE_REQUIRED=1`` (the Phase 0 exit gate sets it).  They
   used to fail unconditionally, which turned the whole fast tier red for
   every other agent in the repo.
2. Acquisition assertions (``test_build_script_is_valid_bash``,
   ``test_provenance_names_the_extlib_source``) always run: the build script
   and the provenance record must exist whether or not the binaries do.
3. ``test_upstream_source_is_untouched`` always runs: ``$OR_SRC`` is read-only
   and must never be dirtied by any of this.

The binaries are resolved here rather than through ``oracle_env.sh`` because
pytest is not guaranteed to have it sourced.  Both exist AND run: the starter's
``-v`` path calls ``HM_BUILD_ID`` from ``libhm_reader``
(``starter/source/starter/execargcheck.F:1200-1201``), so an existence-only
check would pass for a binary that dies on the first call.

Upstream Fortran origins (repo-relative under the ``OpenCourant`` tree,
i.e. ``$OR_SRC`` = /home/valentin/Projects/OpenRadioss/OpenCourant):

* ``CMakeLists.txt:5-30``         -- ``-Dbuild=starter|engine|both`` switch and
  the ``set (EXEC_NAME ${starter})`` / ``set (EXEC_NAME ${engine})`` hand-off
  (a CMake variable, so EXEC_NAME is only set if the caller passes
  ``-Dstarter=``/``-Dengine=``; upstream precedent ``build_windows.bat:86-87,146``).
* ``starter/CMakeLists.txt:136-149`` -- ``arch`` default and the
  ``CMake_Compilers/cmake_${arch}.txt`` include that owns every compile/link
  flag, including the extlib library paths.
* ``starter/CMakeLists.txt:189-193`` -- the ``extlib`` custom target running
  ``load_extlib.py``; ``engine/CMakeLists.txt:266-270`` is the same target.
* ``starter/CMakeLists.txt:242-250`` / ``engine/CMakeLists.txt:347-355`` --
  ``add_executable``, ``add_dependencies(... extlib)`` and the POST_BUILD copy
  into ``${source_directory}/../exec`` (NOT into ``$OR_ROOT``).
* ``engine/CMakeLists.txt:335-340`` -- the CUDA ``*.cu`` glob is discarded when
  ``gpu_cc`` is undefined, so it stays empty without an NVIDIA SDK.
* ``INSTALL.md:34-42`` -- ``RAD_CFG_PATH`` / ``RAD_H3D_PATH`` /
  ``OMP_STACKSIZE`` / ``LD_LIBRARY_PATH``, the runtime environment.
* ``INSTALL.md:110-111`` -- the invocation and the binary names
  (``./starter_linux64_gf -i ... -np 1``) this test asserts.
* ``starter/source/starter/execargcheck.F:200`` (UPCASE, so lowercase ``-v``
  works), ``:1150-1163`` (``PEXECINFO`` -> ``MY_EXIT(0)``), ``:1203-1229`` (the
  starter banner); ``engine/source/engine/execargcheck.F:648-692`` and
  ``:726-742`` are the engine equivalents (upstream spells it "OpenRadios
  Engine").
* ``starter/CMakeLists.txt:210-217`` + ``Compiling_tools/script/or_build_info.py:40-47``
  -- ``build_info.inc`` is generated with ``-arch=${arch}``, hence the
  ``Platform release : linux64_gf`` assertion below.
"""

import json
import os
import pathlib
import re
import subprocess
from pathlib import Path

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_OR_ROOT = pathlib.Path("/home/valentin/OpenRadioss_or")
DEFAULT_OR_BUILD = DEFAULT_OR_ROOT / "source"
DEFAULT_OR_SRC = pathlib.Path("/home/valentin/Projects/OpenRadioss/OpenCourant")

OR_ROOT = pathlib.Path(os.environ.get("OR_ROOT") or DEFAULT_OR_ROOT)
OR_BUILD = pathlib.Path(os.environ.get("OR_BUILD") or DEFAULT_OR_BUILD)
OR_SRC = pathlib.Path(os.environ.get("OR_SRC") or DEFAULT_OR_SRC)

STARTER = pathlib.Path(
    os.environ.get("OR_STARTER") or OR_ROOT / "bin" / "starter_linux64_gf"
)
ENGINE = pathlib.Path(
    os.environ.get("OR_ENGINE") or OR_ROOT / "bin" / "engine_linux64_gf"
)

BUILD_SCRIPT = REPO / "tools" / "oracle" / "build_oracle.sh"
PROVENANCE = REPO / "tools" / "validation_data" / "oracle_provenance.json"

ORACLE_DISABLED = os.environ.get("PYRADIOSS_ORACLE_DISABLED") == "1"
ORACLE_REQUIRED = os.environ.get("PYRADIOSS_ORACLE_REQUIRED") == "1"

_HINT = (
    "run tools/oracle/mirror_and_fetch.sh (acquires extlib) then "
    "tools/oracle/build_oracle.sh"
)


def _require_oracle():
    """Skip when the oracle is absent, fail when the gate demands it."""
    if STARTER.is_file() and ENGINE.is_file():
        return
    if ORACLE_DISABLED:
        pytest.skip("PYRADIOSS_ORACLE_DISABLED=1")
    if ORACLE_REQUIRED:
        pytest.fail(
            f"oracle binaries missing ({STARTER}, {ENGINE}) and "
            f"PYRADIOSS_ORACLE_REQUIRED=1 is set: {_HINT}"
        )
    pytest.skip(f"oracle not built ({_HINT}); set PYRADIOSS_ORACLE_REQUIRED=1 to enforce")


def _runtime_env():
    """INSTALL.md:34-42, retargeted from OPENRADIOSS_PATH to the mirror.

    RAD_H3D_PATH is deliberately absent, exactly as in tools/oracle/oracle_env.sh:
    the reachable libh3dwriter.so is one parameter short of what the pinned source
    calls, and pointing the solvers at it would let a run write silently-wrong H3D
    files.  Without it h3d_dl.c:920-921 returns *IERROR = 1 and genh3d.F:729-731
    aborts any h3d run with MSGID 274.  See
    test_oracle_env_refuses_the_incompatible_h3d_writer.
    """
    env = dict(os.environ)
    env.pop("RAD_H3D_PATH", None)
    env["OPENRADIOSS_PATH"] = str(OR_BUILD)
    env["RAD_CFG_PATH"] = str(OR_BUILD / "hm_cfg_files")
    env["LD_LIBRARY_PATH"] = (
        str(OR_BUILD / "extlib" / "hm_reader" / "linux64")
        + os.pathsep
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["OMP_STACKSIZE"] = "400m"
    assert "RAD_H3D_PATH" not in env
    return env


def test_oracle_binaries_exist():
    _require_oracle()
    for binary in (STARTER, ENGINE):
        assert binary.is_file(), f"oracle binary missing: {binary}"
        assert os.access(binary, os.X_OK), f"oracle binary not executable: {binary}"


@pytest.mark.parametrize(
    "binary,banner,platform",
    [
        # starter: PREXECINFO prints CPUNAM from the tracked machine.inc
        # (starter/share/spe_inc/machine.inc:39,44 -> 'linux64' /
        #  'Linux 64 bits, GNU compiler'), not the generated BNAME.
        (STARTER, "OpenRadioss Starter", "Platform release : linux64"),
        # engine: PREXECINFO prints the generated BNAME
        # (engine/source/engine/execargcheck.F:729), which or_build_info.py
        # writes from -arch=${arch} (starter|engine CMakeLists.txt:210-217,
        # Compiling_tools/script/or_build_info.py:40-47).
        (ENGINE, "OpenRadios Engine", "Platform release : linux64_gf"),
    ],
    ids=["starter", "engine"],
)
def test_oracle_binary_runs(binary, banner, platform, tmp_path):
    _require_oracle()
    proc = subprocess.run(
        [str(binary), "-v"],
        cwd=tmp_path,
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"{binary.name} -v exited {proc.returncode}:\n{out}"
    assert banner in out, f"{binary.name} -v printed no version banner:\n{out}"
    assert platform in out, (
        f"{binary.name} -v did not report the expected platform:\n{out}"
    )
    # PREXECINFO calls HM_BUILD_ID from libhm_reader on the starter path
    # (starter/source/starter/execargcheck.F:1200-1221), so reaching this point
    # also proves LD_LIBRARY_PATH and RAD_H3D_PATH are right.
    assert "Reader :" in out or binary is ENGINE, (
        f"{binary.name} -v did not reach the hm_reader version query:\n{out}"
    )


def test_build_script_is_valid_bash():
    assert BUILD_SCRIPT.is_file(), f"missing build script: {BUILD_SCRIPT}"
    proc = subprocess.run(
        ["bash", "-n", str(BUILD_SCRIPT)], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"bash -n {BUILD_SCRIPT} failed:\n{proc.stderr}"


def test_provenance_names_the_extlib_source():
    assert PROVENANCE.is_file(), f"missing provenance record: {PROVENANCE}"
    data = json.loads(PROVENANCE.read_text())

    assert data.get("extlib"), "provenance must record an extlib entry"
    extlib = data["extlib"]
    for key in ("kind", "reason_upstream_url_unreachable", "sources"):
        assert key in extlib, f"provenance: extlib.{key} missing"
    # Every source must be pinned: a tag, a commit or a digest -- never "latest".
    for src in extlib["sources"]:
        assert isinstance(src, dict), f"provenance: bad source entry {src!r}"
        assert src.get("uri"), f"provenance: source without uri: {src!r}"
        pinned = src.get("tag") or src.get("commit") or src.get("digest")
        assert pinned, (
            f"provenance: source {src.get('uri')} is not pinned "
            "(need one of tag/commit/digest)"
        )
        assert "latest" not in str(pinned), f"provenance: unpinned tag {pinned!r}"

    # The fields every later parity claim cites. null is allowed, absent is not,
    # and a null must carry its reason (never invented).
    for key in ("upstream_git_sha", "toolchain", "built_on", "solvers"):
        assert key in data, f"provenance: top-level '{key}' missing"
    assert data["upstream_git_sha"], "provenance: upstream_git_sha must be filled"
    assert data["toolchain"].get("gfortran"), "provenance: toolchain.gfortran missing"
    assert data["toolchain"].get("cmake"), "provenance: toolchain.cmake missing"
    if data["built_on"] is None:
        assert data.get("not_built_reason"), (
            "provenance: built_on is null but not_built_reason is missing"
        )
    else:
        assert isinstance(data["built_on"], str), "provenance: built_on must be a date"
    for name, entry in data["solvers"].items():
        assert isinstance(entry, dict), f"provenance: solvers.{name} must be an object"
        assert "built" in entry, f"provenance: solvers.{name}.built missing"
        if entry["built"] is not True:
            assert entry.get("reason"), (
                f"provenance: solvers.{name} is not built, so it needs a reason"
            )


def test_provenance_declares_every_extlib_shim():
    """A stale extlib must be impossible to mistake for a pristine one.

    $OR_SRC/EXTLIB_VERSION.json pins the extlib the source expects; the oracle is
    built against whatever extlib is actually reachable.  When those differ, the
    provenance file must carry a non-empty ``extlib_shims`` section, one entry
    per adapted symbol, each naming its upstream caller(s) with file:line, the
    file in the mirror that provides it, and what is therefore untrustworthy.
    """
    data = json.loads(PROVENANCE.read_text())

    version_file = OR_SRC / "EXTLIB_VERSION.json"
    assert version_file.is_file(), f"cannot read {version_file}"
    wanted = int(json.loads(version_file.read_text())["version"])

    recorded = int(data["extlib"]["harvested_version"])
    shims = data.get("extlib_shims")

    if recorded >= wanted:
        # A current extlib needs no adapter; then there must be none to declare.
        assert not shims, (
            f"extlib_shims declared but the recorded extlib is v{recorded} "
            f">= the required v{wanted}"
        )
        return

    assert shims, (
        f"extlib on record is v{recorded} but the source requires v{wanted}: "
        "provenance must carry a non-empty extlib_shims section"
    )
    symbols = set()
    for shim in shims:
        for key in (
            "symbol",
            "upstream_callers",
            "provided_by",
            "trust_statement",
            "kind",
        ):
            assert shim.get(key), f"shim entry missing {key}: {shim!r}"
        assert shim["symbol"] not in symbols, f"duplicate shim {shim['symbol']}"
        symbols.add(shim["symbol"])
        assert shim["upstream_callers"], f"{shim['symbol']}: no caller citation"
        for caller in shim["upstream_callers"]:
            # must be a real path:line citation, not prose
            assert re.search(r"\.[FcChH]\w*:\d+", caller), (
                f"{shim['symbol']}: caller citation is not path:line -> {caller!r}"
            )
        assert Path(shim["provided_by"]).name, f"{shim['symbol']}: bad provided_by"

    # The three symbols the link failed on must be covered by name.
    for needed in (
        "cpp_get_include_file_by_index",
        "cpp_is_part_with_elements_",
        "cpp_sale_mesh_create_",
    ):
        assert needed in symbols, f"no shim recorded for {needed}"


def test_provenance_states_which_parity_evidence_is_admissible():
    data = json.loads(PROVENANCE.read_text())
    admissible = data.get("admissible_parity_evidence")
    assert admissible, "provenance must record admissible_parity_evidence"
    for channel, verdict in admissible.items():
        assert verdict in ("yes", "no"), f"{channel}: verdict must be yes/no, got {verdict!r}"
        if verdict == "no":
            assert data.get("inadmissible_parity_evidence_reasons", {}).get(
                channel
            ), f"{channel}: marked inadmissible but no reason recorded"


def test_oracle_starter_reads_a_multi_part_deck(tmp_path):
    """End-to-end gate on the reader path the hm_reader adapter feeds.

    ``cpp_is_part_with_elements_`` hands the caller's flag back through a
    ``logical(c_bool)``, which is ONE byte (gfortran: ``logical(c_bool)`` has
    kind 1 and storage_size 1, verified with a probe program).  An adapter that
    stored an ``int`` through it overwrote the three bytes above the flag -- in
    ``hm_read_part.F`` the part id itself -- so every /PART came out as
    ``PART: 0``, the deck was rejected with MSGERROR 494 + 402 and the starter
    died.  Both decks below reproduce that; both must now come out clean.

    Decks:
      * ``examples/tensile_bar/TENSILE_0000.rad`` -- vendored in this repo,
        upstream's own tensile example (the deck the reviewer reproduced the
        defect on).
      * ``tests/data/oracle/part_smoke_0000.rad`` -- SYNTHESIZED for this gate:
        the vendored tensile deck with explicit /PART/2 and /PART/3 blocks, so
        three distinct part ids must survive the round trip through the adapter.
    """
    _require_oracle()
    decks = [
        (REPO / "examples" / "tensile_bar" / "TENSILE_0000.rad",
         {1: "steel bar"}),
        (REPO / "tests" / "data" / "oracle" / "part_smoke_0000.rad",
         {1: "steel bar", 2: "steel bar half one", 3: "steel bar half two"}),
    ]

    for deck, expected_parts in decks:
        assert deck.is_file(), f"missing oracle fixture deck: {deck}"
        local = tmp_path / deck.name
        local.write_bytes(deck.read_bytes())
        proc = subprocess.run(
            [str(STARTER), "-i", local.name, "-np", "1"],
            cwd=tmp_path,
            env=_runtime_env(),
            capture_output=True,
            text=True,
            timeout=600,
        )
        listing_file = tmp_path / local.name.replace(".rad", ".out")
        assert listing_file.is_file(), (
            f"{deck.name}: starter wrote no listing file:\n"
            f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
        )
        listing = listing_file.read_text(errors="replace")

        errors = re.findall(r"^\s*(\d+)\s+ERROR\(S\)", listing, re.M)
        warnings = re.findall(r"^\s*(\d+)\s+WARNING\(S\)", listing, re.M)
        assert errors, f"{deck.name}: no ERROR(S) summary in the listing"
        assert int(errors[-1]) == 0, (
            f"{deck.name}: oracle starter reported {errors[-1]} ERROR(S):\n"
            + "\n".join(l for l in listing.splitlines() if "ERROR ID" in l)
        )
        assert warnings and int(warnings[-1]) == 0, (
            f"{deck.name}: oracle starter reported {warnings} WARNING(S):\n"
            + "\n".join(l for l in listing.splitlines() if "WARNING ID" in l)
        )

        # the exact symptom of a clobbered part id
        assert "PART WITH AN ID EQUAL TO 0 IS NOT ALLOWED" not in listing, (
            f"{deck.name}: a part id came back as 0 -- the one-byte flag write "
            "is broken again"
        )
        assert "-- PART ID: 0" not in listing, (
            f"{deck.name}: a part was reported with id 0"
        )

        # every declared part must appear with its own id and title, either in
        # the PART: block ("PART:         1,steel bar") or in the subset part
        # list ("                            2,steel bar half one")
        for part_id, title in expected_parts.items():
            assert f"{part_id},{title}" in listing, (
                f"{deck.name}: part {part_id} ({title!r}) missing from the listing"
            )



def test_provenance_only_claims_verified_evidence():
    """Every channel marked admissible must have been run, with its MSGERROR count.

    Marking T01/A-file/RESTART admissible while no deck had ever been read
    through the oracle is how a reader adapter can stay broken unnoticed.
    """
    data = json.loads(PROVENANCE.read_text())
    admissible = data["admissible_parity_evidence"]
    verified = data.get("evidence_verification")
    assert verified, "provenance must record evidence_verification"

    for channel, verdict in admissible.items():
        assert verdict in ("yes", "no"), f"{channel}: verdict must be yes/no, got {verdict!r}"
        if verdict != "yes":
            assert data.get("inadmissible_parity_evidence_reasons", {}).get(
                channel
            ), f"{channel}: marked inadmissible but no reason recorded"
            continue
        entry = verified.get(channel)
        assert entry, f"{channel}: admissible but nothing in evidence_verification"
        for key in ("deck", "msgerrors", "observed"):
            assert key in entry, f"{channel}: evidence_verification.{key} missing"
        assert isinstance(entry["msgerrors"], int), (
            f"{channel}: msgerrors must be a number, got {entry['msgerrors']!r}"
        )
        assert entry["msgerrors"] == 0, (
            f"{channel}: admissible but the recorded run reported "
            f"{entry['msgerrors']} MSGERROR(s)"
        )
        assert entry["deck"], f"{channel}: no deck named"


def _shim_source(shim):
    """Resolve the repo-owned source file a shim entry declares.

    ``provided_by`` is prose -- "<file in the repo> -> <path in the mirror>" --
    so it must NOT be reduced with ``Path(...).name``: that takes the LAST
    component, which for the compiled adapters is the archive
    (``libp0extlibshims.a``), not the source.  Doing so made every check below
    skip exactly the four adapters the liveness net exists to protect, and a
    deliberately bogus ``liveness_marker`` still passed.  The explicit
    ``source`` field is authoritative; ``provided_by`` is only a fallback.
    """
    declared = shim.get("source")
    if not declared:
        match = re.search(r"[\w./-]+\.[ch]", shim.get("provided_by", "") or "")
        declared = match.group(0) if match else None
    if not declared:
        return None
    path = Path(declared)
    return path if path.is_absolute() else REPO / path


def test_provenance_shims_are_live():
    """Every declared shim must resolve to a real file that really declares it.

    ``p0_h3d_writer_adapter.c`` once claimed to refuse h3d output; it never ran,
    because ``h3d_dl.c:984-1000`` already defines the same two wrappers, so the
    archive member was never pulled in and ``strings`` found no trace of it.  A
    shim that cannot be resolved to a source file, or that resolves to a file
    which does not mention the symbol, is just as dead.
    """
    data = json.loads(PROVENANCE.read_text())
    shims = data.get("extlib_shims") or []
    if not shims:
        return
    _require_oracle()

    checked = 0
    for shim in shims:
        src = _shim_source(shim)
        if shim.get("linked_into_binary"):
            # a linked adapter with no resolvable source cannot be checked, and
            # an unchecked adapter is exactly the dead-code case we are hunting
            assert src is not None and src.is_file(), (
                f"{shim['symbol']}: linked_into_binary is declared but no source "
                f"file could be resolved (source={shim.get('source')!r}, "
                f"provided_by={shim.get('provided_by')!r})"
            )
        if src is None or not src.is_file():
            continue
        body = src.read_text()
        checked += 1
        # every identifier the entry names must really be in the file (an entry
        # may name several, e.g. "Hyper3DElementBegin / Hyper3DElement2Begin")
        stopwords = {"prototypes", "shim", "macro", "declaration", "header"}
        for ident in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", shim["symbol"]):
            if ident.lower() in stopwords:
                continue
            assert ident in body, (
                f"{shim['symbol']} is declared but {src.name} does not mention {ident}"
            )
    assert checked == len(shims), (
        f"only {checked} of {len(shims)} declared shims resolved to a source file"
    )


def test_linked_adapters_are_present_in_the_binary():
    """The guarantee the provenance text claims, checked mechanically.

    A C adapter is compiled into ``libp0extlibshims.a`` and linked into the
    starter.  If its diagnostic marker is absent from the binary, the archive
    member was never pulled in and the adapter is dead code -- the failure mode
    that hid the h3d writer adapter.
    """
    data = json.loads(PROVENANCE.read_text())
    linked = [s for s in (data.get("extlib_shims") or []) if s.get("linked_into_binary")]
    if not linked:
        return
    _require_oracle()

    assert STARTER.is_file(), f"oracle starter missing: {STARTER}"
    blob = STARTER.read_bytes()
    if not blob:  # pragma: no cover - defensive
        blob = subprocess.run(
            ["strings", str(STARTER)], capture_output=True, text=True
        ).stdout.encode()

    for shim in linked:
        marker = shim.get("liveness_marker")
        assert marker, (
            f"{shim['symbol']}: linked_into_binary is set but no liveness_marker "
            "is declared"
        )
        assert marker.encode() in blob, (
            f"{shim['symbol']}: marker {marker!r} is NOT in the starter binary "
            "-- the adapter is dead code, not a safety net"
        )


def test_oracle_env_refuses_the_incompatible_h3d_writer():
    """The h3d safety net must be a real refusal, not a claim in a JSON file.

    ``libh3dwriter.so`` v59 is one parameter short of what the pinned source
    calls, so a run that writes H3D through it silently produces wrong files:
    measured, with ``RAD_H3D_PATH`` pointing at it, the tensile engine reaches
    NORMAL TERMINATION and writes ``TENSILE.h3d``.  The refusal is
    ``common_source/output/h3d/h3d_build_cpp/h3d_dl.c:920-921`` returning
    ``*IERROR = 1`` when the writer cannot be dlopen'ed at all, which
    ``engine/source/output/h3d/h3d_results/genh3d.F:729-731`` turns into
    MSGID 274 + ``ARRET(2)`` (measured: ``MESSAGE ID : 274 / ** ERROR: H3D
    EXTERNAL LIBRARY NOT FOUND``).  That only happens if nothing points the
    solver at the stale writer, so the environment script must not export
    ``RAD_H3D_PATH``.
    """
    env_script = REPO / "tools" / "oracle" / "oracle_env.sh"
    assert env_script.is_file(), env_script
    text = env_script.read_text()
    exported = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("export RAD_H3D_PATH")
    ]
    assert not exported, (
        "oracle_env.sh exports RAD_H3D_PATH "
        f"({exported}); the ABI-incompatible v59 writer would then be loaded and "
        "H3D files would be written through a one-parameter-short ABI"
    )
    # the reason must be written down where an agent will see it
    assert "genh3d.F:729-731" in text, (
        "oracle_env.sh omits RAD_H3D_PATH without citing the refusal path"
    )


def test_upstream_source_is_untouched():
    proc = subprocess.run(
        ["git", "-C", str(OR_SRC), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"git status failed on {OR_SRC}: {proc.stderr}"
    assert proc.stdout.strip() == "", f"$OR_SRC was modified:\n{proc.stdout}"
