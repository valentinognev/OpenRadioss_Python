"""P0.12 -- the oracle records must describe the oracle that is INSTALLED.

``tools/validation_data/oracle_provenance.json`` and
``tools/validation_data/oracle_smoke.json`` are the two records that say WHICH
oracle produced WHICH evidence.  Every parity claim in this program inherits its
authority from them, so a record that names a binary nobody has is worse than no
record: it silently lends the weight of a measurement to an artefact that was
never measured.

This box was migrated and the oracle was rebuilt, which is exactly the event that
turns such a record into a lie.  Three things were true before this gate existed:

* ``oracle_smoke.json`` carried the **pre-rebuild** sha256 of both binaries, and
  **no committed test compared them** -- ``test_stored_golden_record_is_complete``
  only asserts ``assert entry["sha256"]`` (tests/test_p0_oracle_selftest.py:196),
  i.e. that the string is non-empty.  The suite was therefore green over a record
  describing binaries that no longer existed.
* ``oracle_provenance.json`` -> ``toolchain`` named a conda-forge gcc/gfortran
  15.2.0 under ``/home/valentin/anaconda`` -- a directory that does not exist on
  this box -- plus "12 cores" and ``system_gfortran: null``.
* nothing recorded *which machine* the block was talking about, so a reader could
  not tell a stale record from a provenance-of-origin one.

What must NOT change
--------------------
The **physics** evidence is untouched by the rebuild, and this gate is built so
that it says so out loud rather than assuming it.  Measured before anything was
edited, on the rebuilt binaries: three consecutive reference runs of the TENSILE
deck gave ``md5_norm = e3688899358f35e825cd640f9bd94964`` every time, which EQUALS
the committed golden digest in ``oracle_smoke.json`` -> ``t01.md5_normalized``.
So ``test_installed_oracle_reproduces_the_golden_t01_digest`` asserts the anchor
LIVE against the installed oracle, and
``test_golden_digest_matches_the_committed_artefact`` re-derives it from the
committed ``TENSILET01`` with no oracle at all.  If a rebuild ever moves that
digest, this gate fails and the fix is to re-derive the record and say why
(``environment.anchor_scope``), never to accept a new number silently.

Design rule: every assertion below must be able to FAIL.  A gate that passes on
the stale record as well as on the honest one is worthless, so the three that
carry the weight are marked WHICH-STALE-RECORD in their docstrings:

* :func:`test_recorded_oracle_binary_digests_are_the_installed_binaries` -- the
  recorded digest vs the bytes on disk, and the recorded path vs the resolved
  ``$OR_ROOT/bin/...``;
* :func:`test_toolchain_paths_are_real_tools_on_this_box` and
  :func:`test_toolchain_version_banners_are_the_measured_ones` -- every recorded
  tool path must exist AND its ``--version`` banner must be the recorded one;
* :func:`test_toolchain_host_facts_match_this_box` -- the recorded core count is
  ``os.cpu_count()``.

The provenance-of-origin escape hatch
--------------------------------------
A toolchain block is allowed to describe a machine OTHER than the one reading it
(provenance means "where it came from", not "where you are") -- but only when it
says so, loudly, in ``toolchain.machine_of_record``.  This box's oracle was built
here, so the honest value is ``null`` plus a reason; the branch exists so that a
future port cannot make the block describe a foreign toolchain by accident.  A
``null`` marker must carry its reason, so the marker can never be used to switch
the strict checks off silently.

Upstream Fortran origins ($OR_SRC = /home/valentin/Projects/OpenRadioss/OpenCourant):

* ``engine/source/output/th/hist1.F:210-234`` / ``engine/source/system/timer_c.c:30-40``
  -- the 24-character ``ctime()`` stamp the normalised digest zeroes; why the
  anchor is ``t01.md5_normalized`` and not ``hashlib.md5(t01 bytes)``.
* ``engine/source/output/th/hist1.F:212-217`` -- ``CH80(25:33)=' RADIOSS '``,
  ``CH80(34:59)=VERSIO(2)``, ``CH80(60:80)=CPUNAM``: the anchor is bound to the
  build AND the architecture.
* ``INSTALL.md:34-42`` / ``:95-115`` -- the environment the runs are made in.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SMOKE_JSON = REPO / "tools" / "validation_data" / "oracle_smoke.json"
PROVENANCE = REPO / "tools" / "validation_data" / "oracle_provenance.json"

#: The two binaries that are "the oracle".  Anything else in a record is not a
#: binary digest and must not be checked as one.
ORACLE_BINARIES = ("starter_linux64_gf", "engine_linux64_gf")

#: The tools ``toolchain`` is expected to name, and the flag that prints their
#: version banner.  ``gfortran``/``gcc``/``gxx``/``cmake``/``make``/``python3`` are
#: the ones the build actually used; ``system_gfortran`` is checked when it is
#: non-null (a box whose oracle was built by a conda compiler legitimately has
#: none, and says so in ``system_gfortran_reason``).
TOOL_KEYS = ("gfortran", "gcc", "gxx", "make", "cmake", "python3")

#: ``"<banner> - <absolute path>"`` is the convention the block already used for
#: every tool entry; commentary belongs in a sibling ``*_note`` key so the path
#: can be machine-checked.
_TOOL_RE = re.compile(r"^(?P<banner>.+?) - (?P<path>/\S+)$")

#: The key whose presence means "this block describes the machine the oracle was
#: BUILT, which is not necessarily the machine reading it".
MACHINE_OF_RECORD = "machine_of_record"
ORIGIN_OF_RECORD = "the machine where the oracle was BUILT"

ORACLE_DISABLED = os.environ.get("PYRADIOSS_ORACLE_DISABLED") == "1"
ORACLE_REQUIRED = os.environ.get("PYRADIOSS_ORACLE_REQUIRED") == "1"

_HINT = (
    "run tools/oracle/mirror_and_fetch.sh (acquires extlib) then "
    "tools/oracle/build_oracle.sh"
)

# The live-oracle gate: ONE definition, in tests/test_p0_oracle_build.py, shared
# by every oracle test module so the four cannot drift.  Its precedence is the
# one _hint() below already implements -- PYRADIOSS_ORACLE_DISABLED=1 silences,
# PYRADIOSS_ORACLE_REQUIRED=1 turns absence into a failure, a stale export (a
# variable set but resolving to nothing) is always a failure, otherwise skip with
# a reason naming what is missing and how to get it.
from tests.test_p0_oracle_build import _require_live_oracle  # noqa: E402

#: The digest the committed golden run must keep producing.  Written here as a
#: LITERAL so this gate can fail even when oracle_smoke.json has been edited: the
#: committed oracle selftest (tests/test_p0_oracle_selftest.py) pins the record's
#: own value, which a wrong edit could move in lockstep.  Preserved exactly --
#: the rebuild was measured to reproduce it, so it must never move here.
GOLDEN_T01_MD5 = "e3688899358f35e825cd640f9bd94964"


def _hint(reason: str):
    """One place for the DISABLED / REQUIRED / plain-skip decision.

    Same precedence as tests/test_p0_oracle_selftest.py and
    tests/test_p0_oracle_build.py: an operator's silence wins, the Phase 0 gate's
    ``PYRADIOSS_ORACLE_REQUIRED=1`` turns absence into a failure, otherwise skip
    with an actionable reason.
    """
    if ORACLE_DISABLED:
        pytest.skip("PYRADIOSS_ORACLE_DISABLED=1")
    if ORACLE_REQUIRED:
        pytest.fail(f"{reason} and PYRADIOSS_ORACLE_REQUIRED=1 is set: {_HINT}")
    pytest.skip(f"oracle not built ({reason}; {_HINT}); set "
                f"PYRADIOSS_ORACLE_REQUIRED=1 to enforce")


def _installed_binaries():
    """``{basename: Path}`` of the oracle binaries resolved by pyradioss.paths."""
    from pyradioss import paths

    resolved = {}
    for getter, name in ((paths.or_starter, "starter_linux64_gf"),
                         (paths.or_engine, "engine_linux64_gf")):
        try:
            resolved[name] = Path(getter())
        except FileNotFoundError as exc:
            _hint(f"the oracle binaries do not resolve ({exc})")
    return resolved


def _require_oracle(runtime_env=True):
    """The oracle is present -> ``{basename: Path}``; absent -> skip/fail.

    ``runtime_env`` is the shared gate's own flag (tests/test_p0_oracle_build.py
    defines it, this module imports it -- one definition, no second copy): True
    for the tests that LAUNCH the oracle, which cannot start without the writable
    mirror and its extlib reader library (``INSTALL.md:34-42``;
    ``tools/oracle/oracle_env.sh``: "LD_LIBRARY_PATH IS required, not cosmetic"),
    False for the tests that only hash the binaries on disk.  Without it the
    digest test below raised ``FileNotFoundError: OR_BUILD not found`` out of
    ``run_reference()`` and turned the bare fast tier red on a shell that had
    not exported the oracle variables -- a missing resource reported as a
    failure.  It is a presence probe: a configured-but-broken oracle passes it
    and the test that follows goes red on its own assertion.
    """
    _require_live_oracle(runtime_env=runtime_env)
    binaries = _installed_binaries()
    missing = [f"{name} ({path})" for name, path in binaries.items()
               if not path.is_file()]
    if missing:
        _hint(f"the oracle binaries are missing: {', '.join(missing)}")
    return binaries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _recorded_binary_digests(doc, name: str):
    """Every ``{path, sha256}`` pair in ``doc`` that names an oracle binary.

    Generic on purpose: a record that grows a new per-binary digest is covered
    automatically, and a record that DROPS one loses coverage -- hence the
    count assertion in the test below, which is what keeps the scan from
    silently degenerating into "nothing to check".
    """
    found = []

    def walk(node, trail):
        if isinstance(node, dict):
            path = node.get("path")
            sha = node.get("sha256")
            if (isinstance(path, str) and isinstance(sha, str)
                    and Path(path).name in ORACLE_BINARIES):
                found.append((f"{name}:" + ".".join(trail), path, sha))
            for key, value in node.items():
                walk(value, trail + [str(key)])
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, trail + [str(index)])

    walk(doc, [])
    return found


def _measured_banner(path: Path) -> str:
    """First line of ``<tool> --version``, or a readable failure string."""
    proc = subprocess.run([str(path), "--version"], capture_output=True,
                          text=True, timeout=60)
    out = (proc.stdout or proc.stderr or "").strip()
    return out.splitlines()[0] if out else f"<no output, rc={proc.returncode}>"


@pytest.fixture(scope="module")
def provenance():
    assert PROVENANCE.is_file(), f"missing {PROVENANCE}"
    return json.loads(PROVENANCE.read_text())


@pytest.fixture(scope="module")
def smoke():
    assert SMOKE_JSON.is_file(), (
        f"missing golden record {SMOKE_JSON}; regenerate it with "
        f"`python -m tools.oracle.oracle_selftest --write` after {_HINT}"
    )
    return json.loads(SMOKE_JSON.read_text())


# ---------------------------------------------------------------------------
# The binary digests -- the assertion that catches a stale record
# ---------------------------------------------------------------------------

def test_recorded_oracle_binary_digests_are_the_installed_binaries(provenance,
                                                                   smoke):
    """WHICH-STALE-RECORD: every recorded digest is the bytes on disk, now.

    ``oracle_smoke.json`` records a sha256 per oracle binary; after the rebuild
    that value described binaries that no longer existed and nothing noticed,
    because the old check was ``assert entry["sha256"]``.  Two things are pinned
    per entry, because either alone can be satisfied by a lie:

    * the recorded PATH is the binary ``pyradioss.paths`` resolves on this box,
      so the record cannot silently describe another machine's oracle; and
    * the recorded DIGEST is ``sha256`` of that file's bytes, computed here with
      hashlib -- never a constant copied out of the record under test.
    """
    binaries = _require_oracle(runtime_env=False)

    entries = []
    entries += _recorded_binary_digests(provenance, "provenance")
    entries += _recorded_binary_digests(smoke, "smoke")

    names = {Path(path).name for _, path, _ in entries}
    assert names == set(ORACLE_BINARIES), (
        f"the records must carry a sha256 for BOTH oracle binaries, found "
        f"{sorted(names)}. A dropped digest removes the only check in the "
        f"program that ties the record to the installed oracle. Entries: "
        f"{[e[0] for e in entries]}"
    )

    for where, path, recorded in entries:
        name = Path(path).name
        assert Path(path) == binaries[name], (
            f"{where} names {path} but the oracle resolves to "
            f"{binaries[name]} on this box: the record describes an oracle "
            f"that is not installed here"
        )
        measured = _sha256(binaries[name])
        assert recorded == measured, (
            f"{where} records the sha256 of a DIFFERENT {name}:\n"
            f"  recorded  {recorded}\n"
            f"  installed {measured} ({binaries[name]})\n"
            f"The binaries were rebuilt (or replaced) without the record "
            f"following them. Re-measure the record -- "
            f"`python -m tools.oracle.oracle_selftest --write` for the digests, "
            f"and state in the commit what changed. If only the linker's output "
            f"moved and the physics did not, t01.md5_normalized must be "
            f"unchanged ({GOLDEN_T01_MD5})."
        )


def test_both_records_agree_on_the_source_revision(provenance, smoke):
    """The two records must not disagree about WHICH upstream built the oracle.

    The rebuild changed the binaries, not the source: the same OR_SRC revision
    and the same declared extlib/shim set must be named on both sides, or one of
    the records is describing a different oracle than the other.
    """
    revision = provenance["upstream_git_sha"]
    assert provenance["upstream"]["git_sha"] == revision
    assert provenance["upstream"]["git_sha_short"] == revision[:9]
    assert smoke["oracle"]["upstream_git_sha"] == revision, (
        f"oracle_smoke.json cites upstream {smoke['oracle']['upstream_git_sha']}"
        f" while oracle_provenance.json cites {revision}: the two records "
        f"describe different oracles"
    )
    assert (REPO / smoke["oracle"]["provenance"]).resolve() == \
        PROVENANCE.resolve(), smoke["oracle"]["provenance"]

    # The provenance record names each solver's path; it must name the same file
    # the golden run used, and it must say the solver is built at all.  The
    # solvers block is keyed by the binary name (starter_linux64_gf), which is
    # what the smoke record's path basename gives us.
    for key in ("starter", "engine"):
        path = smoke["oracle"][key]["path"]
        entry = provenance["solvers"].get(Path(path).name)
        assert entry is not None, (
            f"oracle_provenance.json -> solvers has no entry for the binary the "
            f"golden run used ({path}); it names "
            f"{sorted(provenance['solvers'])}"
        )
        assert entry["built"] is True, f"solvers.{Path(path).name} says not built"
        assert entry["path"] == path, (
            f"solvers.{Path(path).name}.path {entry['path']} != the binary the "
            f"golden run used {path}"
        )


def test_provenance_names_only_paths_that_exist(provenance):
    """The record's own path fields must point at directories that exist here.

    Scoped to the fields that are *paths* (``upstream.*``, ``solvers.*.path``,
    ``oracle_smoke.json``'s binary and deck paths).  A blanket scan of every
    ``/...`` token in the file is NOT done on purpose: the prose cites deck
    keywords (``/PART``), URIs and ``$OR_BUILD`` templates that are not paths,
    and a check that cannot tell them apart would be either useless or noisy.
    """
    missing = []
    for key in ("read_only_path", "mirror_path"):
        path = Path(provenance["upstream"][key])
        if not path.exists():
            missing.append(f"upstream.{key} = {path}")
    for name, entry in provenance["solvers"].items():
        if not Path(entry["path"]).exists():
            missing.append(f"solvers.{name}.path = {entry['path']}")
    assert not missing, (
        "the provenance record names paths that do not exist on this box: "
        + "; ".join(missing)
        + ". After a migration these are the first thing that stops being true; "
        "either re-measure them or record them as provenance-of-origin with an "
        "explicit note."
    )


# ---------------------------------------------------------------------------
# The toolchain block
# ---------------------------------------------------------------------------

def _toolchain(provenance):
    block = provenance["toolchain"]
    assert block, "provenance: toolchain block missing"
    return block


def test_toolchain_paths_are_real_tools_on_this_box(provenance):
    """WHICH-STALE-RECORD: each named tool exists, or the block says it is from
    elsewhere.

    The stale block named ``/home/valentin/anaconda/bin/gfortran`` (a conda
    prefix the migration removed) as if it were on this box, with no marker
    saying it was describing another machine.  Here the oracle really is built
    with the system compilers, so the honest record is the real paths and
    ``machine_of_record: null`` -- and this test then holds them to the disk.

    The other accepted answer is the explicit origin record: a block whose
    ``machine_of_record`` is an object saying which machine it describes, with a
    reason.  That is legitimate (provenance means "where it came from"), and it
    is the only thing that makes a foreign path honest.
    """
    block = _toolchain(provenance)
    marker = block.get(MACHINE_OF_RECORD, "<absent>")

    if marker == "<absent>":
        pytest.fail(
            f"toolchain.{MACHINE_OF_RECORD} is absent. Either the block "
            f"describes this box (then it must be null, with a reason, and "
            f"every path must exist), or it describes the machine the oracle "
            f"was built on (then it must be an object saying so). An absent key "
            f"leaves a reader unable to tell a stale record from a foreign one."
        )

    if marker is not None:
        assert isinstance(marker, dict), (
            f"toolchain.{MACHINE_OF_RECORD} must be null or an object, got "
            f"{marker!r}"
        )
        assert marker.get("describes") == ORIGIN_OF_RECORD, (
            f"toolchain.{MACHINE_OF_RECORD}.describes must be "
            f"{ORIGIN_OF_RECORD!r} -- that phrase is what tells a reader the "
            f"paths below are NOT necessarily on the box they are reading"
        )
        assert marker.get("note"), (
            "an origin-of-record toolchain block must carry a note saying which "
            "machine and when"
        )
        return

    assert block.get(f"{MACHINE_OF_RECORD}_reason"), (
        f"toolchain.{MACHINE_OF_RECORD} is null but "
        f"toolchain.{MACHINE_OF_RECORD}_reason is missing: a null marker must "
        f"say why the block describes this box instead of a foreign one"
    )

    checked = 0
    for key in TOOL_KEYS + ("system_gfortran",):
        value = block.get(key)
        if value is None:
            continue
        match = _TOOL_RE.match(value)
        assert match, (
            f"toolchain.{key} = {value!r} does not follow the block's "
            f"convention '<version banner> - <absolute path>'. Machine-checking "
            f"the path is what makes a stale record fail."
        )
        path = Path(match.group("path"))
        assert path.exists(), (
            f"toolchain.{key} names {path}, which does not exist on this box "
            f"(measured). A record that names a tool nobody has lends its "
            f"authority to a build that did not happen here."
        )
        checked += 1

    assert checked >= len(TOOL_KEYS), (
        f"only {checked} of {len(TOOL_KEYS)} expected tools "
        f"{list(TOOL_KEYS)} are recorded in toolchain; a build record that "
        f"names none of them describes no toolchain"
    )

    # A system-path compiler IS the system compiler: recording
    # `system_gfortran: null` while gfortran is /usr/bin/gfortran contradicts
    # itself.
    system = block.get("system_gfortran")
    gfortran = _TOOL_RE.match(block["gfortran"]).group("path")
    if gfortran.startswith("/usr/"):
        assert system is not None, (
            "toolchain.gfortran is a system compiler but "
            "toolchain.system_gfortran is null: the block contradicts itself"
        )
    if system is None:
        assert block.get("system_gfortran_reason"), (
            "toolchain.system_gfortran is null with no reason"
        )


def test_toolchain_version_banners_are_the_measured_ones(provenance):
    """WHICH-STALE-RECORD: the recorded banners are what the tools print.

    Existence alone is not enough: a path can be right while the version beside
    it is the one from the machine before the migration.  So each recorded tool
    is executed and its first ``--version`` line must appear verbatim in the
    record.  The stale block recorded "GNU Fortran (GCC) 15.2.0" next to a conda
    path; on this box the same key must read GNU Fortran 13.3.0 from
    /usr/bin/gfortran.
    """
    block = _toolchain(provenance)
    if block.get(MACHINE_OF_RECORD) is not None:
        pytest.skip(
            f"toolchain.{MACHINE_OF_RECORD} declares this block describes the "
            f"BUILD machine, not this one; its banners are provenance, not a "
            f"claim about the box running the tests"
        )

    measured = []
    for key in TOOL_KEYS:
        value = block.get(key)
        if value is None:
            continue
        path = Path(_TOOL_RE.match(value).group("path"))
        assert path.exists(), (
            f"toolchain.{key} names {path}, which does not exist on this box: "
            f"its banner cannot be measured (see "
            f"test_toolchain_paths_are_real_tools_on_this_box)"
        )
        banner = _measured_banner(path)
        measured.append((key, banner))
        assert banner in value, (
            f"toolchain.{key} records {value!r} but {path} --version prints "
            f"{banner!r}. The recorded toolchain is not this box's."
        )
    assert measured, "no toolchain tool was checked"


def test_toolchain_host_facts_match_this_box(provenance):
    """WHICH-STALE-RECORD: the recorded core count is ``os.cpu_count()``.

    Cosmetic on its face, and that is the point: "12 cores" next to a 28-core
    box is the cheapest possible tell that a block was written before a
    migration, and a reader who notices one stale number should distrust the
    hashes beside it.
    """
    block = _toolchain(provenance)
    if block.get(MACHINE_OF_RECORD) is not None:
        pytest.skip("host facts belong to the declared build machine")

    cores = os.cpu_count()
    match = re.search(r"(\d+)\s+cores", block["host"])
    assert match, f"toolchain.host does not state a core count: {block['host']!r}"
    assert int(match.group(1)) == cores, (
        f"toolchain.host says {match.group(1)} cores, this box has {cores} "
        f"(os.cpu_count()): the block predates this machine"
    )
    assert platform_machine() in block["host"], (
        f"toolchain.host {block['host']!r} does not name the architecture "
        f"({platform_machine()})"
    )


def platform_machine() -> str:
    import platform

    return platform.machine()


def test_toolchain_absent_prefixes_are_recorded_as_absent(provenance):
    """A recorded "this used to be here" must still BE here -- i.e. gone.

    ``toolchain`` names the toolchain that built the oracle.  When that
    toolchain is a prefix the box no longer has, the honest thing is to say so
    with a machine-checkable fact rather than to leave the old paths in place.
    Each entry under ``absent_toolchains`` carries the path and the ``exists``
    flag that was measured; this asserts the flag still matches the disk, so the
    record cannot drift back into claiming a conda prefix it does not have.
    """
    block = _toolchain(provenance)
    absent = block.get("absent_toolchains", {})
    assert isinstance(absent, dict), absent
    for name, entry in absent.items():
        assert isinstance(entry, dict) and "path" in entry and "exists" in entry, (
            f"toolchain.absent_toolchains.{name} must record path + exists, got "
            f"{entry!r}"
        )
        assert isinstance(entry["exists"], bool)
        assert Path(entry["path"]).exists() is entry["exists"], (
            f"toolchain.absent_toolchains.{name}: the record says {entry['path']} "
            f"exists={entry['exists']}, the disk says "
            f"{Path(entry['path']).exists()}"
        )
        assert entry.get("reason"), (
            f"toolchain.absent_toolchains.{name} must say why it is absent"
        )


# ---------------------------------------------------------------------------
# The physics anchor -- what must NOT move
# ---------------------------------------------------------------------------

def test_golden_digest_matches_the_committed_artefact(smoke):
    """Re-derive the anchor from the committed T01: no oracle needed.

    Structural half of the "physics unchanged" claim.  It also pins the literal
    ``GOLDEN_T01_MD5`` this module asserts, so an edit of the record cannot move
    both at once and leave the gate green.
    """
    from tools.oracle.oracle_selftest import deterministic_md5

    blob = (REPO / smoke["t01"]["path"]).read_bytes()
    measured = deterministic_md5(blob)
    assert measured == smoke["t01"]["md5_normalized"], (
        f"the committed {smoke['t01']['path']} normalises to {measured} but the "
        f"record says {smoke['t01']['md5_normalized']}"
    )
    assert measured == GOLDEN_T01_MD5, (
        f"the golden T01 digest moved: {measured} != {GOLDEN_T01_MD5}. A "
        f"rebuild of the same source must not change it -- re-read the Fortran "
        f"before touching this constant."
    )


def test_installed_oracle_reproduces_the_golden_t01_digest(smoke,
                                                            tmp_path_factory):
    """Live: the INSTALLED binaries still produce the recorded physics.

    This is the half of the task that protects the evidence.  The records were
    corrected because the binaries were rebuilt; the rebuild must have preserved
    the physics, and the only way to say that is to RUN the installed oracle and
    compare a normalised digest.  One run of this deck costs ~0.4 s of solver
    time, so it stays in the default tier.
    """
    _require_oracle()
    from tools.oracle.oracle_selftest import run_reference

    ref = run_reference("TENSILE",
                        workdir=tmp_path_factory.mktemp("p0_12_reference"))
    assert ref["verdict"] == "NORMAL", (
        f"the installed engine did not terminate normally: {ref['verdict']!r}"
    )
    assert ref["t01_md5_normalized"] == smoke["t01"]["md5_normalized"], (
        f"the installed oracle's normalised T01 digest "
        f"{ref['t01_md5_normalized']} != the recorded "
        f"{smoke['t01']['md5_normalized']}. Correcting the record must NOT move "
        f"this number: if the oracle was legitimately rebuilt or moved to "
        f"another architecture, re-derive the record with --write and say why "
        f"(oracle_smoke.json -> environment.anchor_scope); treat it as a "
        f"physics regression only after reading the Fortran."
    )
    assert ref["t01_md5_normalized"] == GOLDEN_T01_MD5
    assert ref["n_cycles"] == smoke["run"]["n_cycles"]
    assert ref["t01_size_bytes"] == smoke["t01"]["size_bytes"]


def test_recorded_revision_is_the_checked_out_source(provenance):
    """The cited upstream revision is the HEAD of the read-only source tree.

    The rebuild moved the binaries but must not have moved the source; if OR_SRC
    is available, this is a two-second check that the provenance still points at
    the tree the oracle was compiled from.
    """
    from pyradioss import paths

    if not shutil.which("git"):
        pytest.skip("git is not installed")
    try:
        src = paths.or_src()
    except FileNotFoundError:
        pytest.skip("$OR_SRC does not resolve on this box")
    if not (src / ".git").exists():
        pytest.skip(f"{src} is not a git checkout (mirror-style tree)")

    proc = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=60)
    head = proc.stdout.strip()
    assert head == provenance["upstream"]["git_sha"], (
        f"oracle_provenance.json cites upstream "
        f"{provenance['upstream']['git_sha']} but {src} is at {head}: the "
        f"record describes a different source tree than the one on disk"
    )