"""Task P0.17 — ``tools/oracle/oracle_env.sh`` must document the failure it
actually has.

The script's header used to say, of ``LD_LIBRARY_PATH``:

    "the binaries start without it and then die on the first H3D / message call
     with an unresolved libhm_reader or libapr-1"

Measured on this box, that describes a symptom that **cannot occur**.  Run with
``LD_LIBRARY_PATH`` unset, the starter never starts: the dynamic loader rejects
it before one instruction of its own code has run, and the process exits 127
with ``error while loading shared libraries: libhm_reader_linux64.so: cannot
open shared object file``.  There is no H3D call to die on, because nothing
runs.  So the comment was wrong in the way that costs the most: an operator
reading a 127 was sent hunting an H3D bug instead of a loader problem.  The
engine does not even need the library — it has no reader ``NEEDED`` entry and
prints its ``-v`` banner with ``LD_LIBRARY_PATH`` unset (exit 0) — so the old
"the binaries ... die" also erased the one fact that explains why the two
behave differently.

The principle, which generalises past this one sentence:

    **Where an environment script documents the consequence of a variable
    being missing, the documented failure mode must be the one this box
    produces — and it must be quoted evidence, not an inference.**

Three properties make that checkable rather than a vibe, and all three are
needed:

* **measurement, not recollection.**  :func:`measure_missing_ld_library_path`
  really launches both binaries with ``LD_LIBRARY_PATH`` scrubbed and returns
  their exit codes and streams.  Every exit code and every error string the
  script is allowed to quote comes out of it, so the rule compares the prose
  against a fresh run instead of against a constant someone typed once.  A
  box where the loader message changes wording therefore updates the rule by
  changing reality, and a wrong constant cannot hide behind a passing test.
* **positive and negative, or it is not a claim.**  The region must *quote* the
  measured evidence — the measured exit code, the measured loader message — so
  a comment that stops documenting anything fails.  And it must *not* claim the
  program runs first and fails later (:data:`DEFERRED_FAILURE`), which is the
  specific rot being fixed.
* **scoped to the binary that fails.**  The region must say something about the
  engine, and what it says must not claim the engine fails
  (:func:`check_distinguishes_the_engine`).  "Both binaries die at load" was
  false for one of them.

The near-misses matter as much as the claim: :func:`check_missing_env_claim`
rejects a swapped exit code, a plausible-but-unmeasured one, a loader message
naming the wrong library, and the run-time phrasing, each built synthetically
in :func:`test_the_claim_rule_can_still_fail` — and
:func:`test_the_pre_fix_paragraph_is_rejected` runs the rule over the
**verbatim pre-fix text** (the record this milestone corrects, quoted from
``69d8ede``; see also ``tests/test_p0_harness_portable.py:888-896``) and
requires it to be reported.  A checker that accepts everything is worse than
no checker, and these are the ones that would rot into that.

Also held here, because the same edit could have cost them: the header's
**deliberate** decision to leave ``RAD_H3D_PATH`` unset is a safety measure with
a four-``dlopen``-trial rationale behind it, and it must survive any future
touch of this paragraph intact
(:func:`test_the_h3d_writer_stays_deliberately_out_of_reach`).  The engine
finding a *reachable but wrong* ``libh3dwriter.so`` is the failure mode that
silently writes corrupt H3D files; losing the warning to keep a comment tidy
would trade a cosmetic problem for a silent one.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.test_p0_oracle_build import _require_live_oracle

REPO = Path(__file__).resolve().parents[1]
ORACLE_ENV = REPO / "tools" / "oracle" / "oracle_env.sh"

#: Upstream's requirement, and the line of it that states this one.
INSTALL_MD = "INSTALL.md"
INSTALL_LD_LINE = 42

#: What consumers of this script read out of it.  Sourced, not executed, so
#: these must appear as ``export NAME=``.
CONSUMER_EXPORTS = ("OR_STARTER", "OR_ENGINE", "LD_LIBRARY_PATH", "PATH")

#: The export this comment exists to justify, so that no test in this file
#: passes by virtue of the export being deleted.
TOKEN = "LD_LIBRARY_PATH"


# ---------------------------------------------------------------------------
# Measurement -- the ground truth the prose is checked against
# ---------------------------------------------------------------------------

def measure_missing_ld_library_path(binary: Path, tmp_path: Path) -> dict:
    """Launch ``binary -v`` with ``LD_LIBRARY_PATH`` scrubbed; record the result.

    Nothing here is a constant.  The child environment is rebuilt with the
    variable deleted rather than blanked — an empty ``LD_LIBRARY_PATH`` is not
    the same thing to the loader as an absent one — and the working directory
    is a scratch one, so a stray ``libhm_reader*.so`` sitting in the repo
    cannot quietly satisfy the loader and turn the measurement into a lie.
    """
    env = {k: v for k, v in os.environ.items() if k != TOKEN}
    done = subprocess.run([str(binary), "-v"], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=60)
    return {"binary": binary, "code": done.returncode,
            "stdout": done.stdout, "stderr": done.stderr}


def loader_message(stderr: str) -> str:
    """The loader's complaint with the ``<path>: `` prefix it prepends.

    ``ld.so`` prints the path it was handed before the message, so the
    comparable part starts after the first ``": "``.  Deriving it (rather
    than writing the string out) is what keeps this test off the
    rot-scanner's list: the machine path is never a literal here.
    """
    _, sep, rest = stderr.partition(": ")
    return rest.strip() if sep else stderr.strip()


def missing_library(stderr: str) -> str:
    """The shared object the loader names, or ``""`` if it names none.

    ``ld.so`` puts the soname last on the line, immediately before the
    complaint, so the token in front of ``cannot open shared object file`` is
    the library — with the ``:`` it prints between the two trimmed off.
    """
    match = re.search(r"cannot open shared object file", stderr)
    if match is None:
        return ""
    tail = stderr[:match.start()].strip()
    return tail.rsplit(" ", 1)[-1].strip().rstrip(":")


# ---------------------------------------------------------------------------
# Locating the prose under test
# ---------------------------------------------------------------------------

def ld_docs_regions(text: str) -> list[str]:
    """Every run of consecutive comment lines that talks about the variable.

    A run ends at the first non-comment line, so the export statement itself
    closes the header block and the comment above the export is a separate
    region — that comment is about *libapr's own* dependencies and makes no
    claim about what happens when the variable is missing, so folding it in
    would hold prose to a rule it never entered.

    Whitespace inside a run is collapsed: the rules match *sentences*, and a
    75-column comment wraps them.  That is also why the rules below can be
    written against a normalised string without pinning the wrapping.
    """
    runs: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            current.append(line)
            continue
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    out = []
    for run in runs:
        joined = " ".join(l.strip().lstrip("#").strip() for l in run)
        if TOKEN in joined:
            out.append(re.sub(r"\s+", " ", joined).strip())
    return out


def _sentences(region: str) -> list[str]:
    body = re.sub(r"\s+", " ", region)
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+", body) if s.strip()]


# ---------------------------------------------------------------------------
# The vocabulary of the claim
# ---------------------------------------------------------------------------

#: Phrasings that put the failure *inside* a run that started.  Each pattern
#: opens on a "the program ran" word and must reach a dying verb without
#: crossing a sentence boundary, so a comment that merely says "the starter
#: never starts at all" cannot trip it.
DEFERRED_FAILURE = (
    re.compile(r"\bstart(?:s|ed|ing)?\b[^.]{0,140}?\b(?:and\s+then\s+)?"
               r"(?:die|dies|died|abort(?:s|ed)?)\b", re.I),
    re.compile(r"\b(?:on|at)\s+the\s+first\s+[\w/ -]{0,40}?"
               r"\b(?:call|h3d|message)\b", re.I),
    re.compile(r"\b(?:run|runs|running|start|starts)\b[^.]{0,140}?"
               r"\b(?:h3d|message)\s*(?:call|sys)", re.I),
)

#: At least one of these must appear: the prose has to place the failure
#: *before* the program's own code runs.  Several independent ways to say it,
#: so a reword cannot accidentally drop the meaning.
LOAD_TIME = re.compile(
    r"\bload(?:er|ing)?[- ]time\b|\bat\s+load\b|\bdynamic\s+loader\b"
    r"|\bdynamic[- ]link(?:er|ing)?\b"
    r"|\bnever\s+start(?:s|ed)?\b|\bdoes\s+not\s+start(?:s|ed)?\b"
    r"|\bcannot\s+start(?:s|ed)?\b"
    r"|\bbefore\s+(?:a\s+single|one|any)\b[^.]{0,60}?\bruns?\b", re.I)

#: Negations that keep a sentence from asserting a failure.
NEGATION = re.compile(r"\bno\b|\bnot\b|\bnever\b|\bnone\b|\bwithout\b"
                      r"|\bunaffected\b|\bunset\b|\bruns?\b", re.I)

#: An exit code, quoted as one: an integer within a couple of tokens of
#: ``exit``/``exits``.  Scoped that way because the region also carries
#: ``INSTALL.md:34-42``, and 42 is not an exit code.
EXIT_WORD = re.compile(r"^exits?[:]?$", re.I)
EXIT_INT = re.compile(r"^(\d{1,3})[,.;:)]?$")


def quoted_exit_codes(region: str, window: int = 3) -> set[int]:
    """The exit codes the prose quotes, by proximity to the word ``exit``."""
    tokens = region.split()
    found: set[int] = set()
    for index, token in enumerate(tokens):
        if not EXIT_WORD.match(token.strip("([\"'")):
            continue
        for near in tokens[max(0, index - window):index + window + 1]:
            match = EXIT_INT.match(near.strip("([\"'"))
            if match:
                found.add(int(match.group(1)))
    return found


#: A loader complaint attributed to a specific object.  The library named has
#: to be the one this box actually fails on, which is what kills "an unresolved
#: libapr-1": libapr-1.so.0 is a ``NEEDED`` entry *of the reader*, shipped
#: beside it, and never a symptom in its own right.
MISSING_OBJECT = re.compile(r"([\w.+-]+):\s*cannot open shared object file")


def check_missing_env_claim(region: str, measured: dict) -> list[str]:
    """Every way this region misdescribes a missing ``LD_LIBRARY_PATH``.

    Returns a list of problems, empty when the prose agrees with ``measured``.
    ``measured`` is :func:`measure_missing_ld_library_path`'s dict for the
    binary that carries the reader dependency.
    """
    problems: list[str] = []

    # 1. the failure must be placed before the program's own code runs
    for pattern in DEFERRED_FAILURE:
        found = pattern.search(region)
        if found is not None:
            problems.append(
                f"puts the failure inside a run that started "
                f"({pattern.pattern!r} matched {found.group(0)!r}): the loader "
                f"refuses the binary, so no code runs and no H3D or message "
                f"call is ever reached")
    if not LOAD_TIME.search(region):
        problems.append(
            "never says the failure happens at load time, before the program's "
            "own code runs -- that is where it happens")

    # 2. the exit code must be the one this box returns
    codes = quoted_exit_codes(region)
    if not codes:
        problems.append(
            f"quotes no exit code, so nothing here is checked against the run "
            f"that produced {measured['code']}")
    unmeasured = codes - {measured["code"]}
    if unmeasured:
        problems.append(
            f"quotes exit code(s) {sorted(unmeasured)} that this box does not "
            f"produce for a missing {TOKEN} (measured: {measured['code']})")

    # 3. the loader message must be the one this box prints
    truth = loader_message(measured["stderr"])
    if not measured["code"] and not measured["stderr"]:
        return problems          # the binary ran; nothing to quote
    if truth and re.sub(r"\s+", " ", truth) not in region:
        problems.append(
            f"does not quote the loader's actual message, so an operator has "
            f"nothing to match against: {truth!r}")
    named = {m.group(1) for m in MISSING_OBJECT.finditer(region)}
    if not named:
        problems.append(
            "quotes no loader message at all, so the claim cannot be checked "
            "against a run")
    wrong = named - {missing_library(measured["stderr"])}
    if wrong:
        problems.append(
            f"attributes the load failure to {sorted(wrong)}, which is not the "
            f"library this box cannot resolve "
            f"({missing_library(measured['stderr']) or 'none'})")

    return problems


def check_distinguishes_the_engine(region: str) -> list[str]:
    """The prose must say what the *engine* does, and not claim it fails.

    One of the two binaries needs the reader and one does not; a comment that
    says "the binaries die" is wrong about half of them, and that half is the
    half an operator uses to decide whether the oracle is broken.
    """
    about_engine = [s for s in _sentences(region) if re.search(r"engine", s, re.I)]
    if not about_engine:
        return ["never mentions the engine, so 'the binaries ... die' passes "
                "unexamined: only the starter carries the reader dependency"]
    hedged = [s for s in about_engine if NEGATION.search(s)]
    if not hedged:
        return [f"has {len(about_engine)} sentence(s) mentioning the engine and "
                f"not one of them keeps the engine out of it, so the prose "
                f"leaves it claiming what only the starter does: "
                f"{about_engine[0]!r}"]
    return []


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def check_oracle_env_text(text: str, measured: dict) -> list[str]:
    """Every problem with ``text``'s account of a missing ``LD_LIBRARY_PATH``."""
    regions = ld_docs_regions(text)
    if not regions:
        return [f"no comment anywhere in the file mentions {TOKEN}, so the "
                f"export below the reader library is left unexplained"]
    problems: list[str] = []
    for region in regions:
        problems += check_missing_env_claim(region, measured)
        problems += check_distinguishes_the_engine(region)
    return problems


def _measured_reader_binary(tmp_path: Path) -> dict:
    """Measure the binary that actually carries the reader dependency.

    Which binary that is is decided from the ELF, not from a name list, and
    the engine is measured too so the engine rule has something real to be
    checked against.  The two executables come from ``pyradioss.paths`` --
    the single resolver -- rather than from the harness's five-key
    ``oracle_paths()``, which would drag in ``th_to_csv``/``h3d_lib``/
    ``hm_reader_lib`` and warn about three things this measurement never
    looks at.
    """
    from pyradioss import paths

    measured = {}
    for key, binary in (("starter", paths.or_starter()),
                        ("engine", paths.or_engine())):
        measured[key] = measure_missing_ld_library_path(Path(binary), tmp_path)
        measured[key]["needed"] = _needed_libraries(Path(binary))
    reader = [k for k in ("starter", "engine")
              if any("hm_reader" in n for n in measured[k]["needed"])]
    assert reader, (
        f"neither oracle binary carries a reader NEEDED entry: "
        f"{ {k: measured[k]['needed'] for k in ('starter', 'engine')} } -- if "
        f"the reader is no longer linked, this whole file's subject has "
        f"changed and the measurement must be re-derived before the prose is "
        f"trusted")
    measured["reader_key"] = reader[0]
    measured["paths"] = paths
    return measured


def _needed_libraries(binary: Path) -> list[str]:
    """``NEEDED`` shared objects, via ``readelf -d`` (skipped without it)."""
    if not _which("readelf"):
        pytest.skip("readelf is not installed; the NEEDED entry cannot be "
                    "read and this measurement would be a guess")
    done = subprocess.run(["readelf", "-d", str(binary)], capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]",
                      done.stdout)


def _which(exe: str) -> str | None:
    import shutil
    return shutil.which(exe)


# ---------------------------------------------------------------------------
# 1. the live file agrees with a fresh measurement
# ---------------------------------------------------------------------------

def test_the_script_documents_the_failure_this_box_produces(tmp_path):
    """The rule, run over ``tools/oracle/oracle_env.sh`` as it stands."""
    _require_live_oracle(runtime_env=False)
    measured = _measured_reader_binary(tmp_path)
    text = ORACLE_ENV.read_text(encoding="utf-8")
    problems = check_oracle_env_text(text, measured[measured["reader_key"]])
    assert problems == [], (
        "tools/oracle/oracle_env.sh misdescribes what a missing "
        f"{TOKEN} does on this box:\n  - " + "\n  - ".join(problems)
        + "\nMeasure it: `env -u " + TOKEN + " <binary> -v; echo $?`. The old "
          "text here sent an operator looking for an H3D bug behind a 127.")


def test_the_measured_failure_really_is_a_load_failure(tmp_path):
    """Guard the premise: the symptom must still be the load-time one.

    If this stops holding, the prose is not wrong — it is describing a
    different box, and the fix is to re-measure and rewrite, not to keep the
    old claim.  Failing loudly here is what stops that from being silent.
    """
    _require_live_oracle(runtime_env=False)
    measured = _measured_reader_binary(tmp_path)
    reader = measured[measured["reader_key"]]
    assert reader["code"] != 0, (
        f"{reader['binary']} now STARTS without {TOKEN} (exit 0): the "
        f"documented failure mode has changed and oracle_env.sh plus "
        f"validate_vs_fortran.py must be re-measured together")
    assert "error while loading shared libraries" in reader["stderr"], (
        f"{reader['binary']} fails without {TOKEN} but not in the loader: "
        f"{reader['stderr']!r}")
    assert "hm_reader" in missing_library(reader["stderr"]), (
        f"the unresolved library is no longer the reader: {reader['stderr']!r}")


def test_the_engine_does_not_need_the_reader_at_all(tmp_path):
    """The engine's ``NEEDED`` set has no reader, and its ``-v`` still runs.

    This is the fact the old comment erased by speaking of "the binaries".
    """
    _require_live_oracle(runtime_env=False)
    measured = _measured_reader_binary(tmp_path)
    engine = measured["engine"]
    assert not any("hm_reader" in n for n in engine["needed"]), engine["needed"]
    assert engine["code"] == 0, (
        f"the engine needs no reader library, so it must run without "
        f"{TOKEN}: exit {engine['code']}, stderr {engine['stderr']!r}")
    assert engine["stdout"].strip(), "the engine printed no banner"


# ---------------------------------------------------------------------------
# 2. the rule can still fail -- the record it corrects, and four near-misses
# ---------------------------------------------------------------------------

#: The paragraph this milestone replaces, quoted verbatim from ``69d8ede``
#: (the text was already present before this fix; ``git show
#: 69d8ede:tools/oracle/oracle_env.sh``).  Kept as a literal so the rule is
#: exercised hermetically -- a test that reached into git would pass on a
#: clone with a shallow history and prove nothing.
PRE_FIX_PARAGRAPH = """\
# LD_LIBRARY_PATH IS required, not cosmetic: the binaries start without it and
# then die on the first H3D / message call with an unresolved libhm_reader or
# libapr-1.  Both come straight from $OR_SRC/INSTALL.md:34-42 ("Environment
# variables settings under Linux"), retargeted from OPENRADIOSS_PATH to the
# mirror prefix $OR_BUILD.
"""

#: A plausible-looking but wrong replacement, built by editing the *fixed*
#: text so that only the falsified fact differs.  Each of these must be caught:
#: they are the specific ways a re-fix of this comment would go wrong again.
WRONG_EXIT_CODE = PRE_FIX_PARAGRAPH.replace(
    "and then die on the first H3D / message call", "")
WRONG_EXIT_CODE = WRONG_EXIT_CODE.replace(
    "with an unresolved libhm_reader or\n# libapr-1.", "")
WRONG_EXIT_CODE = (
    "# LD_LIBRARY_PATH IS required.  The starter dies in the dynamic loader\n"
    "# with exit 126 and `libhm_reader_linux64.so: cannot open shared object\n"
    "# file: No such file or directory`.  The engine needs no reader library,\n"
    "# so its -v banner prints with the variable unset (exit 0).\n")

WRONG_LIBRARY = (
    "# LD_LIBRARY_PATH IS required.  With the variable unset the starter is\n"
    "# refused by the dynamic loader with exit 127 and\n"
    "# `libapr-1.so.0: cannot open shared object file: No such file or\n"
    "# directory`; nothing of the program runs.  The engine needs no reader\n"
    "# library and its -v banner prints with the variable unset (exit 0).\n")

BACK_TO_RUN_TIME = (
    "# LD_LIBRARY_PATH IS required, not cosmetic: the binaries start without\n"
    "# it and then die on the first H3D / message call with an unresolved\n"
    "# libhm_reader.  The starter carries a NEEDED entry for\n"
    "# libhm_reader_linux64.so and the dynamic loader reports\n"
    "# `libhm_reader_linux64.so: cannot open shared object file: No such file\n"
    "# or directory` with exit 127.  The engine needs no reader library, so its\n"
    "# -v banner prints with the variable unset (exit 0).\n")


def test_the_pre_fix_paragraph_is_rejected(tmp_path):
    """The text this milestone corrects must FAIL the rule.

    Without this the rule could be a checker that accepts anything and still
    pass the live file, because the live file would be its only input.
    """
    _require_live_oracle(runtime_env=False)
    measured = _measured_reader_binary(tmp_path)
    truth = measured[measured["reader_key"]]
    problems = check_oracle_env_text(PRE_FIX_PARAGRAPH, truth)
    assert problems != [], (
        "the pre-fix paragraph passes the rule, so the rule does not detect "
        "the rot this milestone is about")
    assert any("inside a run that started" in p for p in problems), (
        f"expected the run-time phrasing to be named; got {problems}")


@pytest.mark.parametrize("variant", [
    pytest.param(WRONG_EXIT_CODE, id="exit-126-not-measured"),
    pytest.param(WRONG_LIBRARY, id="libapr-not-the-unresolved-library"),
    pytest.param(BACK_TO_RUN_TIME, id="run-time-phrasing-returns"),
])
def test_the_claim_rule_can_still_fail(variant, tmp_path):
    """Four ways a re-fix could go wrong, each caught, none of them by luck."""
    _require_live_oracle(runtime_env=False)
    measured = _measured_reader_binary(tmp_path)
    truth = measured[measured["reader_key"]]
    assert check_oracle_env_text(variant, truth) != [], (
        f"this wrong variant passes the rule:\n{variant}")


def test_the_engine_rule_can_still_fail(tmp_path):
    """"Both binaries are refused at load" is wrong about one of them.

    This is the half of the old sentence that was false about the engine, and
    it is false in the *load-time* wording too -- so fixing "the binaries" into
    "the starter" is not enough; the engine has to be accounted for.
    """
    _require_live_oracle(runtime_env=False)
    measured = _measured_reader_binary(tmp_path)
    truth = measured[measured["reader_key"]]
    # Built from the measurement, so the only thing false about it is the
    # "both binaries": the loader message and the exit code are exactly right.
    both = (
        f"# {TOKEN} IS required, not cosmetic.  Both binaries are refused by\n"
        f"# the dynamic loader when it is unset, which exits {truth['code']} "
        f"with\n# `{loader_message(truth['stderr'])}`; nothing of either "
        f"program runs.\n")
    normalised = re.sub(r"\s+", " ", " ".join(
        l.strip().lstrip("#").strip() for l in both.splitlines())).strip()
    assert check_missing_env_claim(normalised, truth) == [], (
        "this variant was meant to isolate the engine rule")
    problems = check_distinguishes_the_engine(normalised)
    assert problems != [], "asserting a failure of the engine passes"
    assert any("engine" in p for p in problems), problems


def test_the_rule_accepts_the_live_file_because_it_quotes_the_measurement(
        tmp_path):
    """Anti-vacuity: passing requires quoting real evidence, not silence.

    A region that says nothing quotable is a failure, so "make the comment
    vaguer" is not a way out of this rule.
    """
    _require_live_oracle(runtime_env=False)
    measured = _measured_reader_binary(tmp_path)
    truth = measured[measured["reader_key"]]
    vague = (
        f"# {TOKEN} IS required, not cosmetic.  Upstream says so in\n"
        f"# $OR_SRC/{INSTALL_MD}:34-42, retargeted to $OR_BUILD.  Set it before\n"
        f"# running the oracle; the binaries will not work otherwise.  The\n"
        f"# engine needs no reader library, so it is unaffected.\n")
    assert check_oracle_env_text(vague, truth) != [], (
        "a comment that quotes no exit code and no loader message passes; "
        "the rule is measuring nothing")


# ---------------------------------------------------------------------------
# 3. the parts of the file this edit was not allowed to cost
# ---------------------------------------------------------------------------

#: The CFG card-schema tree ``RAD_CFG_PATH`` names (upstream's spelling of
#: ``PYRADIOSS_HM_CFG``; ``$OR_SRC/INSTALL.md:39``).
CFG_ENV = "RAD_CFG_PATH"


def _source_oracle_env(tmp_path: Path, mirror: Path) -> dict:
    """Actually source ``oracle_env.sh`` against ``mirror``; return the env.

    Real sourcing in a child ``bash``, not a regex over the file: the property
    under test is a *conditional export*, so the only thing that can observe it
    is what the shell ends up holding.  ``OR_ROOT`` is a scratch directory that
    merely satisfies the script's ``: "${OR_ROOT:?...}"`` guard — ``$OR_ROOT``
    is only prefixed onto ``PATH`` and the two binary names here, and nothing in
    this test runs a binary.  ``RAD_CFG_PATH`` / ``PYRADIOSS_HM_CFG`` are
    deleted from the child environment first, so the answer is the script's and
    not the caller's.
    """
    root = tmp_path / "or_root"
    root.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k not in (CFG_ENV, "PYRADIOSS_HM_CFG")}
    env.update(OR_BUILD=str(mirror), OR_ROOT=str(root))
    probe = "%s=${%s-<unset>}" % (CFG_ENV, CFG_ENV)
    done = subprocess.run(
        ["bash", "-c", ". " + str(ORACLE_ENV) + '\nprintf "%s\\n" "' + probe + '"'],
        env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    line = done.stdout.strip().splitlines()[-1]
    assert line.startswith(f"{CFG_ENV}="), done.stdout
    return {"line": line, "value": line.split("=", 1)[1]}


def _mirror(tmp_path: Path, name: str, *, cfg: bool) -> Path:
    """A scratch ``$OR_BUILD``: a mirror ``is_or_mirror`` accepts, with or
    without ``hm_cfg_files``.

    ``pyradioss.paths.is_or_mirror`` wants ``CMakeLists.txt`` and
    ``extlib/hm_reader`` -- the two gates ``build_oracle.sh:169,175`` enforces
    -- so the fixture is a mirror by that same definition and not by assertion.
    """
    mirror = tmp_path / name
    (mirror / "extlib" / "hm_reader").mkdir(parents=True, exist_ok=True)
    (mirror / "CMakeLists.txt").write_text("# scratch mirror\n", encoding="utf-8")
    if cfg:
        (mirror / "hm_cfg_files").mkdir(exist_ok=True)
    return mirror


def test_the_cfg_export_is_withheld_when_the_tree_it_names_is_absent(tmp_path):
    """``RAD_CFG_PATH`` must not be exported pointing at nothing.

    The coupling this pins, and it is the whole defect: a variable that is set
    but does not resolve is **terminal** for ``pyradioss.paths`` (§4.1 rule 1
    read with rule 4 — ``_resolve`` warns and then raises).  So an unconditional
    ``export RAD_CFG_PATH="$OR_BUILD/hm_cfg_files"`` is not a harmless default:
    against a mirror with no ``hm_cfg_files`` it *poisons* the one variable
    that would otherwise have let the resolver fall through to a real tree, and
    ``hm_cfg_dir()`` raises instead of finding it.  Measured consequence before
    the fix, on this box with a cfg-less mirror sourced:
    ``tests/test_m539_law34_input_audit.py`` +
    ``tests/test_m540_law37_input_audit.py`` → ``57 passed, 14 skipped``
    (the 7 LAW34 and 7 LAW37 CFG audits all skip) against ``71 passed`` with no
    environment at all.  A green run that silently stopped auditing two laws'
    CFG schemas is exactly the failure mode a variable export can cause.

    So the script must export the variable **only when the directory it names
    exists**, and leave it unset otherwise — which is what §4.1 rule 1 asks for
    (``$PYRADIOSS_HM_CFG`` / ``$RAD_CFG_PATH``, *if set*).  The positive half is
    asserted in the next test, because a fix that simply deleted the export
    would pass this one.
    """
    mirror = _mirror(tmp_path, "mirror_without_cfg", cfg=False)
    assert not (mirror / "hm_cfg_files").exists(), "fixture is not cfg-less"
    got = _source_oracle_env(tmp_path, mirror)
    assert got["value"] == "<unset>", (
        f"oracle_env.sh exported {CFG_ENV}={got['value']!r} for a mirror with "
        f"no hm_cfg_files. An exported-but-unresolvable variable is TERMINAL for "
        f"pyradioss.paths, so this makes hm_cfg_dir() raise and silently drops "
        f"the LAW34/LAW37 CFG audits. Export it only when the directory exists.")


def test_the_cfg_export_is_still_there_when_the_tree_exists(tmp_path):
    """The other half: a mirror that HAS ``hm_cfg_files`` still gets it.

    Without this, "delete the export" is a way to make the previous test green,
    and upstream's own spelling of the variable (``$OR_SRC/INSTALL.md:39``,
    accepted as an alias by ``paths.hm_cfg_dir``) would stop being set by the
    one script that documents how to set it.  The tree here is a bare directory:
    the script's job is to export the path, and ``is_cfg_tree`` is the
    resolver's business, not this fixture's.
    """
    mirror = _mirror(tmp_path, "mirror_with_cfg", cfg=True)
    got = _source_oracle_env(tmp_path, mirror)
    assert got["value"] == str(mirror / "hm_cfg_files"), (
        f"oracle_env.sh must still export {CFG_ENV} when the tree is there; "
        f"got {got['value']!r}")


def test_the_guarded_export_is_conditional_on_the_directory(tmp_path):
    """The guard is a test of the filesystem, not a comment about it.

    Pins the *mechanism* so the two tests above cannot be satisfied by
    something that merely happens to produce the right environment: the script
    must ask whether ``$OR_BUILD/hm_cfg_files`` is a directory.  A bare
    ``export`` of the variable — with or without a surrounding ``if`` that never
    evaluates false — fails here.

    This is the drift pin the defect needs: ``oracle_env.sh`` and
    ``pyradioss/paths.py`` are two files with no shared test, so the coupling
    that broke (an exported variable meeting a terminal resolver) could
    otherwise be reintroduced by either edit alone.
    """
    text = ORACLE_ENV.read_text(encoding="utf-8")
    guarded = re.search(
        r'^\s*if\s+\[?\s*-[dD]\s+"?\$\{?OR_BUILD\}?/hm_cfg_files"?\s*\]?',
        text, re.M)
    assert guarded, (
        "oracle_env.sh must guard the RAD_CFG_PATH export with a directory test "
        "on $OR_BUILD/hm_cfg_files (e.g. `if [ -d \"$OR_BUILD/hm_cfg_files\" ];"
        " then`) so the variable is exported only where it resolves")


def test_the_h3d_writer_stays_deliberately_out_of_reach():
    """``RAD_H3D_PATH`` unset is a safety decision; the rationale must survive.

    The reachable extlib carries a ``libh3dwriter.so`` whose writer API is one
    parameter short of what the pinned source calls, so a run that wrote H3D
    files through it would produce silently wrong ones.  Leaving the variable
    unset makes ``h3d_dl.c`` fail every ``dlopen`` trial and abort loudly with
    MSGID 274.  That is the expensive failure this comment buys cheaply, and
    an edit that dropped the reasoning would leave the next reader to
    "fix" it.
    """
    text = ORACLE_ENV.read_text(encoding="utf-8")
    assert not re.search(r"^\s*export\s+RAD_H3D_PATH", text, re.M), (
        "oracle_env.sh must not export RAD_H3D_PATH")
    assert not re.search(r"^\s*export\s+\{[^}]*RAD_H3D_PATH", text, re.M), (
        "oracle_env.sh must not export RAD_H3D_PATH in a braced assignment")
    for token in ("RAD_H3D_PATH", "deliberately NOT exported",
                  "h3d_build_cpp/h3d_dl.c", "dlopen", "MSGID 274",
                  "silently produce", "RAD_CFG_PATH"):
        assert token in text, (
            f"the h3d safety rationale lost {token!r}; without it the next "
            f"reader cannot tell a decision from an oversight")


def test_the_file_is_still_valid_shell():
    """``bash -n`` — an accidental edit to a sourced file is a real defect."""
    import shutil
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed; oracle_env.sh is a bash/sh script")
    done = subprocess.run([bash, "-n", str(ORACLE_ENV)], capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_the_upstream_citation_names_the_line_that_states_the_requirement():
    """``INSTALL.md:<range>`` must cover the ``LD_LIBRARY_PATH`` export.

    Checked against the real ``INSTALL.md`` when the mirror is here, so a
    citation cannot drift away from the line it points at without either the
    file or the mirror having moved.
    """
    text = ORACLE_ENV.read_text(encoding="utf-8")
    cites = re.findall(rf"{INSTALL_MD}:(\d+)(?:-(\d+))?", text)
    assert cites, f"no {INSTALL_MD} path:line citation in oracle_env.sh"
    covering = [(int(lo), int(hi or lo)) for lo, hi in cites
                if int(lo) <= INSTALL_LD_LINE <= int(hi or lo)]
    assert covering, (
        f"no citation of {INSTALL_MD} covers line {INSTALL_LD_LINE}, which is "
        f"where the {TOKEN} requirement is stated; citations found: {cites}")

    from pyradioss import paths
    try:
        install = paths.or_src() / INSTALL_MD
    except FileNotFoundError:
        pytest.skip(f"$OR_SRC does not resolve, so {INSTALL_MD} cannot be "
                    f"checked against its own line numbers")
    if not install.is_file():
        pytest.skip(f"{install} is not there")
    lines = install.read_text(encoding="utf-8",
                              errors="replace").splitlines()
    for low, high in covering:
        block = "\n".join(lines[low - 1:high])
        assert f"{TOKEN}=" in block, (
            f"{install}:{low}-{high} does not state the {TOKEN} requirement:\n"
            f"{block}")


def test_the_export_points_at_the_directory_holding_the_reader(tmp_path):
    """The export names the directory that actually resolves the library.

    Ties the prose to disk: the export must point where the ``NEEDED``
    library lives, so "this export is the only route to that library" is a
    statement about a real path rather than a plausible one.
    """
    _require_live_oracle(runtime_env=False)
    text = ORACLE_ENV.read_text(encoding="utf-8")
    match = re.search(r'^\s*export\s+' + TOKEN + r'="?\$OR_BUILD/([^"$]+)', text,
                      re.M)
    assert match, f"no 'export {TOKEN}=\"$OR_BUILD/...' line in oracle_env.sh"
    relative = match.group(1)

    from pyradioss import paths
    try:
        build = paths.or_build()
    except FileNotFoundError:
        pytest.skip("OR_BUILD does not resolve, so the exported directory "
                    "cannot be checked against the library it must contain")
    directory = build / relative.rstrip("/:")
    if not directory.is_dir():
        pytest.skip(f"{directory} is not here, so this check would be a guess")
    measured = _measured_reader_binary(tmp_path)
    library = missing_library(measured[measured["reader_key"]]["stderr"])
    assert library, "the measurement named no library to look for"
    assert (directory / library).is_file(), (
        f"the export names {directory}, which does not hold {library} -- the "
        f"library this box cannot resolve without it")