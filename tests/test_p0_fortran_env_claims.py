"""``tools/validate_vs_fortran.py`` must not describe a missing
``LD_LIBRARY_PATH`` as a run-time failure.

The ``LD_LIBRARY_PATH`` bullet of :func:`fortran_env`'s docstring said:

    Not cosmetic: without it the binaries die on the first message call with
    an unresolved ``libhm_reader`` (``tools/oracle/oracle_env.sh``).

Both halves of that are wrong, and ``e1072ca`` already corrected the file this
one *credits*:

* **the symptom.**  With the variable unset the dynamic loader refuses the
  starter before one instruction of its own code runs, so no message call is
  ever reached: the process exits 127 with ``error while loading shared
  libraries: libhm_reader_linux64.so: cannot open shared object file: No such
  file or directory`` (measured at test time, not quoted from here — see
  :func:`measure_missing_ld_library_path`).  ``engine_linux64_gf`` needs no
  reader library at all: its ``-v`` banner prints and it exits 0.
* **the attribution.**  ``tools/oracle/oracle_env.sh`` says exactly the
  opposite of what it is cited for, so the citation pointed an operator at a
  file that refutes the claim.

So this file generalises the rule ``e1072ca`` established for the shell script
(the script's own suite is ``tests/test_p0_oracle_env_script.py``, and its
measurement helpers are imported here rather than duplicated — one measurement,
one vocabulary, so the two suites cannot drift apart):

    **Where a Python module documents the consequence of an environment
    variable being missing, the documented failure mode must be the one this
    box produces, and it must be quoted evidence, not inference.**

Three properties make it checkable rather than a vibe:

* **measured, never typed in.**  The exit code, the loader message and the
  unresolved library all come out of a live run of the binary that carries the
  reader ``NEEDED`` entry (:func:`_measured`), so a box whose loader words the
  message differently updates the rule by changing reality.
* **positive and negative.**  A claim must place the failure at load time
  (:data:`LOAD_TIME`), quote the measured exit code
  (:func:`quoted_exit_codes`) and quote the loader's own message — *and* must
  not place the failure inside a run that started (:data:`DEFERRED_FAILURE`).
  A vaguer comment therefore fails too: there is no wording that gets through
  by saying less.
* **scoped to sentences.**  :func:`_claim_paragraphs` selects by what a
  sentence actually asserts (an absence *and* a symptom *and* a subject), not
  by proximity, so the module's unrelated prose is not held to a rule it never
  entered — while :func:`test_no_surviving_instance_anywhere_in_the_file`
  covers the whole file regardless, which is how a second copy of the claim in
  a different docstring is caught.

The rule can still fail: :func:`test_the_pre_fix_bullet_is_rejected` runs it
over the **verbatim pre-fix text** (quoted from ``git show
14da7b0:tools/validate_vs_fortran.py``, kept as a literal so the check works on
a clone with shallow history), and
:func:`test_the_claim_rule_can_still_fail` over four synthetic near-misses —
notably :data:`REWORDED_BUT_STILL_WRONG`, which is the case that matters here:
the run-time claim rephrased so that none of the pre-fix wording survives,
while the exit code and the loader message are quoted exactly right.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_p0_oracle_build import _require_live_oracle
from tests.test_p0_oracle_env_script import (
    DEFERRED_FAILURE,
    LOAD_TIME,
    MISSING_OBJECT,
    _needed_libraries,
    ld_docs_regions,
    loader_message,
    measure_missing_ld_library_path,
    missing_library,
    quoted_exit_codes,
)

REPO = Path(__file__).resolve().parents[1]
VALIDATE = REPO / "tools" / "validate_vs_fortran.py"
ORACLE_ENV = REPO / "tools" / "oracle" / "oracle_env.sh"

#: The variable whose absence is at issue.
TOKEN = "LD_LIBRARY_PATH"

#: The subject of the sentence: the variable, the library it routes to, or the
#: loader that enforces it.  Deliberately **not** ``the binaries`` / ``the
#: solver``: those words carry half of this module's correct prose about
#: statuses, converters and ``DT_RPATH``, and holding a status comment to a
#: rule about the loader is exactly the false positive this module cannot
#: afford.  The run-time phrasing itself is caught file-wide by
#: :func:`test_no_surviving_instance_anywhere_in_the_file` whatever it names.
SUBJECT = re.compile(r"LD_LIBRARY_PATH|hm_reader|libapr|\bloader\b|\bld\.so\b",
                     re.I)

#: The absence being talked about.  ``no``/``never`` are deliberately absent:
#: they appear all over this module's correct prose ("no RPATH", "never
#: assumed") and would select paragraphs that make no claim at all.
ABSENCE = re.compile(r"\bwithout\b|\bunset\b|\babsent\b|\bmissing\b"
                     r"|\bnot\s+set\b|\bnever\s+set\b|\bexcluded\b", re.I)

#: Something going wrong.  ``unresolved`` is in because the pre-fix sentence
#: used it; ``cannot open`` because it is the loader's own wording.
SYMPTOM = re.compile(r"\b(?:die|dies|died|abort(?:s|ed)?|crash(?:es|ed)?"
                     r"|fail(?:s|ed|ing|ure|ures)?|unresolved|cannot\s+open"
                     r"|refus(?:e|es|ed|al)|reject(?:s|ed)|segfault)\b", re.I)

#: A sentence that **denies** a message/H3D call is not a deferred-failure
#: claim — "no message call is ever reached" is the opposite of "dies on the
#: first message call".  Without this guard the vocabulary above would flag the
#: correction for containing the words it forbids.  ``not`` is excluded on
#: purpose: in "Not cosmetic: without it the binaries die on the first message
#: call" the negation belongs to "cosmetic", and letting it cancel the claim
#: would launder the exact sentence this task removes.
DENIES_MESSAGE_CALL = re.compile(
    r"\b(?:no|never|nothing|neither)\b(?:\s+\w+){0,3}?\s+"
    r"(?:message|h3d)\s*(?:call|sys)", re.I)


def deferred_claims(text: str) -> list[str]:
    """Every sentence that puts the failure inside a run that started.

    Sentence-scoped and negation-aware, which is what makes the finding
    specific: a *reworded* version of the run-time claim is still caught (it
    still places the failure after the run began) while a sentence that says
    the opposite is not caught for saying so.
    """
    out: list[str] = []
    for paragraph in _paragraphs(text):
        for sentence in _sentences(paragraph):
            if DENIES_MESSAGE_CALL.search(sentence):
                continue
            for pattern in DEFERRED_FAILURE:
                found = pattern.search(sentence)
                if found is not None:
                    out.append(f"{found.group(0)!r} in {sentence!r}")
    return out


# ---------------------------------------------------------------------------
# Measurement -- the ground truth the prose is checked against
# ---------------------------------------------------------------------------

def _measured(tmp_path: Path) -> tuple[dict, str]:
    """Run both binaries with ``LD_LIBRARY_PATH`` scrubbed; return the runs
    and the key of the one that carries the reader ``NEEDED`` entry.

    The binaries come from ``pyradioss.paths`` (the single resolver) and the
    reader-carrying one is decided from the ELF, not from a name list — so a
    rebuild that dropped the dependency fails the premise loudly instead of
    leaving the prose describing a different box.
    """
    from pyradioss import paths

    runs = {}
    for key, binary in (("starter", paths.or_starter()),
                        ("engine", paths.or_engine())):
        run = measure_missing_ld_library_path(Path(binary), tmp_path)
        run["needed"] = _needed_libraries(Path(binary))
        runs[key] = run
    reader = [k for k in ("starter", "engine")
              if any("hm_reader" in n for n in runs[k]["needed"])]
    assert reader, (
        "neither oracle binary carries a reader NEEDED entry: "
        f"{ {k: runs[k]['needed'] for k in ('starter', 'engine')} } -- the "
        f"failure this file's prose must describe has changed, so re-measure "
        f"and rewrite instead of trusting the old wording")
    return runs, reader[0]


# ---------------------------------------------------------------------------
# Locating the claims in a Python module (docstrings, not ``#`` comments)
# ---------------------------------------------------------------------------

def _paragraphs(text: str) -> list[str]:
    """Maximal runs of consecutive non-blank lines, whitespace collapsed.

    A paragraph is the unit the evidence rules apply to, because a 75-column
    wrap splits a quoted loader message across lines; the sentence rules inside
    it are what locate the claim.
    """
    out: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return [re.sub(r"\s+", " ", " ".join(l.strip() for l in run)).strip()
            for run in out]


def _sentences(paragraph: str) -> list[str]:
    """Sentences of ``paragraph``, split on ``.``/``;`` only.

    Never on ``:``: the loader's message is ``error while loading shared
    libraries: <soname>: cannot open shared object file: ...`` and splitting
    there would hide the very evidence the rules look for.
    """
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", paragraph) if s.strip()]


def _claim_sentences(paragraph: str) -> list[str]:
    """Sentences that assert a symptom *of the variable being absent*.

    All three parts must be present, which is what keeps this from dragging in
    the module's correct prose: a sentence about ``DT_RPATH`` being absent says
    nothing about what a run does, and a sentence about a failing h3d call
    says nothing about the reader.
    """
    return [s for s in _sentences(paragraph)
            if SUBJECT.search(s) and ABSENCE.search(s) and SYMPTOM.search(s)]


def _claim_paragraphs(text: str) -> list[str]:
    return [p for p in _paragraphs(text) if _claim_sentences(p)]


def check_missing_env_prose(text: str, measured: dict) -> list[str]:
    """Every way ``text`` misdescribes a missing ``LD_LIBRARY_PATH``.

    ``measured`` is one run from :func:`_measured` -- the binary carrying the
    reader dependency.  Two scopes, and the difference matters:

    * **sentence scope** -- every sentence that asserts a symptom of the
      variable's absence must place it at load time and quote the measured
      evidence (exit code, loader message, unresolved library).  A sentence
      about the *engine* (which is not a claim sentence, because it asserts no
      symptom) is therefore free to quote ``exit 0``: that is a different
      binary's different outcome, and a rule that forbade it would force the
      prose to omit the one fact that explains the asymmetry.
    * **file scope** -- no sentence anywhere may place the failure inside a run
      that started (:func:`deferred_claims`), so a paragraph cannot be right
      about the loader in one sentence and wrong about it in the next, and a
      copy of the claim in a neighbouring docstring is caught too.
    """
    problems: list[str] = []
    claims = _claim_paragraphs(text)
    if not claims:
        return [f"no sentence in this file says what a missing {TOKEN} does, "
                f"so the export it builds is left undocumented"]

    # (1) the failure is load-time -- everywhere it is mentioned, not once.
    for found in deferred_claims(text):
        problems.append(
            f"puts the failure inside a run that started ({found}): the loader "
            f"refuses the binary, so no code runs and no message call is "
            f"ever reached")

    # (2) and every claim quotes what the run actually produced.
    truth = re.sub(r"\s+", " ", loader_message(measured["stderr"]))
    unresolved = missing_library(measured["stderr"])
    for paragraph in claims:
        for sentence in _claim_sentences(paragraph):
            if not LOAD_TIME.search(sentence):
                problems.append(
                    f"a sentence asserts a symptom without ever placing it at "
                    f"load time: {sentence!r}")
            # A code quoted *in the claim sentence* must be the measured one.
            # Elsewhere in the paragraph it may be another binary's outcome --
            # the engine's ``exit 0`` is a fact, not a defect.
            unmeasured = quoted_exit_codes(sentence) - {measured["code"]}
            if unmeasured:
                problems.append(
                    f"quotes exit code(s) {sorted(unmeasured)} that this box "
                    f"does not produce for a missing {TOKEN} "
                    f"(measured: {measured['code']}): {sentence!r}")
        # The quoted evidence may sit in the sentence after the claim (a
        # wrapped 75-column comment splits it that way all the time), so
        # presence is required of the paragraph, not of the claim sentence.
        if measured["code"] not in quoted_exit_codes(paragraph):
            problems.append(
                f"quotes no exit code, so nothing here is checked against the "
                f"run that produced {measured['code']}")
        if truth and truth not in paragraph:
            problems.append(
                f"does not quote the loader's actual message, so an operator "
                f"has nothing to match against: {truth!r}")
        named = {m.group(1) for m in MISSING_OBJECT.finditer(paragraph)}
        if not named:
            problems.append(
                f"quotes no loader message at all, so the claim cannot be "
                f"checked against a run")
        wrong = named - {unresolved}
        if wrong:
            problems.append(
                f"attributes the load failure to {sorted(wrong)}, which is not "
                f"the library this box cannot resolve "
                f"({unresolved or 'none'})")
    return problems


# ---------------------------------------------------------------------------
# 1. the live file agrees with a fresh measurement
# ---------------------------------------------------------------------------

def test_the_harness_documents_the_failure_this_box_produces(tmp_path):
    """The rule, run over ``tools/validate_vs_fortran.py`` as it stands."""
    _require_live_oracle(runtime_env=False)
    runs, reader = _measured(tmp_path)
    problems = check_missing_env_prose(VALIDATE.read_text(encoding="utf-8"),
                                       runs[reader])
    assert problems == [], (
        f"tools/validate_vs_fortran.py misdescribes what a missing {TOKEN} "
        f"does on this box:\n  - " + "\n  - ".join(problems)
        + f"\nMeasure it: `env -u {TOKEN} <binary> -v; echo $?`. The old "
          f"text here sent an operator looking for an h3d/message bug behind "
          f"a 127 that the dynamic loader produced.")


def test_no_surviving_instance_anywhere_in_the_file(tmp_path):
    """The claim is searched for across the **whole** file, not one docstring.

    ``e1072ca`` corrected the same sentence in ``oracle_env.sh`` and the
    bullet in ``validate_vs_fortran.py`` was missed; a rule scoped to the
    known paragraph would let a third copy land in a neighbouring docstring
    unnoticed.
    """
    _require_live_oracle(runtime_env=False)
    runs, reader = _measured(tmp_path)
    text = VALIDATE.read_text(encoding="utf-8")
    offenders = deferred_claims(text)
    assert offenders == [], (
        f"the run-time phrasing survives elsewhere in "
        f"tools/validate_vs_fortran.py: {offenders}; grep the whole file for "
        f"'die'/'first message'/'unresolved' and fix every instance, not only "
        f"the one a previous review named")


def test_the_premise_is_still_a_load_failure(tmp_path):
    """Guard the premise: the symptom must still be the load-time one.

    If this stops holding the prose is not wrong, it describes another box,
    and the fix is to re-measure and rewrite — not to keep the old claim.
    """
    _require_live_oracle(runtime_env=False)
    runs, reader = _measured(tmp_path)
    fail = runs[reader]
    assert fail["code"] != 0, (
        f"{fail['binary']} now STARTS without {TOKEN} (exit 0): the failure "
        f"mode changed and this file's prose must be re-derived")
    assert "error while loading shared libraries" in fail["stderr"], (
        f"{fail['binary']} fails without {TOKEN} but not in the loader: "
        f"{fail['stderr']!r}")
    assert "hm_reader" in missing_library(fail["stderr"]), fail["stderr"]


# ---------------------------------------------------------------------------
# 2. the rule can still fail -- the record it corrects, and four near-misses
# ---------------------------------------------------------------------------

#: The bullet as it stood at ``14da7b0`` (quoted verbatim:
#: ``git show 14da7b0:tools/validate_vs_fortran.py``, lines 421-424).  Kept as
#: a literal so the rule is exercised without git history.
PRE_FIX_BULLET = """\
    * ``LD_LIBRARY_PATH`` (POSIX) or ``PATH`` (Windows) — the native ``.k``
      reader and its APR dependency.  Not cosmetic: without it the binaries
      die on the first message call with an unresolved ``libhm_reader``
      (``tools/oracle/oracle_env.sh``).
"""

#: The claim reworded so that none of the pre-fix wording survives, while the
#: exit code and the loader message are quoted **exactly right**.  This is the
#: variant a well-meaning re-fix produces, and it is why the rule needs the
#: per-sentence load-time requirement rather than a ban on one phrase.
REWORDED_BUT_STILL_WRONG = """\
    * ``LD_LIBRARY_PATH`` (POSIX) or ``PATH`` (Windows) — the native ``.k``
      reader.  Not cosmetic: without it ``starter_linux64_gf`` launches and
      then aborts inside its first message call with an unresolved
      ``libhm_reader``.  It exits 127 and the dynamic loader's own complaint,
      ``error while loading shared libraries: libhm_reader_linux64.so: cannot
      open shared object file: No such file or directory``, is what you see.
      ``tools/oracle/oracle_env.sh`` documents the requirement.
"""

#: Load-time wording, right library, plausible-but-unmeasured exit code.
UNMEASURED_EXIT_CODE = """\
    * ``LD_LIBRARY_PATH`` — the native ``.k`` reader.  Not cosmetic: with it
      unset the dynamic loader refuses ``starter_linux64_gf`` before any of
      its own code runs and exits 126 with
      ``error while loading shared libraries: libhm_reader_linux64.so: cannot
      open shared object file: No such file or directory``.
"""

#: Load-time wording, right exit code, a library this box can resolve.  Only
#: the attribution is wrong, so that is the only fault reported.
WRONG_LIBRARY = """\
    * ``LD_LIBRARY_PATH`` — the native ``.k`` reader.  Not cosmetic: with it
      unset the dynamic loader refuses ``starter_linux64_gf`` before any of
      its own code runs and exits 127 with
      ``libapr-1.so.0: cannot open shared object file: No such file or
      directory``.
"""

#: Truthful, load-time, engine-aware — and quoting nothing, so nothing in it
#: can be checked against a run.  Closing this is how "just delete the
#: paragraph" would pass.
VAGUE_BUT_TRUE = """\
    * ``LD_LIBRARY_PATH`` — the native ``.k`` reader.  Not cosmetic: with it
      unset the dynamic loader refuses ``starter_linux64_gf`` before any of
      its own code runs.  ``engine_linux64_gf`` needs no reader library and is
      unaffected.
"""


def test_the_pre_fix_bullet_is_rejected(tmp_path):
    """The text this task corrects must FAIL the rule.

    Without this the rule could be a checker that accepts anything and still
    pass the live file, because the live file would be its only input.
    """
    _require_live_oracle(runtime_env=False)
    runs, reader = _measured(tmp_path)
    problems = check_missing_env_prose(PRE_FIX_BULLET, runs[reader])
    assert problems != [], (
        "the pre-fix bullet passes the rule, so the rule does not detect the "
        "rot this task is about")
    assert any("inside a run that started" in p for p in problems), problems


@pytest.mark.parametrize("variant", [
    pytest.param(REWORDED_BUT_STILL_WRONG, id="reworded-still-run-time"),
    pytest.param(UNMEASURED_EXIT_CODE, id="exit-code-not-measured"),
    pytest.param(WRONG_LIBRARY, id="libapr-not-the-unresolved-library"),
    pytest.param(VAGUE_BUT_TRUE, id="true-but-unquotable"),
])
def test_the_claim_rule_can_still_fail(variant, tmp_path):
    """Four ways a re-fix could go wrong, each caught, none of them by luck."""
    _require_live_oracle(runtime_env=False)
    runs, reader = _measured(tmp_path)
    problems = check_missing_env_prose(variant, runs[reader])
    assert problems != [], f"this wrong variant passes the rule:\n{variant}"


def test_the_rule_accepts_the_corrected_text(tmp_path):
    """Anti-vacuity in the other direction.

    The rule must not simply reject everything: a faithful, load-time,
    evidence-quoting bullet has to come out clean, or "fix" and "delete" are
    the same action and the next reader has no guidance left.
    """
    _require_live_oracle(runtime_env=False)
    runs, reader = _measured(tmp_path)
    truth = runs[reader]
    corrected = (
        f"* ``{TOKEN}`` -- the native ``.k`` reader.  Not cosmetic: without "
        f"it the dynamic loader refuses the binary at load time, before any of "
        f"its own code runs, so no message call is reached; "
        f"``starter_linux64_gf`` exits {truth['code']} with "
        f"``{loader_message(truth['stderr'])}``.  ``engine_linux64_gf`` "
        f"carries no reader NEEDED entry, so it needs no such export.\n")
    assert check_missing_env_prose(corrected, truth) == [], (
        "a faithful load-time paragraph is rejected, so this rule cannot "
        "tell a correct claim from a wrong one")


# ---------------------------------------------------------------------------
# 3. the attribution -- a claim may not be credited to a file that refutes it
# ---------------------------------------------------------------------------

def _cites(paragraph: str) -> list[Path]:
    """Repo-relative files a claim paragraph cites, as ``tools/...`` paths."""
    out = []
    for ref in re.findall(r"`(tools/[A-Za-z0-9_./-]+\.(?:sh|py))`", paragraph):
        path = REPO / ref
        if path.is_file():
            out.append(path)
    return out


def test_a_claim_is_never_credited_to_a_file_that_refutes_it(tmp_path):
    """A citation must not carry a claim its source contradicts.

    The pre-fix bullet ended ``(tools/oracle/oracle_env.sh)`` — and that file,
    as corrected by ``e1072ca``, states the *opposite*: load-time, exit 127, no
    message call.  So the citation was not merely out of date, it pointed at
    the refutation of the sentence it was asked to support.

    Two halves, both needed: the cited file must itself not assert the
    run-time story (:func:`ld_docs_regions` over its comments — the premise),
    and the citing paragraph must not assert it either when it defers to that
    file.  A dangling citation is reported too: a path that does not exist
    cannot corroborate anything.
    """
    _require_live_oracle(runtime_env=False)
    runs, reader = _measured(tmp_path)
    assert runs[reader]["code"] != 0, "premise: the failure must still be real"

    cited_prose = " ".join(
        ld_docs_regions(ORACLE_ENV.read_text(encoding="utf-8")))
    assert cited_prose, (
        f"tools/oracle/oracle_env.sh no longer documents {TOKEN} at all, so "
        f"tools/validate_vs_fortran.py cannot cite it for the requirement")
    premise = deferred_claims(cited_prose)
    assert premise == [], (
        f"the cited file itself describes a missing {TOKEN} as a run-time "
        f"failure ({premise}); fix it there first, or drop the citation")

    text = VALIDATE.read_text(encoding="utf-8")
    problems: list[str] = []
    for paragraph in _claim_paragraphs(text):
        sources = _cites(paragraph)
        if not sources:
            continue
        for found in deferred_claims(paragraph):
            problems.append(
                f"claims {found} and cites "
                f"{', '.join(str(s.relative_to(REPO)) for s in sources)}, "
                f"which says the failure is load-time before any code runs")
    dangling = sorted({ref for ref in re.findall(
        r"`(tools/[A-Za-z0-9_./-]+\.(?:sh|py))`", text)
        if not (REPO / ref).is_file()})
    assert problems == [], "\n".join(problems)
    assert dangling == [], (
        f"tools/validate_vs_fortran.py cites files that do not exist: "
        f"{dangling}")


def test_the_file_cites_the_upstream_requirement_for_the_export():
    """``INSTALL.md:34-42`` is where upstream states the export.

    Kept in this file because the bullet is the place a reader looks first,
    and a citation that drifted off the requirement line would be as useless
    as the citation this task removes.
    """
    text = VALIDATE.read_text(encoding="utf-8")
    cites = re.findall(r"INSTALL\.md:(\d+)(?:-(\d+))?", text)
    assert cites, "no INSTALL.md path:line citation in the harness"
    covering = [(int(lo), int(hi or lo)) for lo, hi in cites
                if int(lo) <= 42 <= int(hi or lo)]
    assert covering, (
        f"no INSTALL.md citation covers line 42, where the {TOKEN} export is "
        f"stated; citations found: {sorted(set(cites))}")