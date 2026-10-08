"""
Subprocess orchestration + listing parsing for the pyradioss GUI.

No Fortran counterpart (see the package docstring). This module deliberately
imports **no tkinter**: everything here is plain logic (subprocess control,
text parsing, file loading, JSON config) so it can be unit-tested head-less
and driven by a scripted caller without opening a window.

What it provides
----------------
* :func:`parse_listing_line` / :func:`parse_termination` — read the Engine's
  periodic cycle listing (``  CYCLE  TIME  TIME-STEP  ENERGY-IE ...``) and the
  ``ENGINE/STARTER TERMINATION : NORMAL|ERROR`` banner lines emitted by
  :mod:`pyradioss.engine.engine` / :mod:`pyradioss.starter.starter`.
* :func:`derive_engine_deck` / :func:`t01_path_for` — the Radioss file-naming
  contract (``RunName_0000.rad`` -> ``RunName_0001.rad`` -> ``RunNameT01.csv``).
* :func:`load_t01` — read the T01 time-history CSV written by
  :mod:`pyradioss.output.time_history` (tolerant of partial rows mid-run).
* :func:`build_deck_summary` — run the port's **read-only** deck reader in
  process and return a model summary (never raises: a bad deck yields the
  captured Starter error listing instead of a crash).
* :class:`GuiConfig` — a tiny JSON config in the user home (last directory,
  backend, threads, selected channels).
* :class:`JobRunner` — runs STARTER then ENGINE as subprocesses (same
  interpreter, ``-m pyradioss.starter`` / ``-m pyradioss.engine``), streaming
  stdout line-by-line into a :class:`queue.Queue` of events on a worker
  thread so the UI never blocks; supports clean ``stop()``.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# File-naming contract (mirrors starter/engine run_name_from_input)
# ---------------------------------------------------------------------------

_RUN_RE = re.compile(r"(.+)_(\d{4})\.rad$", re.IGNORECASE)


def available_cpus() -> int:
    """CPUs this machine offers the run (the upper bound of the GUI's CPU
    spinbox): the scheduler affinity when the OS exposes it, else
    ``os.cpu_count()``, never less than 1."""
    try:
        n = len(os.sched_getaffinity(0))          # Linux: honours cgroups/taskset
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, int(n))


def derive_engine_deck(starter_deck: str) -> str:
    """``RunName_0000.rad`` -> ``RunName_0001.rad`` (the first Engine run).

    Any other run number ``nnnn`` maps to ``nnn+1``; a path that does not
    match the ``_NNNN.rad`` convention is returned unchanged.
    """
    base = os.path.basename(starter_deck)
    m = _RUN_RE.match(base)
    if not m:
        return starter_deck
    nxt = int(m.group(2)) + 1
    engine_base = f"{m.group(1)}_{nxt:04d}.rad"
    return os.path.join(os.path.dirname(starter_deck), engine_base)


def run_name_and_num(deck: str) -> Tuple[str, int]:
    """``RunName_0001.rad`` -> ``('RunName', 1)`` (run number 1 for a
    non-matching name, matching engine.run_name_from_input)."""
    base = os.path.basename(deck)
    m = _RUN_RE.match(base)
    if m:
        return m.group(1), int(m.group(2))
    return os.path.splitext(base)[0], 1


def t01_path_for(engine_deck: str) -> str:
    """The T01 time-history CSV an Engine run of ``engine_deck`` writes:
    ``{RunName}T{nn:02d}.csv`` next to the deck (see
    :mod:`pyradioss.output.time_history` and engine.py's TimeHistory path)."""
    run_name, run_num = run_name_and_num(engine_deck)
    out_dir = os.path.dirname(os.path.abspath(engine_deck))
    return os.path.join(out_dir, f"{run_name}T{run_num:02d}.csv")


# ---------------------------------------------------------------------------
# Listing parsing
# ---------------------------------------------------------------------------

#: names of the nine floating-point fields following the integer CYCLE in a
#: listing line (see engine.py's periodic ``log.info`` block).
_LISTING_FIELDS = ("time", "dt", "ie", "ke", "he", "ce", "en", "ew", "err")


def parse_listing_line(line: str) -> Optional[Dict[str, float]]:
    """Parse one periodic Engine cycle line into a status dict, or ``None``.

    A listing line is::

        <cycle:int> <time> <dt> <IE> <KE> <HE> <CE> <EN> <EXT-WORK> <ERROR%>

    i.e. exactly ten whitespace-separated tokens, the first an integer and
    the rest floats. The header row (``CYCLE TIME TIME-STEP ...``) and every
    other message fail the int/float test and return ``None``.
    """
    tokens = line.split()
    if len(tokens) != 10:
        return None
    try:
        cycle = int(tokens[0])
    except ValueError:
        return None
    try:
        vals = [float(t.rstrip("%")) for t in tokens[1:]]
    except ValueError:
        return None
    status: Dict[str, float] = {"cycle": cycle}
    status.update(zip(_LISTING_FIELDS, vals))
    return status


def parse_termination(line: str) -> Optional[Tuple[str, str, str]]:
    """Detect an ``ENGINE/STARTER TERMINATION`` banner line.

    Returns ``(kind, status, reason)`` where ``kind`` is ``'ENGINE'`` or
    ``'STARTER'``, ``status`` is ``'NORMAL'`` or ``'ERROR'`` and ``reason``
    is the trailing text of an error (may be empty). Returns ``None`` for
    any other line.
    """
    s = line.strip()
    for kind in ("ENGINE", "STARTER"):
        marker = f"{kind} TERMINATION :"
        idx = s.find(marker)
        if idx < 0:
            continue
        rest = s[idx + len(marker):].strip()
        up = rest.upper()
        if up.startswith("NORMAL"):
            return (kind, "NORMAL", "")
        if up.startswith("ERROR"):
            # "ERROR" or "ERROR — <reason>" (em dash) or "ERROR - <reason>"
            reason = rest[len("ERROR"):].strip().lstrip("-—").strip()
            return (kind, "ERROR", reason)
        return (kind, up.split()[0] if up else "", rest)
    return None


# ---------------------------------------------------------------------------
# T01 time-history loading
# ---------------------------------------------------------------------------

class T01Data:
    """A loaded T01 CSV: column names + per-column float series.

    Kept dependency-light (plain lists) so the loader works without numpy;
    :meth:`column` returns the series for a channel by name.
    """

    def __init__(self, columns: List[str], rows: List[List[float]]):
        self.rows = rows
        # transpose to column-major series for easy channel access
        seen = {}
        unique_cols = []
        for c in columns:
            if c in seen:
                seen[c] += 1
                unique_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 1
                unique_cols.append(c)
        self.columns = unique_cols
        self._series: Dict[str, List[float]] = {c: [] for c in self.columns}
        for row in rows:
            for c, v in zip(self.columns, row):
                self._series[c].append(v)

    @property
    def nrows(self) -> int:
        return len(self.rows)

    @property
    def time(self) -> List[float]:
        return self._series.get(self.columns[0], []) if self.columns else []

    def column(self, name: str) -> List[float]:
        return self._series.get(name, [])

    def has(self, name: str) -> bool:
        return name in self._series


def load_t01(path: str) -> T01Data:
    """Load the T01 time-history CSV (see
    :mod:`pyradioss.output.time_history`).

    Line 1 is a ``#`` comment, line 2 the header, the rest data rows.
    Robust to a partial final row (the Engine flushes each row, but a live
    reader can still catch a truncated line): any row whose field count does
    not match the header, or that fails to parse, is skipped.
    """
    columns: List[str] = []
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not columns:
                columns = line.split(",")
                continue
            fields = line.split(",")
            if len(fields) != len(columns):
                continue
            try:
                rows.append([float(f) for f in fields])
            except ValueError:
                continue
    return T01Data(columns, rows)


# ---------------------------------------------------------------------------
# Read-only deck summary (deck-info tab)
# ---------------------------------------------------------------------------

def _read_engine_t_end(engine_deck: str) -> float:
    """Parse the Engine deck's ``/RUN`` end time (T_stop) in process,
    read-only. Returns 0.0 on any problem (no progress estimate then)."""
    if not os.path.exists(engine_deck):
        return 0.0
    import contextlib
    import io
    try:
        from ..input.deck_reader import read_deck
        from ..input.engine_keywords import parse_engine_deck
        from ..common.messages import MessageLog
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            controls = parse_engine_deck(read_deck(engine_deck), MessageLog())
        return float(controls.t_end)
    except Exception:
        return 0.0


def build_deck_summary(deck_path: str) -> Dict[str, object]:
    """Run the port's read-only deck reader on ``deck_path`` and return a
    model summary dict. **Never raises** — a deck that cannot be read yields
    a summary carrying the captured Starter error listing under ``'error'``
    / ``'listing'`` instead of crashing the GUI.

    Keys: ``path``, ``title``, ``nodes`` (int), ``elements`` (dict
    type->count), ``materials`` / ``properties`` / ``parts`` / ``contacts``
    (lists of short dicts), ``keywords`` (list of (keyword, count)),
    ``warnings`` / ``errors`` (int), ``listing`` (captured stdout),
    ``error`` (top-level failure message or '').
    """
    import contextlib
    import io

    summary: Dict[str, object] = {
        "path": deck_path,
        "title": "",
        "nodes": 0,
        "elements": {},
        "materials": [],
        "properties": [],
        "parts": [],
        "contacts": [],
        "keywords": [],
        "warnings": 0,
        "errors": 0,
        "listing": "",
        "error": "",
    }
    buf = io.StringIO()
    try:
        from ..input.deck_reader import read_deck
        from ..input.keywords import parse_starter_deck
        from ..model.model import Model
        from ..common.messages import MessageLog

        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            blocks = read_deck(deck_path)
            model = Model()
            log = MessageLog()
            parse_starter_deck(blocks, model, log)

        summary["title"] = getattr(model, "title", "")
        summary["nodes"] = int(model.numnod)

        # element counts by type (raw_elems is filled at parse time, before
        # the Starter groups them into ElementGroups)
        elem_counts = {k: len(v) for k, v in model.raw_elems.items() if v}
        summary["elements"] = elem_counts

        summary["materials"] = [
            {
                "id": mat.id,
                "law": mat.law,
                "law_name": getattr(mat, "law_name", "") or f"LAW{mat.law}",
                "title": mat.title,
            }
            for mat in model.materials.values()
        ]
        summary["properties"] = [
            {"id": p.id, "type": p.type, "title": p.title}
            for p in model.properties.values()
        ]
        summary["parts"] = [
            {"id": p.id, "prop_id": p.prop_id, "mat_id": p.mat_id,
             "title": p.title}
            for p in model.parts.values()
        ]
        summary["contacts"] = [
            {"id": itf.id, "type": itf.type}
            for itf in model.interfaces
        ]

        # keyword histogram (deck order collapsed to counts)
        kw_counts: Dict[str, int] = {}
        for block in blocks:
            key = block.keyword or "/".join(block.parts)
            kw_counts[key] = kw_counts.get(key, 0) + 1
        summary["keywords"] = sorted(kw_counts.items())

        summary["warnings"] = len(log.warnings)
        summary["errors"] = len(log.errors)
    except Exception as exc:  # never let a bad deck crash the GUI
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        summary["listing"] = buf.getvalue()
    return summary


# ---------------------------------------------------------------------------
# Persistent config
# ---------------------------------------------------------------------------

class GuiConfig:
    """A tiny JSON config in the user home (``~/.pyradioss_gui/config.json``).

    Remembers the last-used directory, backend, CPU count (``-np`` SPMD
    domains), thread count and the Results-tab channel selection across
    sessions. Load/save never raise —
    a missing or corrupt file falls back to defaults.
    """

    DEFAULTS = {
        "last_dir": "",
        "backend": "auto",
        # CPUs = SPMD domains handed to `starter -np N` / `engine -np N`
        # (1 = the serial run; N > 1 decomposes the model into N domains
        # that run as N threads, or as N MPI processes under mpirun).
        "nspmd": 1,
        "nthread": 0,
        "channels": ["IE", "KE", "EW", "ERR%"],
        # Post-processing: the OpenRadioss Fortran converter exe directory and
        # the "convert automatically after a clean run" toggles (see
        # pyradioss.gui.postproc).
        "exec_dir": "",
        "auto_d3plot": False,
        "auto_vtk": False,
        "auto_th_csv": False,
    }

    def __init__(self, path: Optional[str] = None):
        self.path = path or self.default_path()
        self.data: Dict[str, object] = dict(self.DEFAULTS)

    @staticmethod
    def default_path() -> str:
        return os.path.join(os.path.expanduser("~"), ".pyradioss_gui",
                            "config.json")

    def load(self) -> "GuiConfig":
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                for key in self.DEFAULTS:
                    if key in stored:
                        self.data[key] = stored[key]
        except (OSError, ValueError):
            pass  # missing / corrupt -> keep defaults
        return self

    def save(self) -> None:
        try:
            dirname = os.path.dirname(self.path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self.path, "w", encoding="utf-8", errors="replace") as fh:
                json.dump(self.data, fh, indent=2)
        except OSError:
            pass  # a read-only home must not crash the GUI

    def get(self, key: str, default: object = None) -> object:
        return self.data.get(key, default)

    def set(self, key: str, value: object) -> None:
        self.data[key] = value


# ---------------------------------------------------------------------------
# Job runner (subprocess orchestration)
# ---------------------------------------------------------------------------

# Event tuples pushed onto the queue (first element is the tag):
#   ("t_end", float)                       run end time for progress
#   ("phase", "starter"|"engine")          a phase started
#   ("line", phase, text)                  one raw stdout line
#   ("status", phase, status_dict)         a parsed cycle line
#   ("term", kind, status, reason)         a termination banner
#   ("done", returncode)                   the whole job finished/stopped

class JobRunner:
    """Runs STARTER then ENGINE as subprocesses of the *same* interpreter.

    stdout is streamed line-by-line on a worker thread into ``self.queue``
    (a :class:`queue.Queue` of the event tuples above) so a Tk UI can drain
    it from its event loop without ever blocking; a scripted caller can use
    :meth:`run_to_completion` instead.
    """

    def __init__(self, starter_deck: str, backend: str = "auto",
                 nthread: int = 0, python_exe: Optional[str] = None,
                 event_queue: Optional["queue.Queue"] = None,
                 post_actions: Optional[List[str]] = None,
                 exec_dir: Optional[str] = None, nspmd: int = 1):
        self.starter_deck = os.path.abspath(starter_deck)
        self.engine_deck = derive_engine_deck(self.starter_deck)
        self.backend = backend
        self.nthread = int(nthread)
        # CPUs: the number of SPMD domains (`-np N` of BOTH the Starter,
        # which writes one restart per domain, and the Engine, which runs
        # the domains — pyradioss.spmd). 1 is the plain serial run.
        self.nspmd = max(1, int(nspmd))
        self.python_exe = python_exe or sys.executable
        self.queue: "queue.Queue" = event_queue or queue.Queue()
        self.work_dir = os.path.dirname(self.starter_deck)
        self.t01_path = t01_path_for(self.engine_deck)
        # Post-processing conversions to run automatically after a clean run
        # (any subset of postproc.PostProcRunner.ALL_ACTIONS); streamed into
        # the same queue on the same worker thread.
        self.post_actions = list(post_actions or [])
        self.exec_dir = exec_dir
        self.post_results: Dict[str, Dict[str, object]] = {}

        self.t_end = 0.0
        self.returncode: Optional[int] = None
        self.termination: Optional[Tuple[str, str, str]] = None
        self.last_status: Optional[Dict[str, float]] = None

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        """Launch the worker thread (returns immediately)."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("job already running")
        self._stop.clear()
        self.returncode = None
        self.termination = None
        self.last_status = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="pyradioss-job")
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        """Ask the worker to stop and terminate the live child cleanly."""
        self._stop.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except OSError:
                pass

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- worker ---------------------------------------------------------
    def _emit(self, event: tuple) -> None:
        self.queue.put(event)

    def _base_cmd(self, module: str) -> List[str]:
        # -u: unbuffered child stdout so lines stream live to the pipe.
        cmd = [self.python_exe, "-u", "-m", module]
        if self.nspmd > 1:
            # the Starter decomposes into N domains and the Engine must be
            # run with the same N (inipar.F coherence test) — one flag,
            # both programs.
            cmd += ["-np", str(self.nspmd)]
        if self.nthread > 0:
            cmd += ["-nt", str(self.nthread)]
        return cmd

    def _starter_cmd(self) -> List[str]:
        return self._base_cmd("pyradioss.starter") + ["-i", self.starter_deck]

    def _engine_cmd(self) -> List[str]:
        cmd = self._base_cmd("pyradioss.engine") + ["-i", self.engine_deck]
        if self.backend:
            cmd += ["-backend", self.backend]
        return cmd

    def _run(self) -> None:
        self.t_end = _read_engine_t_end(self.engine_deck)
        self._emit(("t_end", self.t_end))

        # phase 1: STARTER
        self._emit(("phase", "starter"))
        rc = self._run_phase("starter", self._starter_cmd())
        if rc != 0 or self._stop.is_set():
            if self._stop.is_set():
                rc = -1
            self.returncode = rc
            self._emit(("done", rc))
            return

        # phase 2: ENGINE
        self._emit(("phase", "engine"))
        cmd = self._engine_cmd()
        rc = self._run_phase("engine", cmd)
        if self._stop.is_set():
            rc = -1
        self.returncode = rc

        # optional auto-convert after a clean, NORMAL engine run (same worker
        # thread + same queue, so it never blocks the UI).
        if rc == 0 and self.post_actions and not self._stop.is_set() \
                and self._engine_terminated_normally():
            self._run_post_actions()

        self._emit(("done", rc))

    def _engine_terminated_normally(self) -> bool:
        term = self.termination
        return bool(term and term[0] == "ENGINE" and term[1] == "NORMAL")

    def _run_post_actions(self) -> None:
        """Fire the selected post-processing conversions, streaming their
        events into this runner's queue. Import is local so the base runner
        stays importable even if postproc's optional deps are absent."""
        from . import postproc
        self._emit(("phase", "postproc"))
        try:
            postproc.run_post_actions(
                self.work_dir, self.post_actions, exec_dir=self.exec_dir,
                emit=self._emit, python_exe=self.python_exe,
                results=self.post_results)
        except Exception as exc:  # a converter must never crash the job
            self._emit(("line", "postproc",
                        f" ** post-processing error: {exc}"))

    @staticmethod
    def _package_env(backend: Optional[str] = None) -> Dict[str, str]:
        """Child env with the ``pyradioss`` package root on ``PYTHONPATH`` so
        ``python -m pyradioss.starter`` imports even when the child's cwd is
        the deck directory (a source checkout is not on the default path)."""
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # pin BOTH ends of the pipe to UTF-8: without this the child's
        # stdout follows the Windows locale (cp1252) while _run_phase now
        # decodes UTF-8 (same latent crash class as the postproc streamer)
        env["PYTHONIOENCODING"] = "utf-8"
        # dir that contains the 'pyradioss' package (two levels up from this
        # module: .../pyradioss/gui/runner.py -> repo root / install root)
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        existing = env.get("PYTHONPATH", "")
        parts = [pkg_root] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        if backend:
            env["PYRADIOSS_BACKEND"] = str(backend)
        return env

    def _run_phase(self, name: str, cmd: List[str]) -> int:
        env = self._package_env(backend=self.backend)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                cwd=self.work_dir, env=env)
        except OSError as exc:
            self._emit(("line", name, f" ** GUI: failed to launch {name}: "
                                      f"{exc}"))
            return -1
        with self._lock:
            self._proc = proc
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                self._emit(("line", name, line))
                term = parse_termination(line)
                if term is not None:
                    self.termination = term
                    self._emit(("term",) + term)
                else:
                    status = parse_listing_line(line)
                    if status is not None:
                        self.last_status = status
                        self._emit(("status", name, status))
                if self._stop.is_set():
                    break
        finally:
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except OSError:
                pass
            try:
                rc = proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
            with self._lock:
                self._proc = None
        return rc

    # -- scripted convenience ------------------------------------------
    def run_to_completion(self, timeout: float = 300.0) -> Dict[str, object]:
        """Start the job and drain the queue until it finishes (or times
        out). Returns a small result dict; for head-less / test use."""
        self.start()
        deadline = time.time() + timeout
        while True:
            try:
                event = self.queue.get(timeout=1.0)
            except queue.Empty:
                if time.time() > deadline:
                    self.stop()
                    raise TimeoutError(
                        f"job did not finish within {timeout:.0f}s")
                continue
            if event[0] == "done":
                break
            if time.time() > deadline:
                self.stop()
                raise TimeoutError(
                    f"job did not finish within {timeout:.0f}s")
        self.join(timeout=10)
        return {
            "returncode": self.returncode,
            "termination": self.termination,
            "last_status": self.last_status,
            "t01_path": self.t01_path,
            "t_end": self.t_end,
        }
