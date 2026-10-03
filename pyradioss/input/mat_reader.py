"""
Generic cfg-driven /MAT reader — EVERY material law parses (M37).

Fortran origin: the ``hm_read_mat*.F`` routines under
``starter/source/materials/mat/mat###`` are all driven by the *same* card
descriptions the pre-processors use: the CFG files shipped with the
Fortran build (``hm_cfg_files/config/CFG/radioss*/MAT/*.cfg``).  Each cfg
file declares the law's attributes (``ATTRIBUTES(COMMON)``), their
defaults (``DEFAULTS(COMMON)``) and — the part this module executes — a
``FORMAT(radioss<version>)`` block: a tiny imperative program of
``CARD("%20lg%10d...", attr, ...)`` statements, conditionals on already-
read values, ``CARD_LIST``/``CELL_LIST`` loops and ``ASSIGN``s.  That
program *is* the authoritative card layout the real Starter parses.

This module

1. **parses the CFG DSL** (the subset the MAT family uses — see
   :class:`_CfgInterpreter` for the statement inventory) into per-law
   schemas, resolving each law to the newest cfg definition at or below
   ``radioss2022`` exactly like the HM reader does (the version
   directories are incremental: a law untouched since radioss110 lives
   there and nowhere else);

2. **reads any /MAT/<LAW> block** into a :class:`GenericMaterialRecord`
   — law name/number, id, title, ``params`` {cfg attribute -> value} and
   the density — using the cfg card layout, fixed-column slicing
   included (a 20-char field whose neighbour touches it, e.g. LAW37's
   ``...2089.0                 0.01.00000000000000E-03``, parses
   correctly where any whitespace tokenizer would fail);

3. hands the record to a registered **physics constructor** when one
   exists (:data:`MAT_PHYSICS_REGISTRY`), and otherwise stores an
   :class:`InactiveMaterial`: the full parameter set and the density are
   available (mass initialization works, the Starter listing reports the
   material as *parsed, physics not implemented (M37)*), but the Engine
   REFUSES to run a model whose element groups reference it
   (:func:`refuse_inactive_materials`) — parse-clean but honestly not
   simulatable.

The physics-registry contract (for the LAW19/24/35/44/70/81, VOID, GAS
builders of the next phase)
---------------------------------------------------------------------
Register a constructor under the law's canonical name (and/or its
``LAW<n>`` alias — lookup tries the header spelling, the canonical cfg
name and ``LAW<number>``)::

    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW19"] = build_law19       # also finds FABRI

The constructor receives the :class:`GenericMaterialRecord` and must
return an object satisfying the existing material kernel API (see
``materials/law02_johnson_cook.py`` and ``model/entities.Material`` for
the contract), i.e. at least:

* ``id``, ``law`` (int — the ``materials`` package dispatches on it),
  ``rho0``, ``title``, ``params`` (dict), ``fail`` (None or a
  FailureModel), ``eos`` (None or an EquationOfState);
* the elastic properties ``E``, ``nu``, ``G``, ``K`` (properties on
  ``Material`` — populate ``params['E']``/``params['nu']``), which
  drive sound speed / time step / contact stiffness;
* the stress-update entry points for its law number must exist in
  ``pyradioss.materials`` (``solid_update``/``shell_update`` dispatch) —
  that is the physics half the registry builder brings.

The simplest valid constructor returns a plain
``Material(id=r.id, law=<n>, rho0=r.density, title=r.title,
params={...})`` whose params carry what its ``sigeps`` port needs.

If the cfg tree is installed nowhere (no ``PYRADIOSS_HM_CFG``, no upstream
``RAD_CFG_PATH``, no sibling-of-build or Windows-compat tree — the resolver
in :mod:`pyradioss.paths` reports every location it tried) the reader
degrades to a heuristic: the density is taken from the first data card and
the raw cards are kept in ``params['raw_cards']`` — still parse-clean, just
without named fields.  The degradation is never silent: the warning carries
the resolver's full candidate list.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..common.messages import MessageLog
from ..model.entities import Material
from ..paths import hm_cfg_dir, is_cfg_schema_dir
from .deck_reader import Card, KeywordBlock, parse_fortran_float

# ============================================================================
# Configuration: where the cfg tree lives, and which input version to target
# ============================================================================

#: newest cfg FORMAT/directory version considered (the 2022 fixed format —
#: the dialect the M35/M36 validation harness proved against the real
#: Starter binary).
MAX_CFG_VERSION = 2022

#: Replayed in the "/MAT: no cfg schema found" warning when the cfg tree is
#: nowhere to be found — set by :func:`_find_cfg_root`, read by the
#: heuristic fallback.  Before Task P0.6 the reader probed a tuple built
#: from ``os.environ`` at IMPORT time plus two hardcoded roots and returned
#: ``None`` in silence, which is ``docs/OPEN_BUGS.md`` item 6: /MAT/LAW4
#: lost ``E`` and its test failed with no clue why.
_CFG_SEARCH_FAILED = ""


class _Resolve:
    """Sentinel for "resolve the cfg root when this catalogue is built".

    Distinct from ``root=None``, which is a real answer: "there is no cfg
    tree".  Without it ``CfgCatalogue()`` could not tell a caller asking
    for an explicit root off from one asking for the resolved one.
    """

    __slots__ = ()

    def __repr__(self) -> str:                       # pragma: no cover
        return "<resolve>"


_RESOLVE = _Resolve()


def _find_cfg_root() -> Optional[str]:
    """The ``config/CFG`` schema directory of the resolved cfg tree, or
    ``None`` when there is no readable cfg tree anywhere.

    All *path* knowledge lives in :mod:`pyradioss.paths` (the single
    resolver, Task P0.6).  What this function adds is the **layout**
    knowledge, and it accepts **both** spellings of the variable because
    both are in use:

    * ``PYRADIOSS_HM_CFG=<tree>`` — the documented one
      (``plan/00_ORCHESTRATION.md`` §4.1, upstream's ``RAD_CFG_PATH`` =
      ``$OPENRADIOSS_PATH/hm_cfg_files``, ``$OR_SRC/INSTALL.md:34-42``), from
      which ``<tree>/config/CFG`` is taken;
    * ``PYRADIOSS_HM_CFG=<tree>/config/CFG`` — what
      ``.github/workflows/ci.yml`` exports and what the pre-P0.6
      ``_DEFAULT_CFG_ROOTS`` tuple accepted, i.e. the schema directory
      itself.

    The two are told apart by **inspecting the filesystem** (does
    ``<cfg>/config/CFG`` exist?), never by the shape of the string.  A
    directory named ``CFG`` that carries no ``radioss<version>`` schema
    subdirectory is rejected: accepting it would resolve the tree and then
    parse nothing, which is the silent degradation Task P0.6 removed.
    """
    global _CFG_SEARCH_FAILED
    try:
        cfg = hm_cfg_dir()
    except FileNotFoundError as exc:
        _CFG_SEARCH_FAILED = str(exc)
        return None
    nested = os.path.join(str(cfg), "config", "CFG")
    if os.path.isdir(nested):
        root = nested
    elif is_cfg_schema_dir(cfg):
        root = str(cfg)               # the variable pointed at the CFG dir
    else:
        _CFG_SEARCH_FAILED = (
            f"{cfg} is neither an hm_cfg_files tree (…/config/CFG) nor a "
            f"CFG schema directory (…/radioss<version>); upstream's layout "
            f"is $OPENRADIOSS_PATH/hm_cfg_files/config/CFG "
            f"($OR_SRC/INSTALL.md:34-42)")
        return None
    if not is_cfg_schema_dir(root):  # defence in depth for an injected root
        _CFG_SEARCH_FAILED = (
            f"{root} carries no radioss<version> schema subdirectory — an "
            f"empty or partial hm_cfg_files tree cannot be read")
        return None
    _CFG_SEARCH_FAILED = ""
    return root


# ============================================================================
# Records and the physics registry
# ============================================================================

@dataclass
class GenericMaterialRecord:
    """Everything a /MAT block said, in cfg-attribute terms.

    ``params`` maps the cfg ATTRIBUTE names (``MAT_RHO``, ``MAT_E``,
    ``FUN_A1`` ...) to the parsed values — scalars for VALUE attributes,
    lists for ARRAY attributes; unread fields carry their cfg DEFAULTS.
    ``density`` is the initial density (``MAT_RHO`` in nearly every law;
    0.0 for the few laws without one, e.g. /MAT/GAS)."""

    law_name: str                       # header spelling, e.g. "FABRI"
    law_number: Optional[int]           # 19 for FABRI; None if unknown
    id: int
    title: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    density: float = 0.0
    subtype: str = ""                   # e.g. "MASS" for /MAT/GAS/MASS
    unit_id: Optional[int] = None       # /MAT/<LAW>/<id>/<unit_id>
    cfg_file: str = ""                  # provenance (schema used)
    raw_cards: List[str] = field(default_factory=list)

    @property
    def rho0(self) -> float:
        """Initial density alias for Material compatibility."""
        return self.density


#: law name -> constructor(record) -> material-kernel-compatible object.
#: The physics builders (LAW19/24/35/44/70/81, VOID, GAS ...) register
#: themselves here with one line — see the module docstring contract.
MAT_PHYSICS_REGISTRY: Dict[str, Callable[[GenericMaterialRecord], object]] = {}


@dataclass
class InactiveMaterial(Material):
    """A /MAT law that PARSES (full params, density, mass init works)
    but has no ported physics: the Engine refuses to run element groups
    referencing it (see :func:`refuse_inactive_materials`).  ``law`` is
    the Radioss law number when known (-1 otherwise); ``E``/``nu`` in
    ``params`` are safe fallbacks (from ``MAT_E``/``MAT_NU`` when the
    law has them) so Starter-side stiffness/time-step *estimates* never
    divide by zero — they carry no physical meaning here."""

    law_name: str = ""
    record: Optional[GenericMaterialRecord] = None

    #: duck-typing marker the Starter checks / Engine refusal use
    #: (class attribute — survives pickling through the restart file)
    inactive = True


class InactiveMaterialError(RuntimeError):
    """Raised by the Engine when the model references a parsed-but-not-
    implemented material (see :func:`refuse_inactive_materials`)."""


def refuse_inactive_materials(model, log: Optional[MessageLog] = None) -> None:
    """The Engine's group-build refusal point (M37): raise
    :class:`InactiveMaterialError` if any element group's part references
    an :class:`InactiveMaterial`, naming every offending law.  Materials
    that are merely *defined* (e.g. a /MAT/GAS for an airbag injector)
    but referenced by no elements do not block the run."""
    bad: Dict[Tuple[int, str], set] = {}
    for name, group in model.element_groups():
        for _sl, mat, _prop in group.state.get("slices", []):
            if getattr(mat, "inactive", False):
                key = (mat.id, getattr(mat, "law_name", None)
                       or f"LAW{mat.law}")
                bad.setdefault(key, set()).add(name)
    if not bad:
        return
    lines = [f"/MAT/{law}/{mid}: parsed, physics not implemented (M37) — "
             f"referenced by {', '.join(sorted(groups))} elements"
             for (mid, law), groups in sorted(bad.items())]
    msg = ("the Engine cannot run this model — element groups reference "
           "material laws whose physics is not implemented:\n  "
           + "\n  ".join(lines)
           + "\n  (the Starter accepted them at parse level; implement or "
             "register the law in MAT_PHYSICS_REGISTRY, or change the "
             "part's material)")
    if log is not None:
        log.error(msg, "MAT CHECK")
    raise InactiveMaterialError(msg)


# ============================================================================
# CFG file parsing — the DSL subset the MAT family uses
# ============================================================================

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(text: str) -> str:
    """Remove ``//`` line comments and ``/* */`` block comments.  The
    cfg format strings never contain ``//`` (checked over the whole MAT
    tree), so plain regex stripping is safe."""
    text = _BLOCK_COMMENT.sub("", text)
    out = []
    for line in text.splitlines():
        # keep '//' inside double quotes (defensive; none observed)
        cut = None
        in_q = False
        i = 0
        while i < len(line) - 1:
            if line[i] == '"':
                in_q = not in_q
            elif not in_q and line[i] == "/" and line[i + 1] == "/":
                cut = i
                break
            i += 1
        out.append(line[:cut] if cut is not None else line)
    return "\n".join(out)


def _match_brace(text: str, pos: int, open_ch: str = "{",
                 close_ch: str = "}") -> int:
    """Index of the brace closing the one at ``pos`` (quote-aware)."""
    depth = 0
    in_q = False
    for i in range(pos, len(text)):
        c = text[i]
        if c == '"':
            in_q = not in_q
        elif not in_q:
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return i
    raise ValueError(f"unbalanced {open_ch}...{close_ch} in cfg file")


def _split_args(argstr: str) -> List[str]:
    """Split an argument list on top-level commas (quote/paren aware)."""
    args, depth, in_q, cur = [], 0, False, []
    for c in argstr:
        if c == '"':
            in_q = not in_q
            cur.append(c)
        elif in_q:
            cur.append(c)
        elif c in "([":
            depth += 1
            cur.append(c)
        elif c in ")]":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if cur and "".join(cur).strip():
        args.append("".join(cur).strip())
    return args


_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _parse_statements(text: str, pos: int, end: int) -> Tuple[list, int]:
    """Parse the statement list of a FORMAT body (or nested block) into
    tuples — the statement inventory of the MAT cfg family:

        ("call", NAME, [args...])          CARD/COMMENT/ASSIGN/HEADER/...
        ("blank",)                         BLANK;
        ("if", [(cond, stmts), ...], else_stmts_or_None)
        ("loop", NAME, count_expr, stmts)  CARD_LIST / FREE_CARD_LIST
    """
    stmts: list = []
    while pos < end:
        c = text[pos]
        if c.isspace() or c == ";":
            pos += 1
            continue
        if c == "}":
            break
        m = _WORD.match(text, pos)
        if not m:
            pos += 1                       # stray token — skip
            continue
        word = m.group(0)
        pos = m.end()
        # ---- if / else if / else chains --------------------------------
        if word == "if":
            branches = []
            else_stmts = None
            while True:
                p = text.index("(", pos)
                q = _match_brace(text, p, "(", ")")
                cond = text[p + 1:q]
                pos = q + 1
                while pos < end and text[pos].isspace():
                    pos += 1
                if pos < end and text[pos] == "{":
                    bend = _match_brace(text, pos)
                    body, _ = _parse_statements(text, pos + 1, bend)
                    pos = bend + 1
                else:                      # single-statement branch
                    body, pos = _parse_one(text, pos, end)
                branches.append((cond, body))
                # else / else if?
                save = pos
                while pos < end and text[pos].isspace():
                    pos += 1
                m2 = _WORD.match(text, pos)
                if not m2 or m2.group(0) != "else":
                    pos = save
                    break
                pos = m2.end()
                while pos < end and text[pos].isspace():
                    pos += 1
                m3 = _WORD.match(text, pos)
                if m3 and m3.group(0) == "if":
                    pos = m3.end()
                    continue               # another (cond, body) branch
                if pos < end and text[pos] == "{":
                    bend = _match_brace(text, pos)
                    else_stmts, _ = _parse_statements(text, pos + 1, bend)
                    pos = bend + 1
                else:
                    else_stmts, pos = _parse_one(text, pos, end)
                break
            stmts.append(("if", branches, else_stmts))
            continue
        # ---- word without parens (BLANK;) ------------------------------
        while pos < end and text[pos].isspace():
            pos += 1
        if pos >= end or text[pos] != "(":
            stmts.append(("blank",) if word.upper() == "BLANK"
                         else ("call", word.upper(), []))
            continue
        # ---- NAME(args) [;{...}] ---------------------------------------
        q = _match_brace(text, pos, "(", ")")
        args = _split_args(text[pos + 1:q])
        pos = q + 1
        upper = word.upper()
        if upper in ("CARD_LIST", "FREE_CARD_LIST"):
            while pos < end and text[pos].isspace():
                pos += 1
            body: list = []
            if pos < end and text[pos] == "{":
                bend = _match_brace(text, pos)
                body, _ = _parse_statements(text, pos + 1, bend)
                pos = bend + 1
            stmts.append(("loop", upper, args[0] if args else "0", body))
        else:
            stmts.append(("call", upper, args))
    return stmts, pos


def _parse_one(text: str, pos: int, end: int) -> Tuple[list, int]:
    """Parse exactly one statement (for brace-less if/else branches)."""
    sub, newpos = _parse_statements(text, pos, end)
    # _parse_statements consumes as much as it can; for a brace-less
    # branch we only want the first statement — re-scan conservatively.
    if not sub:
        return [], pos
    # find the end of the first statement: the first ';' at depth 0
    depth = 0
    i = pos
    while i < end:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == ";" and depth == 0:
            i += 1
            break
        i += 1
    one, _ = _parse_statements(text, pos, i)
    return one[:1], i


# ---- schema containers ------------------------------------------------------

@dataclass
class _Attr:
    name: str
    type: str            # INT / FLOAT / STRING / FUNCT / ... (cfg spelling)
    is_array: bool = False
    size_var: str = ""


@dataclass
class CfgLawSchema:
    """One law's parsed cfg: attributes, defaults and the FORMAT program
    selected for :data:`MAX_CFG_VERSION`."""

    path: str
    attributes: Dict[str, _Attr]
    defaults: Dict[str, object]
    format_version: int
    statements: list
    law_names: List[str]                 # every keyword spelling seen
    law_number: Optional[int]


_ATTR_RE = re.compile(
    r"(\w+)\s*=\s*(VALUE|SIZE|ARRAY)\s*(?:\[\s*(\w+)\s*\])?\s*\(\s*"
    r"([A-Za-z_]+)?")
_DEFAULT_RE = re.compile(r"(\w+)\s*=\s*([^;{}]+);")
_FORMAT_RE = re.compile(r"FORMAT\s*\(\s*radioss(\d+)\s*\)")
_KEYSTR_RE = re.compile(r'ASSIGN\s*\(\s*KEYWORD_STR\s*,\s*"([^"]+)"')


def _region(text: str, header_re: re.Pattern) -> List[str]:
    """All brace regions following a header regex match."""
    out = []
    for m in header_re.finditer(text):
        try:
            b = text.index("{", m.end())
            e = _match_brace(text, b)
            out.append(text[b + 1:e])
        except ValueError:
            continue
    return out


def _parse_default_value(s: str):
    s = s.strip()
    if s in ("TRUE",):
        return 1
    if s in ("FALSE",):
        return 0
    try:
        f = float(s.replace("D", "E").replace("d", "e"))
        return int(f) if f.is_integer() and "e" not in s.lower() \
            and "." not in s else f
    except ValueError:
        return s.strip('"')


def parse_cfg_file(path: str,
                   max_version: int = MAX_CFG_VERSION) -> CfgLawSchema:
    """Parse one MAT cfg file into a :class:`CfgLawSchema` (attributes +
    defaults + the newest FORMAT block at or below ``max_version``)."""
    with open(path, "r", errors="replace") as fh:
        text = _strip_comments(fh.read())

    attributes: Dict[str, _Attr] = {}
    for body in _region(text, re.compile(r"ATTRIBUTES(?:\s*\(\s*COMMON\s*\))?")):
        for m in _ATTR_RE.finditer(body):
            name, kind, size_var, typ = m.groups()
            if kind == "SIZE":
                attributes[name] = _Attr(name, "INT")
            elif kind == "ARRAY":
                attributes[name] = _Attr(name, typ or "FLOAT",
                                         is_array=True,
                                         size_var=size_var or "")
            else:
                attributes[name] = _Attr(name, typ or "FLOAT")

    defaults: Dict[str, object] = {}
    for body in _region(text, re.compile(r"DEFAULTS(?:\s*\(\s*COMMON\s*\))?")):
        for m in _DEFAULT_RE.finditer(body):
            defaults[m.group(1)] = _parse_default_value(m.group(2))

    # newest FORMAT <= max_version
    best = None
    for m in _FORMAT_RE.finditer(text):
        ver = int(m.group(1))
        if ver > max_version:
            continue
        if best is None or ver > best[0]:
            try:
                b = text.index("{", m.end())
                e = _match_brace(text, b)
            except ValueError:
                continue
            best = (ver, b, e)
    if best is None:
        raise ValueError(f"{path}: no FORMAT(radioss<= {max_version}) block")
    ver, b, e = best
    statements, _ = _parse_statements(text, b + 1, e)

    # law names: HEADER literals + KEYWORD_STR assigns + filename number
    names: List[str] = []
    for m in re.finditer(r'HEADER\s*\(\s*"([^"]+)"', text):
        parts = [p for p in m.group(1).strip("/").split("/") if p]
        if not parts:
            continue
        if parts[0] == "MAT" and len(parts) > 1 and "%" not in parts[1]:
            if parts[1] not in names:
                names.append(parts[1])
        elif parts[0] in ("ALE", "EULER", "HEAT") and len(parts) > 1 \
                and parts[1] == "MAT":
            key = f"{parts[0]}/MAT"
            if key not in names:
                names.append(key)
    if re.search(r'ASSIGN\s*\(\s*KEYWORD_STR\s*,\s*"/MAT"', text):
        for m in _KEYSTR_RE.finditer(text):
            nm = m.group(1).strip("/").strip()
            if nm and nm != "MAT" and "%" not in nm and " " not in nm \
                    and "*" not in nm and nm not in names:
                names.append(nm)
    law_number = None
    for nm in names:
        m = re.fullmatch(r"LAW(\d+)", nm)
        if m:
            law_number = int(m.group(1))
            break
    if law_number is None:
        m = re.search(r"(?:matl?|mat_law|law)(\d+)",
                      os.path.basename(path).lower())
        if m:
            law_number = int(m.group(1))
            if f"LAW{law_number}" not in names:
                names.append(f"LAW{law_number}")
    return CfgLawSchema(path=path, attributes=attributes, defaults=defaults,
                        format_version=ver, statements=statements,
                        law_names=names, law_number=law_number)


# ============================================================================
# The catalogue: law name -> cfg file, resolved like the HM reader
# ============================================================================

class CfgCatalogue:
    """Scan the cfg version directories (newest <= ``max_version`` first;
    they are INCREMENTAL — each directory only carries the laws whose
    definition changed in that version) and map every law spelling to
    its cfg file.  Schemas are parsed lazily and cached."""

    def __init__(self, root: object = _RESOLVE,
                 max_version: int = MAX_CFG_VERSION):
        # ``root=_RESOLVE`` means "resolve it now"; ``root=None`` is the
        # real answer "there is no cfg tree" and must not be re-resolved.
        self.root = _find_cfg_root() if root is _RESOLVE else root
        self.max_version = max_version
        self._files: Dict[str, str] = {}       # law key -> cfg path
        self._by_basename: Dict[str, str] = {}  # 'law51_iflag_6' -> path
        self._schemas: Dict[str, CfgLawSchema] = {}
        self._scanned = False

    # -- directory scan (cheap regex pass, no full parse) --------------------

    def _scan(self) -> None:
        if self._scanned:
            return
        self._scanned = True
        if not self.root:
            return
        dirs = []
        for d in os.listdir(self.root):
            m = re.fullmatch(r"radioss(\d+)", d)
            if m and int(m.group(1)) <= self.max_version:
                mat_dir = os.path.join(self.root, d, "MAT")
                if os.path.isdir(mat_dir):
                    dirs.append((int(m.group(1)), mat_dir))
        dirs.sort(reverse=True)                # newest version first
        for _ver, mat_dir in dirs:
            entries = []
            for fn in os.listdir(mat_dir):
                if not fn.endswith(".cfg"):
                    continue
                low = fn.lower()
                if "unsupported" in low or "user" in low:
                    continue
                # canonical files first: variant files (law51_Iflag_2.cfg)
                # must not shadow the dispatching mat_law51.cfg
                score = 1 if "iflag" in low else 0
                entries.append((score, fn))
            for _score, fn in sorted(entries):
                path = os.path.join(mat_dir, fn)
                self._by_basename.setdefault(fn[:-4].lower(), path)
                try:
                    with open(path, "r", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    continue
                for key in self._law_keys(text, fn):
                    self._files.setdefault(key, path)

    @staticmethod
    def _law_keys(text: str, filename: str) -> List[str]:
        keys: List[str] = []

        def add(k: str) -> None:
            k = k.upper()
            if k and k not in keys:
                keys.append(k)

        for m in re.finditer(r'HEADER\s*\(\s*"([^"]+)"', text):
            parts = [p for p in m.group(1).strip("/").split("/") if p]
            if not parts:
                continue
            if parts[0] == "MAT" and len(parts) > 1 and "%" not in parts[1]:
                add(parts[1])
            elif parts[0] in ("ALE", "EULER", "HEAT") and len(parts) > 1 \
                    and parts[1] == "MAT":
                add(f"{parts[0]}/MAT")
        if re.search(r'ASSIGN\s*\(\s*KEYWORD_STR\s*,\s*"/MAT"', text):
            for m in _KEYSTR_RE.finditer(text):
                nm = m.group(1).strip("/").strip()
                if nm and nm != "MAT" and "%" not in nm and " " not in nm \
                        and "*" not in nm:
                    add(nm)
        m = re.search(r"(?:matl?|mat_law|law)(\d+)", filename.lower())
        if m:
            add(f"LAW{int(m.group(1))}")
        return keys

    # -- public API -----------------------------------------------------------

    _SYNONYMS: Dict[str, str] = {
        "HILL": "LAW32",
        "LAW32": "LAW32",
        "SOIL": "LAW10",
        "SOIL_CONC": "LAW10",
        "JWL": "LAW5",
        "HONEYCOMB": "LAW28",
        "BOLTZMAN": "LAW34",
        "VISC_MAXW": "LAW34",
        "BOLTZMANN": "LAW34",
        "LAW34": "LAW34",
        "VISC_TAB": "LAW38",
        "LAW38": "LAW38",
        "COMP_PLAS": "LAW25",
        "COMPSH": "LAW25",
        "TSAI_WU": "LAW25",
        "CRASURV": "LAW25",
        "COMPOSITE_PLAS": "LAW25",
        "LAW25": "LAW25",
        "CHANG": "LAW15",
        "PLAS_ANISO": "LAW15",
        "COMP_CHANG": "LAW15",
        "LAW15": "LAW15",
        "LAW12": "LAW12",
        "3D_COMP": "LAW12",
        "COMP_3D": "LAW12",
        "3PARBI": "LAW12",
        "RAGAB": "LAW12",
        "LAW14": "LAW14",
        "COMPSO": "LAW14",
        "COMP_SOL": "LAW14",
        "LAW69": "LAW69",
        "HYP_ELAS": "LAW69",
        "HYPERELASTIC": "LAW69",
        "LAW69_HYP_ELAS": "LAW69",
        "MAT_LAW69": "LAW69",
    }


    def canonical_law_name(self, law_name: str) -> str:
        """Map a law spelling to its canonical name if known in synonyms."""
        key = law_name.upper()
        return self._SYNONYMS.get(key, key)

    def schema(self, law_name: str) -> Optional[CfgLawSchema]:
        """The parsed schema for a law spelling ('FABRI', 'LAW19',
        'ALE/MAT' ...), or None when the catalogue has no cfg for it."""
        self._scan()
        key = law_name.upper()
        path = self._files.get(key)
        if path is None and key in self._SYNONYMS:
            key = self._SYNONYMS[key]
            path = self._files.get(key)
        if path is None:
            return None
        if path not in self._schemas:
            try:
                self._schemas[path] = parse_cfg_file(path, self.max_version)
            except (ValueError, OSError) as exc:
                # a malformed cfg must never take the reader down
                self._schemas[path] = None       # type: ignore[assignment]
                self._files[key] = None          # type: ignore[assignment]
                return None
        return self._schemas.get(path)

    def subobject_schema(self, name: str) -> Optional[CfgLawSchema]:
        """Schema of an inline SUBOBJECT (``/SUBOBJECT/LAW51_IFLAG_6`` ->
        ``law51_Iflag_6.cfg``), matched on the cfg file basename."""
        self._scan()
        path = self._by_basename.get(name.lower())
        if path is None:
            return None
        if path not in self._schemas:
            try:
                self._schemas[path] = parse_cfg_file(path, self.max_version)
            except (ValueError, OSError):
                self._schemas[path] = None       # type: ignore[assignment]
        return self._schemas.get(path)

    def known_laws(self) -> List[str]:
        self._scan()
        return sorted(k for k, v in self._files.items() if v)


_CATALOGUE: Optional[CfgCatalogue] = None
#: the cfg root ``_CATALOGUE`` was built from (``_RESOLVE`` = never built).
_CATALOGUE_ROOT: object = _RESOLVE


def catalogue() -> CfgCatalogue:
    """The process-wide lazy catalogue singleton, re-resolved when the cfg
    root moves.

    The root is keyed on every call rather than frozen at first use
    (Task P0.6 fix round 1): several test modules evaluate
    ``mat_reader.catalogue()`` in a module-level ``pytestmark`` — i.e. at
    *collection* time — so a frozen root would survive ``paths.reload()``
    and a test that re-points ``PYRADIOSS_HM_CFG`` would keep reading the
    tree that happened to be current when pytest started collecting.  The
    resolver memoises, so re-resolving costs one dict lookup on the happy
    path; the catalogue is rebuilt only when the root actually changed.
    """
    global _CATALOGUE, _CATALOGUE_ROOT
    root = _find_cfg_root()
    if _CATALOGUE is None or root != _CATALOGUE_ROOT:
        _CATALOGUE = CfgCatalogue(root)
        _CATALOGUE_ROOT = root
    return _CATALOGUE


# ============================================================================
# The FORMAT interpreter
# ============================================================================

_FMT_SPEC = re.compile(r"%(-?)(\d*)(lg|lf|le|lg|g|d|i|s|f|e)")

_INT_SPECS = ("d", "i")
_STR_SPECS = ("s",)


def _split_fmt(fmt: str) -> List[Tuple[str, ...]]:
    """Format string -> segments: ('lit', text) | ('spec', type, width)."""
    segs: List[Tuple[str, ...]] = []
    i = 0
    while i < len(fmt):
        if fmt[i] == "%":
            m = _FMT_SPEC.match(fmt, i)
            if m:
                width = int(m.group(2)) if m.group(2) else 0
                segs.append(("spec", m.group(3), width))
                i = m.end()
                continue
        j = fmt.find("%", i + 1)
        j = len(fmt) if j < 0 else j
        segs.append(("lit", fmt[i:j]))
        i = j
    return segs


class _Env(dict):
    """Expression environment: unknown identifiers read as 0 (the cfg
    convention — everything defaults to zero/empty)."""

    def __missing__(self, key):
        return 0


class _CfgInterpreter:
    """Execute a FORMAT program in IMPORT mode over a block's data cards.

    Supported statements (the complete MAT-family inventory, measured
    over the whole cfg tree): CARD, COMMENT, CARD_PREREAD, ASSIGN,
    HEADER, BLANK, SUBOBJECTS, FREE_CARD, CARD_LIST, FREE_CARD_LIST,
    CELL_LIST, FREE_CELL_LIST, if/else-if/else.

    Robustness rules (parse-clean philosophy — a material card must
    never take the whole deck down):

    * a CARD past the end of the block leaves every target at its
      default (real writers omit all-default trailing cards);
    * a field that fails numeric conversion reads as its default;
    * BLANK consumes one card IF the current card is whitespace-only
      (the deck_reader records dropped blank lines in
      ``KeywordBlock.blank_slots`` and ``_effective_cards`` reinserts
      them); a writer that omitted the blank line entirely loses
      nothing — BLANK then consumes nothing.
    """

    MAX_LIST = 1000                     # runaway-loop guard
    MAX_SUBOBJECT_DEPTH = 4

    def __init__(self, schema: CfgLawSchema, cards: List[Card],
                 header_parts: List[str], mat_id: int, depth: int = 0):
        self.schema = schema
        self.cards = cards
        self.header_parts = header_parts
        self.mat_id = mat_id
        self.depth = depth
        self.i = 0                      # card cursor
        self.v: Dict[str, object] = {}
        for name, attr in schema.attributes.items():
            if attr.is_array:
                self.v[name] = []
        for name, val in schema.defaults.items():
            if name in schema.attributes and schema.attributes[name].is_array:
                continue
            self.v[name] = val
        self.v["IO_FLAG"] = 1           # import mode

    # -- helpers ---------------------------------------------------------------

    def _attr(self, name: str) -> Optional[_Attr]:
        return self.schema.attributes.get(name)

    def _set(self, name: str, value) -> None:
        if name in ("_ID_", "_BLANK_"):
            return
        attr = self._attr(name)
        if attr is not None and attr.is_array:
            self.v.setdefault(name, [])
            if value is not None:
                self.v[name].append(value)      # type: ignore[union-attr]
            return
        if value is None:
            return                       # blank field -> keep default
        if attr is not None and not attr.is_array:
            if attr.type in ("INT", "BOOL", "FUNCT", "MULTIOBJECT", "MAT",
                             "PROP", "NODE", "SETS", "CURVE", "TABLE",
                             "SENSOR", "SUBOBJECT"):
                try:
                    value = int(float(str(value).replace("D", "E")
                                      .replace("d", "e"))) \
                        if not isinstance(value, str) or value.strip() else 0
                except (ValueError, TypeError):
                    pass
        self.v[name] = value

    def _eval(self, expr: str):
        py = expr.replace("&&", " and ").replace("||", " or ")
        py = re.sub(r"!(?!=)", " not ", py)
        py = re.sub(r"\bFALSE\b", "0", py)
        py = re.sub(r"\bTRUE\b", "1", py)
        env = _Env(self.v)
        env["_GET_NB_FREE_CARDS"] = lambda: max(0, len(self.cards) - self.i)
        try:
            return eval(py, {"__builtins__": {}}, env)
        except Exception:
            return 0

    def _convert(self, typ: str, raw: str):
        raw = raw.strip()
        if typ in _STR_SPECS:
            return raw
        if not raw:
            return None
        try:
            val = parse_fortran_float(raw)
        except (ValueError, TypeError):
            return None
        return int(val) if typ in _INT_SPECS else val

    def _read_line(self, fmt: str, args: List[str], line: str) -> None:
        segs = _split_fmt(fmt)
        pos = 0
        ai = 0
        tokens = line.split()

        # Detect free-format input: any fixed-width slice containing >1 whitespace-separated token
        is_free_format = False
        scan_pos = 0
        for seg in segs:
            if seg[0] == "lit":
                scan_pos += len(seg[1])
                continue
            _tag, _typ, width = seg
            if width:
                if len(line[scan_pos:scan_pos + width].split()) > 1:
                    is_free_format = True
                    break
                scan_pos += width

        if is_free_format and tokens:
            data_segs = [s for s in segs if s[0] != "lit"]
            non_blank_pairs = [(s[1], a) for s, a in zip(data_segs, args) if a != "_BLANK_"]
            if len(tokens) <= len(non_blank_pairs):
                for tok_idx, (typ, arg) in enumerate(non_blank_pairs):
                    raw = tokens[tok_idx] if tok_idx < len(tokens) else ""
                    self._set(arg, self._convert(typ, raw))
                return
            tok_idx = 0
            for seg in segs:
                if seg[0] == "lit":
                    continue
                _tag, typ, _w = seg
                if ai < len(args):
                    raw = tokens[tok_idx] if tok_idx < len(tokens) else ""
                    tok_idx += 1
                    self._set(args[ai], self._convert(typ, raw))
                    ai += 1
            return

        for seg in segs:
            if seg[0] == "lit":
                pos += len(seg[1])
                continue
            _tag, typ, width = seg
            if width:
                raw = line[pos:pos + width]
                pos += width
            else:                        # widthless spec: next token
                m = re.compile(r"\S+").search(line, pos)
                raw = m.group(0) if m else ""
                pos = m.end() if m else len(line)
            if ai < len(args):
                val = self._convert(typ, raw)
                self._set(args[ai], val)
                ai += 1

    # -- statement execution -----------------------------------------------------

    def run(self, stmts: Optional[list] = None) -> None:
        for stmt in (self.schema.statements if stmts is None else stmts):
            self._exec(stmt)

    def _exec(self, stmt) -> None:
        kind = stmt[0]
        if kind == "blank":
            # consume the blank card if it is physically present
            if self.i < len(self.cards) \
                    and not self.cards[self.i].raw.strip():
                self.i += 1
            return
        if kind == "if":
            _, branches, else_stmts = stmt
            for cond, body in branches:
                if self._eval(cond):
                    self.run(body)
                    return
            if else_stmts:
                self.run(else_stmts)
            return
        if kind == "loop":
            _, name, count_expr, body = stmt
            if name == "CARD_LIST":
                n = self._eval(count_expr)
                try:
                    n = int(n)
                except (TypeError, ValueError):
                    n = 0
                for _ in range(max(0, min(n, self.MAX_LIST))):
                    self.run(body)
            else:                        # FREE_CARD_LIST: until cards end
                count = 0
                while self.i < len(self.cards) and count < self.MAX_LIST:
                    before = self.i
                    self.run(body)
                    count += 1
                    if self.i == before:
                        break            # body consumed nothing — stop
                self._set(count_expr, count)
            return
        # ---- plain calls -----------------------------------------------
        _, name, args = stmt
        if name in ("COMMENT", "SEPARATOR"):
            return
        if name == "SUBOBJECTS":
            self._exec_subobjects(args)
        elif name == "ASSIGN":
            self._exec_assign(args)
        elif name == "CARD":
            self._exec_card(args, consume=True)
        elif name == "CARD_PREREAD":
            self._exec_card(args, consume=False)
        elif name == "FREE_CARD":
            self._exec_free_card(args)
        elif name in ("CELL_LIST", "FREE_CELL_LIST"):
            self._exec_cell_list(name, args)
        elif name == "HEADER":
            self._exec_header(args)
        # anything else (GUI leftovers) is ignored — parse-clean philosophy

    def _exec_subobjects(self, args: List[str]) -> None:
        """Inline SUBOBJECT expansion: LAW51's per-Iflag card groups are
        split into their own cfg files (``law51_Iflag_6.cfg`` etc.) and
        pulled in with ``SUBOBJECTS(attr, /SUBOBJECT/LAW51_IFLAG_6)`` —
        the cards continue in the SAME deck block, so the child schema
        runs on the shared cursor and its values merge back.  Subobjects
        that live in their OWN deck blocks (/SUBOBJECT/HEAT, /VISC ...)
        have no basename match here and are no-ops — their gate flags
        (Heat_Inp_opt ...) are never set on import anyway."""
        if len(args) < 2 or self.depth >= self.MAX_SUBOBJECT_DEPTH:
            return
        name = args[1].strip().strip('"')
        upper = name.upper()
        if not upper.startswith("/SUBOBJECT/"):
            return
        key = name[len("/SUBOBJECT/"):].strip("/").replace("/", "_")
        schema = catalogue().subobject_schema(key)
        if schema is None:
            return
        child = _CfgInterpreter(schema, self.cards, self.header_parts,
                                self.mat_id, depth=self.depth + 1)
        child.v.update(self.v)          # parent values visible to child
        child.i = self.i
        child.run()
        self.i = child.i
        self.v.update(child.v)

    def _exec_assign(self, args: List[str]) -> None:
        if len(args) < 2:
            return
        mode = args[2].strip().upper() if len(args) > 2 else ""
        if mode == "EXPORT":
            return                       # import mode: skip export assigns
        target = args[0].strip()
        expr = args[1].strip()
        if expr.startswith('"') and expr.endswith('"'):
            self._set(target, expr[1:-1])
        else:
            self._set(target, self._eval(expr))

    def _exec_card(self, args: List[str], consume: bool) -> None:
        if not args:
            return
        fmt = args[0].strip().strip('"')
        targets = [a.strip() for a in args[1:]]
        if self.i < len(self.cards):
            line = self.cards[self.i].raw
            if consume:
                self.i += 1
        else:
            line = ""                    # past the end: all defaults
        self._read_line(fmt, targets, line)

    def _exec_free_card(self, args: List[str]) -> None:
        # FREE_CARD(flag, "fmt", targets...) or FREE_CARD("fmt", targets...)
        if not args:
            return
        if args[0].strip().startswith('"'):
            flag, rest = None, args
        else:
            flag, rest = args[0].strip(), args[1:]
        present = self.i < len(self.cards)
        if flag:
            self._set(flag, 1 if present else 0)
        if present:
            self._exec_card(rest, consume=True)

    def _exec_cell_list(self, name: str, args: List[str]) -> None:
        """CELL_LIST(N, "fmt", attrs..., line_width): N rows of ``fmt``
        packed ``line_width`` characters per card (e.g. 5 x %20lg per
        100-char card).  FREE_CELL_LIST: rows until the cards run out."""
        if len(args) < 3:
            return
        count_expr = args[0].strip()
        fmt = args[1].strip().strip('"')
        line_width = 0
        tail = args[-1].strip()
        attrs = [a.strip() for a in args[2:]]
        if re.fullmatch(r"\d+", tail):
            line_width = int(tail)
            attrs = attrs[:-1]
        segs = _split_fmt(fmt)
        row_width = sum(s[2] if s[0] == "spec" else len(s[1]) for s in segs)
        per_card = max(1, (line_width or 100) // max(row_width, 1))
        if name == "CELL_LIST":
            n = self._eval(count_expr)
            try:
                n = int(n)
            except (TypeError, ValueError):
                n = 0
            n = max(0, min(n, self.MAX_LIST))
            read = 0
            while read < n:
                line = self.cards[self.i].raw if self.i < len(self.cards) \
                    else ""
                if self.i < len(self.cards):
                    self.i += 1
                for k in range(min(per_card, n - read)):
                    chunk = line[k * row_width:(k + 1) * row_width]
                    self._read_line(fmt, attrs, chunk)
                    read += 1
        else:                            # FREE_CELL_LIST
            read = 0
            while self.i < len(self.cards) and read < self.MAX_LIST:
                line = self.cards[self.i].raw
                self.i += 1
                for k in range(per_card):
                    chunk = line[k * row_width:(k + 1) * row_width]
                    if not chunk.strip():
                        continue
                    self._read_line(fmt, attrs, chunk)
                    read += 1
            self._set(count_expr, read)

    def _exec_header(self, args: List[str]) -> None:
        """IMPORT-side HEADER: match the block's keyword path against the
        pattern, assigning %s/%d captures (``HEADER("/MAT/%3s",LAW_NO)``
        reads the first 3 characters of the law component — that is how
        the cfgs distinguish the LAW<n> spelling from the name)."""
        if not args:
            return
        fmt = args[0].strip().strip('"')
        targets = [a.strip() for a in args[1:]]
        pat_parts = [p for p in fmt.strip("/").split("/") if p]
        ai = 0
        for k, pel in enumerate(pat_parts):
            part = self.header_parts[k] if k < len(self.header_parts) else ""
            m = re.match(r"([^%]*)%(-?)(\d*)([a-z]+)", pel)
            if not m:
                continue                 # literal component
            lit, _minus, width, typ = m.groups()
            val = part[len(lit):] if part.startswith(lit) else part
            if width:
                val = val[:int(width)]
            if ai < len(targets):
                target = targets[ai]
                ai += 1
                if target == "_ID_":
                    continue
                if typ in _INT_SPECS:
                    try:
                        self._set(target, int(val))
                    except ValueError:
                        self._set(target, 0)
                else:
                    self._set(target, val)


# ============================================================================
# Block-level reading
# ============================================================================

#: attribute names that are bookkeeping, not material parameters
_NOISE_ATTRS = {
    "IO_FLAG", "DUMMY", "KEYWORD_STR", "NUM_COMMENTS", "COMMENTS",
    "CommentEnumField", "TITLE", "LAW_NO", "TYPE_NO", "SUBTYPE",
    "Mat_Name_OR_LawNo", "_BLANK_",
}

#: params whose value names the initial density, in preference order
_DENSITY_KEYS = ("MAT_RHO", "RHO_I", "Rho", "MAT_RHO0", "RHO")


def _is_numeric_card(card: Card) -> bool:
    toks = card.tokens()
    if not toks:
        return False
    for t in toks:
        try:
            parse_fortran_float(t)
        except ValueError:
            return False
    return True


def _effective_cards(block: KeywordBlock) -> List[Card]:
    """The block's TRUE card stream for a fixed-format layout: re-insert
    the whitespace-only lines the deck_reader dropped (they are blank
    cards — every field at its default — and real decks lean on them,
    e.g. LAW37's blank Psh card).  Trailing blanks are stripped so the
    FREE_CARD/FREE_CELL_LIST statements do not read phantom rows."""
    fixed = getattr(block, "fixed_cards", None)
    if callable(fixed):
        cards = list(fixed())
    else:                                # hand-built minimal blocks
        cards = list(block.cards)
        offset = 0
        for pos in getattr(block, "blank_slots", []):
            cards.insert(pos + offset, Card(raw="", source=block.source))
            offset += 1
    while cards and not cards[-1].raw.strip():
        cards.pop()
    return cards


def _title_and_data(block: KeywordBlock) -> Tuple[str, List[Card]]:
    cards = _effective_cards(block)
    if cards and not _is_numeric_card(cards[0]):
        return cards[0].raw.strip(), cards[1:]
    return "", cards


def _mat_id_and_unit(block: KeywordBlock) -> Tuple[Optional[int],
                                                   Optional[int]]:
    """/MAT/<LAW>/<id> or /MAT/<LAW>/<id>/<unit_id> — with two trailing
    integers the FIRST is the material id (the Radioss unit-per-option
    convention); the deck_reader only strips the last one."""
    ints = []
    for p in reversed(block.parts):
        try:
            ints.append(int(p))
        except ValueError:
            break
    if len(ints) >= 2:
        return ints[1], ints[0]          # ints is reversed: [-1], [-2]
    if len(ints) == 1:
        return ints[0], None
    return None, None


def parse_generic_mat(block: KeywordBlock,
                      log: Optional[MessageLog] = None,
                      cat: Optional[CfgCatalogue] = None
                      ) -> Optional[GenericMaterialRecord]:
    """Parse any /MAT/<LAW> block into a :class:`GenericMaterialRecord`
    using the cfg schema (or the heuristic fallback).  Returns None only
    when the block has no usable law name / id."""
    if len(block.parts) < 2:
        if log:
            log.error("/MAT block without a law name", block.source)
        return None
    law_name = block.parts[1].upper()
    mat_id, unit_id = _mat_id_and_unit(block)
    if mat_id is None:
        if log:
            log.error(f"/MAT/{law_name}: no material id in the keyword",
                      block.source)
        return None
    subtype = ""
    if len(block.parts) > 2:
        try:
            int(block.parts[2])
        except ValueError:
            subtype = block.parts[2].upper()
    title, cards = _title_and_data(block)
    cat = cat or catalogue()
    schema = cat.schema(law_name)

    if schema is None:
        # ---- heuristic fallback: density from the first data card ------
        density = 0.0
        for c in cards:
            try:
                density = float(c.tokens()[0].replace("D", "E")
                                .replace("d", "e"))
                break
            except (ValueError, IndexError):
                continue
        m = re.fullmatch(r"LAW(\d+)", law_name)
        rec = GenericMaterialRecord(
            law_name=law_name, law_number=int(m.group(1)) if m else None,
            id=mat_id, title=title,
            params={"raw_cards": [c.raw for c in cards]},
            density=density, subtype=subtype, unit_id=unit_id,
            raw_cards=[c.raw for c in cards])
        if log:
            log.warning(f"/MAT/{law_name}/{mat_id}: no cfg schema found "
                        f"(hm_cfg_files not installed or unknown law) — "
                        f"stored with heuristic density {density:g}"
                        + (f"\n{_CFG_SEARCH_FAILED}"
                           if _CFG_SEARCH_FAILED else ""),
                        block.source)
        return rec

    interp = _CfgInterpreter(schema, cards,
                             header_parts=list(block.parts), mat_id=mat_id)
    try:
        interp.run()
    except Exception as exc:             # never take the deck down
        if log:
            log.warning(f"/MAT/{law_name}/{mat_id}: cfg-driven parse "
                        f"stopped early ({exc}) — fields read so far are "
                        f"kept", block.source)

    params = {k: v for k, v in interp.v.items() if k not in _NOISE_ATTRS}
    density = 0.0
    for key in _DENSITY_KEYS:
        val = params.get(key)
        if isinstance(val, (int, float)) and val != 0:
            density = float(val)
            break
    if density == 0.0:
        # laws without a MAT_RHO (multi-phase: LAW37/LAW51...) — take the
        # first positive *_RHO_* parameter (scalar or per-phase array
        # entry) as the mass-init estimate
        for k, v in params.items():
            ku = k.upper()
            if "RHO" not in ku or any(x in ku for x in
                                      ("REF", "OPTION", "CP", "CV")):
                continue
            cands = v if isinstance(v, list) else [v]
            for c in cands:
                if isinstance(c, (int, float)) and c > 0:
                    density = float(c)
                    break
            if density > 0:
                break
    canonical = schema.law_names[0] if schema.law_names else law_name
    return GenericMaterialRecord(
        law_name=law_name, law_number=schema.law_number, id=mat_id,
        title=title, params=params, density=density, subtype=subtype,
        unit_id=unit_id, cfg_file=schema.path,
        raw_cards=[c.raw for c in cards])


def _registry_lookup(rec: GenericMaterialRecord
                     ) -> Optional[Callable[[GenericMaterialRecord], object]]:
    """Try the header spelling, subtype-qualified spelling, the LAW<n>
    alias and every cfg spelling."""
    # the physics builders register themselves when the materials package
    # loads — make sure it has (lazy: mat_reader is imported by the deck
    # reader long before any Engine module pulls in the laws)
    from .. import materials  # noqa: F401  (registration side effect)
    keys = [rec.law_name]
    if rec.subtype:
        keys.insert(0, f"{rec.law_name}/{rec.subtype}")
    if rec.law_number is not None:
        keys.append(f"LAW{rec.law_number}")
    for key in keys:
        fn = MAT_PHYSICS_REGISTRY.get(key)
        if fn is not None:
            return fn
    return None


def _safe_elastic(params: Dict[str, object]) -> Tuple[float, float]:
    """Fallback E/nu for Starter-side stiffness ESTIMATES (time-step
    factors, contact stiffness previews).  Physically meaningless for an
    inactive material — the Engine refuses to run it anyway."""
    e = 0.0
    for key in ("MAT_E", "MAT_E0", "MAT_EA", "E", "LSD_E"):
        val = params.get(key)
        if isinstance(val, (int, float)) and val > 0:
            e = float(val)
            break
    nu = 0.3
    for key in ("MAT_NU", "NU", "LSD_NU", "MAT_PR"):
        val = params.get(key)
        if isinstance(val, (int, float)) and 0.0 <= float(val) < 0.5:
            nu = float(val)
            break
    return (e if e > 0 else 1.0), min(max(nu, 0.0), 0.495)


def make_inactive_material(rec: GenericMaterialRecord) -> InactiveMaterial:
    """Wrap a record as an :class:`InactiveMaterial` ready for
    ``model.materials`` (density available -> mass init works)."""
    params = dict(rec.params)
    e, nu = _safe_elastic(params)
    params.setdefault("E", e)
    params.setdefault("nu", nu)
    return InactiveMaterial(
        id=rec.id, law=rec.law_number if rec.law_number is not None else -1,
        rho0=rec.density, title=rec.title, params=params,
        law_name=rec.law_name, record=rec)


def read_generic_mat(block: KeywordBlock, model, log: MessageLog) -> None:
    """The /MAT dispatch hook for every law WITHOUT a dedicated reader
    (starter_keywords.read_mat falls through to this): parse via the cfg
    schema, then either build the registered physics material or store
    an :class:`InactiveMaterial` (parse-clean, not simulatable)."""
    rec = parse_generic_mat(block, log)
    if rec is None:
        return
    builder = _registry_lookup(rec)
    if builder is not None:
        try:
            mat = builder(rec)
        except Exception as exc:
            log.error(f"/MAT/{rec.law_name}/{rec.id}: registered physics "
                      f"constructor failed: {exc}", block.source)
            return
        mat.record = rec
        model.materials[rec.id] = mat
        return
    model.materials[rec.id] = make_inactive_material(rec)
    lawno = f" (LAW{rec.law_number})" if rec.law_number is not None else ""
    log.info(f"     /MAT/{rec.law_name}/{rec.id}{lawno} "
             f"'{rec.title}': parsed, physics not implemented (M37)")


# ============================================================================
# /ALE/MAT, /EULER/MAT, /HEAT/MAT — parse-only notes (M37)
# ============================================================================

def read_mat_note(kind: str, block: KeywordBlock, model,
                  log: MessageLog) -> None:
    """``/ALE/MAT/mat_ID``, ``/EULER/MAT/mat_ID``, ``/HEAT/MAT/mat_ID``:
    modifiers of an existing material (ALE/Euler formulation flags,
    thermal data).  Parsed via their cfg (mat_ALE.cfg / mat_EULER.cfg /
    mat_HEAT.cfg) and stored as a NOTE on the model
    (``model.raw_mat_notes``; attached to the material's params by
    ``resolve_materials``) — accepted, remembered, no physics (M37).
    These blocks carry no title card."""
    mat_id = block.user_id
    if mat_id is None:
        mat_id, _unit = _mat_id_and_unit(block)
    if mat_id is None:
        log.error(f"/{kind}: no material id in the keyword", block.source)
        return
    schema = catalogue().schema(kind)
    params: Dict[str, object] = {}
    if schema is not None:
        interp = _CfgInterpreter(schema, _effective_cards(block),
                                 header_parts=list(block.parts),
                                 mat_id=mat_id)
        try:
            interp.run()
        except Exception:
            log.warning(f"     /{kind}/{mat_id}: CFG interpreter failed, "
                        f"partial params may be incomplete",
                        block.source)
        params = {k: v for k, v in interp.v.items()
                  if k not in _NOISE_ATTRS}
    else:
        params = {"raw_cards": [c.raw for c in block.cards]}
    notes = getattr(model, "raw_mat_notes", None)
    if notes is None:
        notes = model.raw_mat_notes = []
    notes.append((kind, mat_id, params, block.source))
    log.info(f"     /{kind}/{mat_id}: parsed as a note — "
             f"formulation not implemented (M37)")


# Alias for callers importing read_mat from mat_reader
read_mat = read_generic_mat
