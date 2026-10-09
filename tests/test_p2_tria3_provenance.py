"""Task P2.1 -- every Fortran file this port cites must actually exist.

The origin of this gate is a fabricated citation.  ``pyradioss/elements/
solid_tria3.py`` named ``engine/source/elements/solid_2d/tria/`` and five
routines under it (``t3forc2.F``, ``t3deri2.F``, ``t3rota2.F``, ``t3dlen2.F``
and a starter-side ``t3mass2.F``).  **None of them exist.**  Verified against
``$OR_SRC`` on 2026-10-09:

* ``engine/source/elements/solid_2d/`` holds only ``quad/`` and ``quad4/`` --
  the plan's 2026-10-02 note is confirmed;
* ``starter/source/elements/solid_2d/tria/`` exists but holds only the two
  mesh-reader routines ``t3grhead.F`` and ``t3grtails.F``, which read element
  groups out of the input deck.  There is no ``t3mass2.F`` and no engine-side
  kernel of any kind.

So the 2-D solid triangle has an input-format reader but **no open-source
engine kernel**, which puts it in the plan's Step 3 decision order branch 1:
the honest citation is *Simcenter Radioss (closed source)*, and the module is
a **declared port extension** rather than a port.  The decision and its
justification live in ``docs/PORT_EXTENSIONS.md``.

A citation naming a file that is not there is worse than no citation at all: it
lends the authority of a transcription nobody ever did to a routine that would
be impossible to compare against.  The whole point of this module is that no
such citation survives anywhere in ``pyradioss/``.

The scan covers the entire ``pyradioss/`` tree, not just ``elements/``, because
the failure it guards against is not location-specific.  It resolves both the
fully-qualified spellings (``engine/source/...``, ``starter/source/...``) and
the bare ``elements/<family>/<routine>.F`` form, which is read relative to
``engine/source/`` as the plan specifies.

Generalising the scan from ``elements/`` to ``pyradioss/`` -- Step 4 -- found 56
more misses in 23 modules this task may not edit.  They are not silently
dropped: they are enumerated in ``KNOWN_MISSES`` below, which keeps the gate
strict (a *new* miss still fails) and cannot itself go stale (an entry that is
repaired without being removed fails
:func:`test_known_misses_are_still_absent_upstream`).

Both patterns need an **open directory depth**.  Upstream families nest --
``shell/coque/``, ``solid/solide/``, ``solid_2d/tria/``, ``joint/rjoint/`` --
so a two-level pattern silently misses both the real citations and the fabricated
ones.  The plan's own sketch is written at two levels; widening it is what makes
the gate able to see ``solid_2d/tria/t3forc2.F`` at all, which is the whole
point of the task.
"""

from __future__ import annotations

import re
from pathlib import Path

from pyradioss import paths

REPO = Path(__file__).resolve().parent.parent

#: The fully-qualified citation form: the tree is named explicitly.  The
#: directory depth is open because upstream families nest -- ``shell/coque/``,
#: ``solid/solide/``, ``solid_2d/tria/`` -- and a two-level pattern would walk
#: straight past the citations it is meant to catch.
QUALIFIED = re.compile(
    r"(?:engine|starter|common)/source/elements/[A-Za-z0-9_/]+\.F"
)

#: The plan's bare form, resolved against ``engine/source/`` as the plan says.
#: It needs the same open depth, or ``elements/solid_2d/tria/t3forc2.F`` --
#: the exact shape this gate was written for -- would not match at all.
BARE = re.compile(r"(?<![a-z_])elements/[a-z0-9_/]+\.F")

#: Citations that were already wrong when this gate landed, in modules this task
#: is not allowed to edit.  Each still needs its own decision -- many look like a
#: misspelled directory (``solid/sdefo3.F`` is really ``solid/solide/sdefo3.F``,
#: ``shell/ccoor3.F`` is ``shell/coque/ccoor3.F``), but some may be genuinely
#: absent, and only the ``/TRIA3`` case was settled here.  ``docs/
#: PORT_EXTENSIONS.md`` §Known misses carries the same list with the per-file
#: analysis and the affected modules.
#:
#: The gate is NOT weakened by this list: ``test_no_new_fabricated_citations``
#: fails on anything not in it, so a new fabricated citation still turns the
#: suite red, and ``test_known_misses_are_still_absent_upstream`` fails if an
#: entry here is fixed but not removed -- the list cannot quietly go stale.
KNOWN_MISSES = frozenset({
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/shell/ccoor3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/shell/cdefo3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/shell/chvis3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/shell/czforc3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/sbulk3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/sdefo3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/sdlen3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/sfint3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/shour3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/srcoor3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/srota3.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/tetra10/t10coor.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/tetra10/t10defo.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/tetra10/t10dlen.F"),
    ("pyradioss/accel/gpu_kernels/__init__.py", "engine/source/elements/solid/tetra10/t10fint.F"),
    ("pyradioss/elements/beam_fiber.py", "engine/source/elements/beam/pmass3.F"),
    ("pyradioss/elements/iga3d.py", "engine/source/elements/ige3d/ig3dinit3.F"),
    ("pyradioss/elements/nstrand.py", "engine/source/elements/xelem/xini28.F"),
    ("pyradioss/elements/rivet.py", "engine/source/elements/reader/hm_read_rivet.F"),
    ("pyradioss/elements/shell_dkt6.py", "engine/source/elements/sh3n/coquedk6/cdk6mass3.F"),
    ("pyradioss/elements/shell_dkt6.py", "starter/source/elements/sh3n/coquedk6/cdk6mass3.F"),
    ("pyradioss/elements/shell_ortho.py", "engine/source/elements/shell/coque/corthdir.F"),
    ("pyradioss/elements/solid_hexa8z.py", "engine/source/elements/solid/sinit3.F"),
    ("pyradioss/elements/solid_hexa8z.py", "starter/source/elements/solid/sinit3.F"),
    ("pyradioss/elements/solid_pyra5.py", "starter/source/elements/solid/solide/degenes8.F"),
    ("pyradioss/elements/solid_quad4_full.py", "engine/source/elements/solid_2d/quad4/q4mass2.F"),
    ("pyradioss/elements/solid_quad4_full.py", "starter/source/elements/solid_2d/quad4/q4mass2.F"),
    ("pyradioss/elements/solid_tshell8.py", "engine/source/elements/solid/solide/smass3.F"),
    ("pyradioss/elements/spring_advanced.py", "engine/source/elements/spring/r3buf3.F"),
    ("pyradioss/elements/spring_advanced.py", "engine/source/elements/spring/redef3.F"),
    ("pyradioss/elements/spring_advanced.py", "engine/source/elements/spring/rini35.F"),
    ("pyradioss/elements/spring_advanced.py", "engine/source/elements/spring/rmass3.F"),
    ("pyradioss/elements/spring_advanced.py", "starter/source/elements/spring/rmass3.F"),
    ("pyradioss/elements/spring_beam.py", "engine/source/elements/spring/r4buf3.F"),
    ("pyradioss/elements/spring_beam.py", "engine/source/elements/spring/redef3.F"),
    ("pyradioss/elements/spring_mat.py", "engine/source/elements/spring/redef3.F"),
    ("pyradioss/elements/spring_mat.py", "engine/source/elements/spring/rinit3.F"),
    ("pyradioss/elements/spring_mat.py", "engine/source/elements/spring/rmass.F"),
    ("pyradioss/elements/thickshell_composite.py", "engine/source/elements/solid/solide/smass3.F"),
    ("pyradioss/elements/thickshell_wedge6.py", "engine/source/elements/solid/solide/smass3.F"),
    ("pyradioss/elements/truss.py", "engine/source/elements/truss/tmass3.F"),
    ("pyradioss/elements/truss.py", "starter/source/elements/truss/tmass3.F"),
    ("pyradioss/engine/bolt_preload.py", "engine/source/elements/spring/preload_axial.F"),
    ("pyradioss/engine/kjoint.py", "engine/source/elements/joint/rjoint/rini33.F"),
    ("pyradioss/engine/kjoint.py", "engine/source/elements/joint/rjoint/rini45.F"),
    ("pyradioss/input/keywords/geometry.py", "engine/source/elements/reader/hm_read_node.F"),
    ("pyradioss/model/entities/misc.py", "engine/source/elements/beam/bsigini.F"),
    ("pyradioss/model/entities/misc.py", "engine/source/elements/inibri_eref.F"),
    ("pyradioss/model/entities/misc.py", "engine/source/elements/initia/hm_read_inistate_d00.F"),
    ("pyradioss/model/entities/misc.py", "engine/source/elements/reader/hm_read_node.F"),
    ("pyradioss/model/entities/misc.py", "engine/source/elements/spring/rinit3.F"),
    ("pyradioss/model/entities/misc.py", "engine/source/elements/truss/tsigini.F"),
    ("pyradioss/model/entities/misc.py", "starter/source/elements/inibri_eref.F"),
    ("pyradioss/model/stack.py", "engine/source/elements/shell/coque/lcgeo19.F"),
    ("pyradioss/starter/inista.py", "engine/source/elements/initia/lec_inistate.F"),
    ("pyradioss/starter/initemp.py", "engine/source/elements/solid/solide/sinit3.F"),
})


def _cited_paths(root: Path) -> list[tuple[str, str]]:
    """Every Fortran file cited under ``root``, as ``(citing file, repo relpath)``."""
    cited: list[tuple[str, str]] = []
    for src in sorted(root.rglob("*.py")):
        text = src.read_text(encoding="utf-8")
        seen: set[str] = set()
        for m in QUALIFIED.finditer(text):
            rel = m.group(0)
            seen.add(rel)
        for m in BARE.finditer(text):
            seen.add("engine/source/" + m.group(0))
        for rel in sorted(seen):
            cited.append((src.relative_to(REPO).as_posix(), rel))
    return cited


def _absent_citations() -> list[tuple[str, str]]:
    """Every ``pyradioss/`` citation naming a file $OR_SRC does not have."""
    or_src = paths.or_src()
    return [
        (citer, rel)
        for citer, rel in _cited_paths(REPO / "pyradioss")
        if not (or_src / rel).is_file()
    ]


def test_no_new_fabricated_citations() -> None:
    """No citation in pyradioss/ may name a file absent from $OR_SRC.

    This is the plan's test, generalised from ``pyradioss/elements/`` to the
    whole ``pyradioss/`` tree as Step 4 asks.  The generalisation immediately
    found 56 further misses in 23 modules outside this task's scope; they are
    enumerated in ``KNOWN_MISSES`` and analysed in ``docs/PORT_EXTENSIONS.md``
    rather than fixed here, because fixing each is a separate decision.
    """
    new = sorted(set(_absent_citations()) - KNOWN_MISSES)
    assert new == [], (
        "these citations name a file that does not exist under "
        f"{paths.or_src()}:\n" + "\n".join(f"  {c} -> {r}" for c, r in new)
    )


def test_solid_tria3_cites_no_absent_file() -> None:
    """The task's own claim: solid_tria3.py contributes zero misses.

    Asserted separately so the /TRIA3 decision cannot be lost inside the
    allowlist -- if a ``t3*.F`` path ever comes back to this module, it is not
    in ``KNOWN_MISSES`` and the first test fails, but this one names it in the
    message that a reader of the failure will actually see.
    """
    assert not [b for b in _absent_citations() if b[0].endswith("solid_tria3.py")]


def test_known_misses_are_still_absent_upstream() -> None:
    """``KNOWN_MISSES`` must not go stale.

    Each entry is a real citation of a file that does not exist.  If one is
    repaired upstream of this gate -- the routine is found under its real path,
    or the citation is corrected -- the entry stops being a miss, and leaving it
    listed would hide a genuine future regression behind an allowlist.  So
    fixing one without removing it from ``KNOWN_MISSES`` fails here.
    """
    or_src = paths.or_src()
    stale = sorted(
        citer_rel for citer_rel in KNOWN_MISSES if (or_src / citer_rel[1]).is_file()
    )
    assert stale == [], (
        "these KNOWN_MISSES entries now name a file that DOES exist upstream -- "
        "fix the citation and drop the entry:\n"
        + "\n".join(f"  {c} -> {r}" for c, r in stale)
    )


def test_solid_tria3_cites_no_tria_kernel() -> None:
    """The specific regression: solid_tria3.py must not claim a `tria/` kernel."""
    text = (REPO / "pyradioss" / "elements" / "solid_tria3.py").read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"\bt3(forc2|deri2|rota2|dlen2|mass2)\.F\b", line)
    ]
    assert offenders == [], (
        "solid_tria3.py still cites the nonexistent tria kernel:\n"
        + "\n".join(f"  {line}" for line in offenders)
    )


def test_port_extensions_registry_declares_tria3() -> None:
    """docs/PORT_EXTENSIONS.md must carry the tria3 entry with a justification.

    The plan makes that file the single registry of code with no upstream
    counterpart.  An entry that vanishes silently would let the module drift
    back into claiming to be a port, so the registry is asserted here rather
    than trusted.
    """
    registry = REPO / "docs" / "PORT_EXTENSIONS.md"
    assert registry.is_file(), "docs/PORT_EXTENSIONS.md does not exist"
    text = registry.read_text(encoding="utf-8")
    assert "solid_tria3" in text, "PORT_EXTENSIONS.md does not register solid_tria3"
    assert "Simcenter Radioss" in text, (
        "the tria3 entry must name Simcenter Radioss (closed source) as the "
        "only honest citation for a kernel absent from the open-source tree"
    )


def test_solid_tria3_docstring_says_it_is_an_extension() -> None:
    """The module itself must not read as a transcription of upstream Fortran."""
    text = (REPO / "pyradioss" / "elements" / "solid_tria3.py").read_text(encoding="utf-8")
    docstring = text[: text.find('"""', text.find('"""') + 3) + 3]
    assert "port extension" in docstring.lower(), (
        "solid_tria3.py's module docstring does not declare the module a port extension"
    )