"""P4.7 — the rivet kernel module holds physics and nothing else.

Upstream split (fidelity governs):

* physics: ``engine/source/elements/rivet/rivet1.F`` (RIVET1 — the kernel)
* reader:  ``starter/source/properties/rivet/hm_read_prop05.F`` (GEO array)
* reader:  ``starter/source/elements/reader/hm_read_rivet.F`` (elements)

So ``pyradioss/elements/rivet.py`` must contain no reader code: no card
parsing, no ``HM_GET``-style field extraction, not even a citation of the
reader routines — that provenance lives on the reader functions in
``pyradioss/input/keywords/properties.py``. The kernel's public surface is
the RIVET1 physics API: the ``rivet_step`` entry, the two IMOD formulation
branches, the elliptic failure-surface projection, and the Rivet/RivetProp
element buffer.
"""

from pathlib import Path

import pyradioss.elements.rivet as rivet

RIVET_SRC = Path(__file__).resolve().parents[1] / "pyradioss" / "elements" / "rivet.py"


def test_rivet_module_has_no_reader_code():
    src = RIVET_SRC.read_text()
    assert "HM_GET" not in src and "hm_read" not in src
    assert "KeywordBlock" not in src and "parse_prop" not in src
    # The kernel's public surface is the RIVET1 physics API and nothing else.
    assert callable(rivet.rivet_step)
    assert callable(rivet._rivet_rigid_body)
    assert callable(rivet._rivet_rigid_link)
    assert callable(rivet._apply_failure_surface)
    assert hasattr(rivet.Rivet, "off") and hasattr(rivet.RivetProp, "fn_max")


def test_rivet_reader_owns_the_geo_contract():
    """The GEO(1..5) storage contract moved to the reader side."""
    from pyradioss.input.keywords import properties

    assert "hm_read_prop05" in (properties.read_prop_type5.__doc__ or "")
