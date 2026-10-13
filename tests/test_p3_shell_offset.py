"""P3.8 — shell offset / warping-membrane initialisation.

THE UPSTREAM CONTRACT (read from the Fortran, not assumed)
----------------------------------------------------------
``engine/source/elements/shell/shell_offset_wm_ini.F90`` (109 lines) computes
one scalar per through-thickness integration point::

    wmi = 0; if (npt > 1):
        fac = 1/(npt-1)
        lobatto = (1/2 - fac/3)*fac   if |Z01(ipt,npt)| == 1/2 else fac
        pos = Z01(ipt,npt) + zoffset
        wmi = pos*lobatto

Its only caller is ``engine/source/materials/mat_share/mulawc.F90:778``
(verified present in the open tree)::

    if (zshift == zero):  wmc = wm(ipt,npt)            ! coqini_wm table
    else:                call shell_offset_wm_ini(ipt,nptt,zshift,wm_zshift)

with ``zshift = GEO(199, pid)`` (``mulawc.F90:598``; ``GEO(199)`` is set by
the ``hm_read_prop*.F`` readers, e.g. ``hm_read_prop01.F:337``).  ``wmc``
accumulates the through-thickness stress MOMENTS
(``mulawc.F90:2662-2664``: ``mom += wmc*sig``), so the offset shifts the
moment weights while the membrane (lobatto) factor carries no ``zoffset``
at all.  The plan's sketch for the second test below has the two backwards;
per ``AGENTS.md`` the Fortran wins and the corrected direction is pinned
here (cf. P3.1's note on the plan's kind-1 identity).

THREE PLAN NOTES THAT DID NOT SURVIVE CONTACT WITH THE TREE
-----------------------------------------------------------
* ``engine/source/interfaces/shell_offset/`` is NOT empty: it holds
  ``inter_offset_dim.F90``, ``inter_offset_ini.F90`` and ``offset_nproj.F90``
  (contact offset treatment, called by ``resol_init``).  That directory is a
  separate concern and is explicitly NOT claimed by this task.
* ``common_source/modules/elements/element_mod.F90`` contains NO offset
  field (zero hits for ``offset``/``zshift``/``wmc``): the offset lives in
  ``GEO(199,*)``, not in the element module.
* The Starter-side ``starter/source/elements/shell/shell_offset/``
  (``shell_offset_ini.F90``, called by ``lectur``) is surveyed but NOT
  ported here: its trigger (``/DEF_SHELL IOFFSET``) has no field in the
  port's ``read_def_shell``, so a port of it would be unreachable dead
  code.  ``initialise`` below is the group-level entry point the Starter's
  element init (and the engine material path) consumes.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements.shell_basis import WM1, Z01
from pyradioss.elements.shell_offset import (
    initialise,
    membrane_weights,
    reference_positions,
    wm_weight,
    wm_weights,
)
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                               initialize_elements_and_mass,
                                               resolve_node_groups,
                                               resolve_surfaces)

# a 2x2 patch of flat BT quads: 4 elements x nip=3 -> the plan's (4, 3)
_NODES = (
    "1 -8.0 -7.6 0.0\n2 0.0 -7.6 0.0\n3 8.0 -7.6 0.0\n"
    "4 -8.0 0.0 0.0\n5 0.0 0.0 0.0\n6 8.0 0.0 0.0\n"
    "7 -8.0 7.6 0.0\n8 0.0 7.6 0.0\n9 8.0 7.6 0.0\n"
)
_SHELLS = ("1 1 2 5 4\n2 2 3 6 5\n3 4 5 8 7\n4 5 6 9 8\n")
_MAT = "/MAT/LAW1/1\nm\n0.01\n1000. 0.\n"


def _patch(tmp_path, prop="SHELL"):
    """Four flat BT shells through the real Starter pipeline.

    ``prop="SHELL"`` is the plain ``/PROP/SHELL`` deck (no offset keyword);
    ``prop="TYPE51"`` swaps in ``/PROP/TYPE51`` with ``ZSHIFT = 0.1`` —
    a real reader-supported offset (``prop_reader.prop_type_ok`` accepts
    TYPE51 on shells), not an injected param.
    """
    if prop == "SHELL":
        prop_block = "/PROP/SHELL/1\nsh\n0 0 0 0\n0 0 0 0 0\n3 0 20.0\n"
    elif prop == "TYPE51":
        prop_block = "/PROP/TYPE51/1\nsh\n0 0 0 0 0.001 0.1\n"
    else:  # pragma: no cover - defensive, the helper owns its inputs
        raise ValueError(f"unknown prop block {prop!r}")
    deck = ("/BEGIN\np38 offset patch\n/NODE\n" + _NODES + "/SHELL/1\n"
            + _SHELLS + "/PART/1\np\n1 1\n" + _MAT + prop_block + "/END\n")
    f = tmp_path / "P38_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_starter_deck(read_deck(str(f)), model, log)
        build_element_groups(model, log)
        resolve_node_groups(model, log)
        resolve_surfaces(model, log)
        initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    return model


# ---------------------------------------------------------------------------
# the plan's named assertions
# ---------------------------------------------------------------------------

def test_offset_initialisation_is_zero_without_the_keyword(tmp_path):
    """/PROP/SHELL has no ZSHIFT field, so the offset branch of
    ``mulawc.F90:774-779`` never fires and ``initialise`` is exactly the
    plan's ``zeros((4, 3))``: 4 elements x nip 3."""
    g = _patch(tmp_path, prop="SHELL").shells
    assert np.array_equal(initialise(g, None), np.zeros((4, 3)))


def test_offset_shifts_the_moment_weights_not_the_membrane_weights(tmp_path):
    """The plan names this ``..._membrane_stiffness_not_the_moment``; the
    Fortran has it backwards (``mulawc.F90:778-779`` shifts the ``wmc``
    MOMENT weights, ``mulawc.F90:2662-2664``, while the lobatto factor
    takes no ``zoffset``), so the corrected direction is pinned here.

    With ``ZSHIFT = 0.1`` every row is the shifted table and differs from
    the unshifted ``WM1`` row the ``zshift == 0`` branch would use, while
    the membrane factors are offset-free by construction."""
    g = _patch(tmp_path, prop="TYPE51").shells
    w = initialise(g, model=None)
    assert w.shape == (4, 3)
    assert np.allclose(w, np.tile(wm_weights(3, 0.1), (4, 1)))
    assert not np.allclose(w, np.tile(WM1()[2, :3], (4, 1)))
    # the membrane half: no zoffset enters it (l.95-101 of the routine)
    assert np.allclose(membrane_weights(3), [1.0 / 6.0, 0.5, 1.0 / 6.0])


# ---------------------------------------------------------------------------
# the transcription itself: operation order, guards, table agreement
# ---------------------------------------------------------------------------

def test_wm_weight_spot_values_are_hand_computable():
    """``shell_offset_wm_ini.F90`` body, transcribed without reassociating::

        npt=2: fac=1, edge lobatto=(1/2-1/3)*1=1/6, w=pos/6
        npt=4, ipt=1: edge lobatto=(1/2-1/9)/3=7/54, w=-7/108
    """
    assert wm_weight(1, 1, 0.0) == 0.0          # the npt>1 guard
    assert wm_weight(1, 2, 0.0) == pytest.approx(-0.5 / 6.0, rel=1e-12)
    assert wm_weight(2, 2, 0.0) == pytest.approx(0.5 / 6.0, rel=1e-12)
    assert wm_weight(1, 2, 0.1) == pytest.approx(-0.4 / 6.0, rel=1e-12)
    assert wm_weight(2, 2, 0.1) == pytest.approx(0.6 / 6.0, rel=1e-12)
    assert wm_weight(1, 4, 0.0) == pytest.approx(-7.0 / 108.0, rel=1e-12)
    # interior points take lobatto=fac over the (decimal) Z01 abscissa
    z = Z01()[3, 1]
    assert wm_weight(2, 4, 0.1) == pytest.approx((z + 0.1) / 3.0, rel=1e-12)


def test_zero_offset_reproduces_the_standard_wm_table():
    """At ``zoffset == 0`` the offset branch must agree with the ``coqini_wm``
    table ``mulawc.F90:776`` uses instead — the two branches of the same
    ``if``.  Residuals are the ``WM1`` table's own 7-digit rounding
    (worst case 5e-8), so ``atol=1e-7`` bounds the table, not the
    operation order."""
    for npt in range(2, 12):
        assert np.allclose(wm_weights(npt, 0.0), WM1()[npt - 1, :npt],
                           rtol=1e-6, atol=1e-7), npt


def test_reference_positions_are_the_shifted_abscissae():
    """``pos = Z01(ipt,npt) + zoffset`` (l.102): the shifted reference
    surface the moment arm is measured from."""
    assert np.allclose(reference_positions(2, 0.1), [-0.4, 0.6])
    assert np.allclose(reference_positions(3, 0.0), Z01()[2, :3])


def test_initialise_caches_the_table_on_group_state(tmp_path):
    """``initialise`` stores ``group.state["wm_offset"]`` for the engine
    material path (the ``mulawc`` equivalent) and returns the same array."""
    g = _patch(tmp_path, prop="TYPE51").shells
    w = initialise(g, None)
    assert g.state["wm_offset"] is w
    g0 = _patch(tmp_path, prop="SHELL").shells
    assert np.array_equal(initialise(g0, None), g0.state["wm_offset"])


def test_bad_integration_counts_refuse_loudly():
    """``Z01`` holds rules 1..11: ``npt > 11`` would read past it (upstream
    would index out of bounds), and ``ipt`` outside ``1..npt`` names no
    point of the rule.  Both are ``ValueError``, never a quiet zero."""
    with pytest.raises(ValueError, match="npt"):
        wm_weight(1, 12, 0.0)
    with pytest.raises(ValueError, match="ipt"):
        wm_weight(0, 3, 0.0)
    with pytest.raises(ValueError, match="ipt"):
        wm_weight(4, 3, 0.0)
