"""P3.13 outcome (b): the ``auto`` rule excludes shells_qbat/shells_qeph.

VALIDATION.md §6.5(c) measured the QBAT/QEPH numba path SLOWER than NumPy
(c04: 395 s vs 330 s — dispatch overhead with nothing compiled), so ``auto``
picks the wrong backend for those groups. Real mirrors have since appeared in
``accel/jit_kernels/shells_qbat.py`` / ``shells_qeph.py`` and ARE faster at
the kernel level (2-6x on 36-100-element groups), but they are not yet
bitwise-identical to the NumPy reference (measured: QBAT ``lc`` 1e20 vs
0.866 on a unit square — the mirror re-derives the condensed length from
lengths where the reference uses squared half-diagonals — plus a different
flat/warped classifier; end-to-end forces differ at rel ~1.4 for QBAT and
~0.34 for QEPH, i.e. formula drift, not reduction noise). Until outcome (a)
— mirrors that are BOTH bitwise-identical and faster — lands, ``auto`` must
not serve these kernels: ``accel.get`` returns None for them on the
auto-selected path, so QBAT/QEPH groups always run the NumPy reference
while every other kernel still accelerates.

A user PIN (``PYRADIOSS_BACKEND=numba`` / ``-backend numba``) still serves
the mirrors — the exclusion is scoped to the auto rule, per the task.
"""

import numpy as np
import pytest

from pyradioss import accel
from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_qbat, shell_qeph
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_materials,
                                              resolve_node_groups,
                                              resolve_surfaces)

try:
    import numba  # noqa: F401 — the exclusion only matters when numba exists
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

needs_numba = pytest.mark.skipif(
    not HAS_NUMBA, reason="numba not installed (optional dependency)")

THR = accel._AUTO_MIN_ELEMENTS

#: the kernel blocks the QBAT/QEPH ``forces()`` paths request via accel.get
EXCLUDED = ("qbat_pre", "qbat_post", "qeph_pre", "qeph_post")


@pytest.fixture(autouse=True)
def _isolate_backend(monkeypatch):
    monkeypatch.delenv("PYRADIOSS_BACKEND", raising=False)
    monkeypatch.delenv("PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS", raising=False)
    accel._state.update(name=None, mod=None, forced=False)
    yield
    accel._state.update(name=None, mod=None, forced=False)
    accel.select_backend("numpy")


class _Grp:
    def __init__(self, n):
        self.n = n


class _Model:
    def __init__(self, *group_sizes):
        self._groups = [(f"g{i}", _Grp(n)) for i, n in enumerate(group_sizes)]

    def element_groups(self):
        return iter(self._groups)


@needs_numba
@pytest.mark.parametrize("kernel", EXCLUDED)
def test_auto_excludes_qbat_qeph_kernels(kernel):
    """Above the size threshold ``auto`` still selects numba overall, serves
    the qualified kernels — but NOT the QBAT/QEPH blocks."""
    assert accel.auto_select_backend(_Model(THR * 2)) == "numba"
    assert accel.get("hexa_pre") is not None      # control: others accelerate
    assert accel.get(kernel) is None, (
        f"auto must not serve {kernel} until the mirrors are bitwise + faster")


@needs_numba
def test_pinned_numba_still_serves_qbat_qeph():
    """The exclusion is scoped to the auto rule: an explicit numba pin keeps
    serving the QBAT/QEPH mirrors (the user's request wins)."""
    accel.select_backend("numba")
    for kernel in EXCLUDED:
        assert accel.get(kernel) is not None, kernel


def _plate_deck(nx, ny, ishell):
    """Flat nx*ny steel shell plate (nip=3); Ishell 12 -> QBAT, 24 -> QEPH."""
    L = ["/BEGIN", "p3.13 plate", "/NODE"]
    k = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            L.append(f"{k} {float(i)} {float(j)} 0.0")
            k += 1

    def nid(i, j):
        return i + j * (nx + 1) + 1

    L.append("/SHELL/1")
    e = 1
    for j in range(ny):
        for i in range(nx):
            L.append(f"{e} {nid(i,j)} {nid(i+1,j)} {nid(i+1,j+1)} {nid(i,j+1)}")
            e += 1
    L += ["/PART/1", "plate", "1 1",
          "/MAT/LAW1/1", "steel", "7.8e-6", "210.0 0.3",
          "/PROP/SHELL/1", "plate prop", f"{ishell} 0 0 0", "0 0 0 0 0",
          "3 0 0.1", "/END"]
    return "\n".join(L) + "\n"


def _build(deck, tmp_path, tag):
    f = tmp_path / f"{tag}_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    resolve_materials(model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    resolve_surfaces(model, log)
    initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    return model


def _forces(mod, g, model):
    rng = np.random.default_rng(20261016)
    v = (rng.random((model.numnod, 3)) - 0.5) * 0.02
    vr = (rng.random((model.numnod, 3)) - 0.5) * 0.002
    fint = np.zeros((model.numnod, 3))
    mint = np.zeros((model.numnod, 3))
    dt = mod.forces(g, model.x, v, vr, 1e-4, fint, mint)
    return fint, mint, dt


def _auto_vs_numpy(mod, attr, ishell, tmp_path, tag):
    # 6x6 = 36 shells >= THR, so auto resolves to the numba backend —
    # from which the QBAT/QEPH blocks must be excluded.
    assert 36 >= THR
    m_auto = _build(_plate_deck(6, 6, ishell), tmp_path, tag + "A")
    m_np = _build(_plate_deck(6, 6, ishell), tmp_path, tag + "N")
    g_auto, g_np = getattr(m_auto, attr), getattr(m_np, attr)
    assert g_auto is not None and g_auto.n == 36

    accel._state.update(name=None, mod=None, forced=False)
    assert accel.auto_select_backend(m_auto) == "numba"
    f_auto, m_auto_, dt_auto = _forces(mod, g_auto, m_auto)

    accel.select_backend("numpy")
    f_np, m_np_, dt_np = _forces(mod, g_np, m_np)

    # Same NumPy reference path in both runs: agreement far below any
    # physics scale (pre-exclusion this failed at rel ~1 for QBAT). The
    # residual ~1e-16 is the shared scatter3 primitive's documented
    # reduction-association tolerance, not a QBAT/QEPH kernel difference.
    assert np.allclose(f_auto, f_np, rtol=1e-9, atol=1e-12), (
        f"auto must run the NumPy reference: max rel "
        f"{np.abs(f_auto - f_np).max() / max(np.abs(f_np).max(), 1e-30):.3e}")
    assert np.allclose(m_auto_, m_np_, rtol=1e-9, atol=1e-12), (
        "auto must run the NumPy reference")
    assert np.allclose(np.atleast_1d(dt_auto), np.atleast_1d(dt_np),
                       rtol=1e-9, atol=0.0)


@needs_numba
def test_auto_qbat_forces_match_numpy_reference(tmp_path):
    """User-visible effect for QBAT: an above-threshold deck resolved by
    ``auto`` reproduces the forced-NumPy forces to print precision (the
    residual ~1e-16 is the shared scatter3 primitive's tolerance)."""
    _auto_vs_numpy(shell_qbat, "shells_qbat", 12, tmp_path, "Q13")


@needs_numba
def test_auto_qeph_forces_match_numpy_reference(tmp_path):
    """User-visible effect for QEPH: same contract as the QBAT case."""
    _auto_vs_numpy(shell_qeph, "shells_qeph", 24, tmp_path, "E13")
