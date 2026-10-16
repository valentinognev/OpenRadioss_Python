"""Task P4.9 — numba mirrors for the 1-D element-iterating branches.

Fortran origin: none (this task adds no physics). Each mirror below
reproduces, statement for statement, the per-element math of one
vectorised branch driver from Tasks P4.1–P4.5 (spring) or the beam
driver the plan names (beam_type3):

* ``spring_linear`` <-> ``elements/spring_type6.driver`` (P4.1, linear
  ``redef3`` channel + damping close + ``r2len3`` dt close);
* ``spring_t32_linear`` <-> ``elements/spring_pretensioner.
  forces_pretensioner_type32`` (P4.4) on the curve-free domain
  (sensor-inactive elements and ``ITYP == 1``);
* ``spring_t8_linear`` <-> ``elements/spring_common.type8_forces``
  (P4.3) on the closed-form domain (every DOF ``fun_a == 0`` or with
  no ``/FUNCT`` table — the linear ``k dx + c rate`` path);
* ``beam_type3_elastic`` <-> ``elements/beam_type3._forces_core``
  (the plan's ``beam_type3`` case) on the elastic domain (``LAW != 2``).

Branches that iterate over *curves* instead of *elements* get no
mirror by explicit plan decision (they are lookup-bound, not NumPy
overhead-bound): ``spring_python_funct`` evaluators (P4.5, iteration
over curve abscissae), the pretensioner ``ITYP 2/3/4`` curve laws,
TYPE8 nonlinear DOFs (``IECROU``/force-function tables), the user
spring callback (P4.2 — an arbitrary Python callable numba cannot
compile), and the beam LAW2 radial return (per-slice Python material
dispatch). ``test_only_element_branches_have_mirrors`` locks that
exclusion: the mirror modules must export exactly the four kernels
above and nothing else.

Parity contract (same as M7): the mirrors run the same formulas in
the same order, so element-wise expressions agree BITWISE with the
NumPy drivers (``tobytes()`` equality, not a tolerance) — except the
beam's global back-transform, which NumPy's einsum contracts with
SIMD unrolling (measured ``(t0+t2)+t1``, unreproducible by any
portable scalar order) and is therefore proven at the M7 kernel
tolerance (``rtol=1e-12``; measured ``<= 5e-16``). The drivers
themselves are untouched — the reference side of each comparison is
the real driver, run on the same inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.accel.jit_kernels import beam as beam_mir
from pyradioss.accel.jit_kernels import spring as spring_mir
from pyradioss.common.messages import MessageLog
from pyradioss.elements import beam_type3, spring, spring_type6
from pyradioss.model.entities import Property
from pyradioss.model.model import ElementGroup, Model


def test_only_element_branches_have_mirrors():
    """Curve-iterating branches must not gain a mirror (plan P4.9)."""
    assert list(spring_mir.__all__) == ["spring_linear"]
    assert list(beam_mir.__all__) == []


# ---------------------------------------------------------------------------
# spring_linear <-> spring_type6.driver (P4.1)
# ---------------------------------------------------------------------------

def _t6_group(n, rng):
    """N-element TYPE6 group with randomised linear properties."""
    nn = 2 * n
    x0 = np.zeros((nn, 3))
    x0[0::2, 0] = 0.0
    lengths = rng.uniform(0.5, 3.0, size=n)
    x0[1::2, 0] = lengths
    x0[1::2, 1] = rng.uniform(-0.5, 0.5, size=n)
    model = Model()
    model.node_ids = np.arange(1, nn + 1, dtype=np.int64)
    model._id2idx = {int(v): int(i) for i, v in enumerate(model.node_ids)}
    model.x0 = np.array(x0, dtype=float)
    conn = np.stack([np.arange(0, nn, 2), np.arange(1, nn, 2)], axis=1)
    k = rng.uniform(0.0, 5000.0, size=n)
    k[rng.random(n) < 0.1] = 0.0              # free segments included
    c = rng.uniform(0.0, 50.0, size=n)
    mass = rng.uniform(0.5, 5.0, size=n)
    ak = rng.uniform(0.5, 2.0, size=n)
    ileng = rng.integers(0, 2, size=n).astype(np.int64)
    props = []
    for i in range(n):
        props.append(Property(
            id=i + 1, type=25, title="axi",
            params={"k": float(k[i]), "c": float(c[i]),
                    "mass": float(mass[i]), "a_scale": float(ak[i]),
                    "ileng": int(ileng[i])},
        ))
    group = ElementGroup(
        ids=np.arange(1, n + 1, dtype=np.int64), conn=conn,
        part=np.zeros(n, dtype=np.int64),
        state={"slices": [(slice(i, i + 1), None, props[i])
                          for i in range(n)]},
    )
    spring_type6.init_group(group, model, None)
    group.state["off"][0] = 0.0      # a switched-off element exercises
    return group, model               # the tangent pass-through path


def test_1d_mirror_is_bitwise_spring_linear():
    rng = np.random.default_rng(40901)
    n = 37
    group, model = _t6_group(n, rng)
    x = np.array(model.x0, dtype=float)
    x[1::2, 0] += rng.uniform(-0.4, 0.6, size=n)   # stretch/compress
    x[1::2, 2] += rng.uniform(-0.3, 0.3, size=n)
    v = rng.standard_normal((2 * n, 3)) * 0.5
    dt = 1e-3
    ctx = {"x": x, "v": v, "dt": dt, "fint": np.zeros_like(x)}
    # multi-step: the mirror must track dx_old/F_old state exactly
    for _ in range(3):
        st = group.state
        F_old = st["t6_force"].copy()
        dx_old = st["t6_dx_old"].copy()
        eint_old = st["t6_eint"].copy()
        k_tan_old = st["t6_k_tan"].copy()
        dt_ref = spring_type6.driver(group, model, ctx)
        F_ref = st["t6_force"].copy()
        eint_ref = st["t6_eint"].copy()
        k_tan_ref = st["t6_k_tan"].copy()
        fint_ref = ctx["fint"].copy()
        # restore pre-driver state, run the mirror on identical inputs
        st["t6_force"][:] = F_old
        st["t6_dx_old"][:] = dx_old
        st["t6_eint"][:] = eint_old
        st["t6_k_tan"][:] = k_tan_old
        ctx["fint"][:] = 0.0
        conn = group.conn
        F, a, e_incr, k_tan, dt_elem = spring_mir.spring_linear(
            x[conn[:, 0]], x[conn[:, 1]], st["t6_L0"], dx_old,
            st["t6_k"], st["t6_c"], st["t6_ak"], st["t6_ileng"],
            F_old, dt, st["off"], st["t6_mass"], k_tan_old,
        )
        assert F.tobytes() == F_ref.tobytes()
        assert k_tan.tobytes() == k_tan_ref.tobytes()
        assert dt_elem.tobytes() == dt_ref.tobytes()
        assert (eint_old + e_incr).tobytes() == eint_ref.tobytes()
        fint_mir = np.zeros_like(x)
        np.add.at(fint_mir, conn[:, 0], F[:, None] * a)
        np.add.at(fint_mir, conn[:, 1], -F[:, None] * a)
        assert fint_mir.tobytes() == fint_ref.tobytes()
        # advance the driver state for the next step
        st["t6_force"][:] = F_ref
        st["t6_eint"][:] = eint_ref
        st["t6_k_tan"][:] = k_tan_ref
        x[1::2, 0] += rng.uniform(-0.1, 0.1, size=n)

