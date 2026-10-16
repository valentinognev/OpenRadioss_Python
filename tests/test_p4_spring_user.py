"""Tests for IGTYP 29/30/31 user springs (Task P4.2).

Fortran origin:
  engine/source/elements/spring/rforc3.F (the ``IGTYP >= 29 .AND. IGTYP <= 31``
  block: copies GBUF state into USER_* scratch, calls ENG_USERLIB_RUSER when
  the user library is linked, else ANCMSG(MSGID=257) + ARRET(2))
"""

import numpy as np
import pytest

from pyradioss.elements import spring_user
from pyradioss.elements.spring_user import UserSpringNotRegistered


class _UserSpringGroup:
    """Minimal driver-contract group: one IGTYP shared by all n elements."""

    def __init__(self, igtyp: int, n: int):
        self.igtyp = igtyp
        self.n = n
        self.state: dict = {}


def _user_spring_group(igtyp: int = 29, n: int = 1) -> _UserSpringGroup:
    return _UserSpringGroup(igtyp=igtyp, n=n)


def _ctx(group: _UserSpringGroup) -> dict:
    return {"t": 0.0, "dt": 1.0e-6}


@pytest.fixture(autouse=True)
def _clean_registry():
    spring_user.clear_user_springs()
    yield
    spring_user.clear_user_springs()


def test_unregistered_user_spring_is_refused_not_silently_skipped():
    g = _user_spring_group(igtyp=29)
    with pytest.raises(UserSpringNotRegistered, match="IGTYP 29"):
        spring_user.driver(g, None, _ctx(g))


def test_registered_user_spring_is_called_once_per_element():
    calls = []
    spring_user.register_user_spring(29, lambda el, ctx: calls.append(el) or np.zeros(3))
    g = _user_spring_group(igtyp=29, n=4)
    spring_user.driver(g, None, _ctx(g))
    assert len(calls) == 4


@pytest.mark.parametrize("igtyp", [29, 30, 31])
def test_each_user_igtyp_refuses_by_name_when_unregistered(igtyp):
    g = _user_spring_group(igtyp=igtyp)
    with pytest.raises(UserSpringNotRegistered, match=f"IGTYP {igtyp}"):
        spring_user.driver(g, None, _ctx(g))


def test_register_rejects_igtyp_outside_29_30_31():
    with pytest.raises(ValueError, match="29, 30 or 31"):
        spring_user.register_user_spring(8, lambda el, ctx: np.zeros(3))


def test_registered_callback_force_is_stored_per_element():
    spring_user.register_user_spring(30, lambda el, ctx: np.array([1.0, 0.0, 0.0]))
    g = _user_spring_group(igtyp=30, n=2)
    spring_user.driver(g, None, _ctx(g))
    assert np.allclose(g.state["force"], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
