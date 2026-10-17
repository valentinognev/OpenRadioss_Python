"""Task P4.6 — the nine ``/PROP/TYPE33`` joint topologies.

Starter origin (one reader per joint type, dispatched by ``type``)::

  starter/source/properties/spring/hm_read_prop33.F            (ITYP 1..9 dispatch)
  starter/source/properties/spring/hm_read_prop33_sph_jnt.F    (ITYP 1)
  starter/source/properties/spring/hm_read_prop33_rev_jnt.F    (ITYP 2)
  starter/source/properties/spring/hm_read_prop33_cyl_jnt.F    (ITYP 3)
  starter/source/properties/spring/hm_read_prop33_plan_jnt.F   (ITYP 4)
  starter/source/properties/spring/hm_read_prop33_univ_jnt.F   (ITYP 5)
  starter/source/properties/spring/hm_read_prop33_trans_jnt.F  (ITYP 6)
  starter/source/properties/spring/hm_read_prop33_old_jnt.F    (ITYP 7)
  starter/source/properties/spring/hm_read_prop33_fix_jnt.F    (ITYP 8)
  starter/source/properties/spring/hm_read_prop33_free_jnt.F   (ITYP 9)

Element origin::

  engine/source/elements/joint/ruser33.F      (RUSER33 + DEF_FDOF free-DOF table)
  engine/source/elements/joint/rgjoint.F      (RGJOINT driver)
  engine/source/elements/joint/joint_block_stiffness.F

DOF index convention: 0..2 are the translations (Tx, Ty, Tz), 3..5 the
rotations (Rx, Ry, Rz) — the 0-based form of ``DEF_FDOF``'s 1-based
``FDOF(1..6)``, and of ``KJointEngine._setup_dof_masks`` in
``pyradioss/engine/kjoint.py``, which this matrix agrees with on all nine
kinds. ``free_dofs`` is read off each reader's ``SET_U_GEO(4..9)`` stiffness
slots (a slot carrying the joint stiffness is free; a slot carrying ``KNN``
is blocked), not off the joint's name.

``ncmax`` is the number of KNN-blocked relative-DOF slots, i.e.
``6 - dofs`` — the count of penalty-constraint rows the joint imposes. It is
read off the same ``SET_U_GEO(4..9)`` tables, so ``ncmax == 6 - dofs`` holds
for every kind by construction, not by coincidence.

Two upstream quirks, recorded so Phase 13 does not rediscover them:

* ``DEF_FDOF`` has no ``JTYP==5`` (universal) branch, so under ``IGTYP==45``
  a universal joint takes ``FDOF = 0`` on the ``SENS_BLOCK`` sensor path.
  The ``hm_read_prop33_univ_jnt.F`` reader itself stores ``Kry, Krz`` in
  ``GEO(8,9)`` — Ry, Rz free — which is what this matrix follows.
* A joint is computed once on the owning domain and frontier-summed (the
  fact ``docs/OPEN_BUGS.md`` SPMD-3 records the port getting wrong before
  ``model.kjoints`` was refused in ``check_spmd_support`` citing
  ``ruser33.F`` / ``spmd_exch_a.F``). Phase 13 consumes this matrix for the
  per-topology domain treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    "JointTopology",
    "TOPOLOGIES",
    "by_ityp",
    "routed",
    "refused",
]


@dataclass(frozen=True)
class JointTopology:
    """One ``/PROP/TYPE33`` joint kind.

    ``ityp`` is the Radioss joint-type number (``hm_read_prop33.F:103-129``).
    ``dofs`` is ``len(free_dofs)``. ``ncmax`` is the KNN-blocked slot count.
    """

    kind: str
    ityp: int
    dofs: int
    free_dofs: Tuple[int, ...]
    ncmax: int


#: One entry per joint kind, in ITYP order (``hm_read_prop33.F`` dispatch).
TOPOLOGIES: Dict[str, JointTopology] = {
    "spherical": JointTopology("spherical", 1, 3, (3, 4, 5), 3),
    "revolute": JointTopology("revolute", 2, 1, (3,), 5),
    "cylindrical": JointTopology("cylindrical", 3, 2, (0, 3), 4),
    "planar": JointTopology("planar", 4, 3, (1, 2, 3), 3),
    "universal": JointTopology("universal", 5, 2, (4, 5), 4),
    "translation": JointTopology("translation", 6, 1, (0,), 5),
    "old": JointTopology("old", 7, 2, (1, 2), 4),
    "fixed": JointTopology("fixed", 8, 0, (), 6),
    "free": JointTopology("free", 9, 6, (0, 1, 2, 3, 4, 5), 0),
}

_BY_ITYP: Dict[int, JointTopology] = {t.ityp: t for t in TOPOLOGIES.values()}


def by_ityp(ityp: int) -> JointTopology:
    """Return the topology for a Radioss joint-type number 1..9.

    Anything else is refused, mirroring the ``ELSE`` branch of
    ``hm_read_prop33.F`` (``ANCMSG`` 409, wrong joint type).
    """
    try:
        return _BY_ITYP[int(ityp)]
    except (KeyError, TypeError, ValueError):
        raise KeyError(
            f"unknown /PROP/TYPE33 joint type {ityp!r}: "
            "hm_read_prop33.F accepts ITYP 1..9 (ANCMSG 409)"
        ) from None


def routed() -> Tuple[str, ...]:
    """Kinds computed by the engine.

    All nine: ``engine/kjoint.py`` ``KJointEngine`` builds an explicit
    free/locked mask per joint type (``_setup_dof_masks``), and the
    cylindrical kind is additionally covered by ``engine/cyl_joint.py``.
    """
    return tuple(TOPOLOGIES)


def refused() -> Dict[str, str]:
    """Kinds with no engine path, each mapped to the upstream gap citation.

    Empty today — every topology is computed — but the mapping stays so a
    future gap is refused with its reader cited (``refused_solid_routines``
    shape). Every value must start with
    ``engine/source/elements/joint/``.
    """
    return {}
