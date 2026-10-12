"""
pyradioss.elements — element kernels.

Fortran origin: ``engine/source/elements/`` — one sub-directory per element
family (``solid/solide`` 8-node bricks, ``shell/coque`` 4-node shells,
``truss``, ``spring`` ...). In the Engine's main loop these are called from
the force driver (ALEMAIN/FORINT) for one *group* of same-type elements at
a time; the group loop is `pyradioss.engine.engine`.

Every kernel module exposes the same two entry points:

``init_group(group, model, log)``
    Called once by the Starter: builds the per-element ``state`` arrays
    (the *element buffer*, Fortran ``ELBUF_TAB``: stress, plastic strain,
    volumes...), computes element masses and returns the nodal mass (and
    inertia) contributions used to build the lumped mass matrix.

``forces(group, x, v, vr, dt, fint, mint)``
    Called every cycle by the Engine: computes internal + hourglass forces,
    scatters them into the global arrays ``fint`` (translations) and
    ``mint`` (rotations, shells only), updates the element state (stresses,
    energies) and returns the per-element critical time step array.

Sign convention: kernels ACCUMULATE the internal force with a MINUS sign
into ``fint`` — i.e. after all kernels ran, ``fint`` is the net force such
that  a = (fext + fint) / m  (this matches the Fortran A(3,*) accumulation).
"""

from . import (beam_fiber, beam_type3, iga3d, shell_basis, shell_bt4, shell_dkt18, shell_dkt6, shell_qbat,  # noqa: F401
               shell_qeph, shell_thick16, shell_tri3, solid_bric20, solid_cohesive, solid_connect, solid_heph, solid_hexa8,
               solid_hexa8_eas, solid_hexa8_full, solid_hexa8z, solid_penta6, solid_penta6_heph, solid_pyra5,
               solid_quad, solid_quad4_full, solid_shell_ha8, solid_tetra10, solid_tetra4,
               solid_tetra4_sfem, solid_tria3, solid_tshell8, spring, spring_advanced, spring_mat,
               thickshell_composite, thickshell_wedge6, truss, nstrand, xfem_shell, xfem_crack)
from .shell_basis import force_equilibrium  # noqa: F401  (fequilibre.F -- re-exported for the Engine)

KERNELS = {
    # Isogeometric Analysis 3D Solids (NURBS)
    "iga3d": iga3d,
    "ige3d": iga3d,

    # 3D Solids (8-node hexas, 20-node, wedges, tetras, pyramids, cohesive, solid shells)
    "bricks": solid_hexa8,
    "bricks_full": solid_hexa8_full,
    "bricks_eas": solid_hexa8_eas,
    "bricks_heph": solid_heph,
    "bricks_hexa8z": solid_hexa8z,
    "solid_shells_ha8": solid_shell_ha8,
    "cohesives": solid_cohesive,
    "solid_connect": solid_connect,
    "connectors": solid_connect,
    "bric20s": solid_bric20,
    "penta6s": solid_penta6,
    "penta6s_heph": solid_penta6_heph,
    "pyra5s": solid_pyra5,
    "tetras": solid_tetra4,
    "tetras_sfem": solid_tetra4_sfem,
    "tetra10s": solid_tetra10,

    # Thick Shells & Composites
    "tshells": solid_tshell8,
    "shel16s": shell_thick16,
    "thickshell_wedges": thickshell_wedge6,
    "thickshell_composites": thickshell_composite,

    # Thin Shells (4-node & 3-node & rotation-free & XFEM)
    "shells": shell_bt4,
    "shells_qbat": shell_qbat,
    "shells_qeph": shell_qeph,
    "sh3n": shell_tri3,
    "sh3n_dkt18": shell_dkt18,
    "shells_dkt6": shell_dkt6,
    "xfem_shell": xfem_shell,

    # 2D Continuum Solids (quads & triangles)
    "quads": solid_quad,
    "quads_full": solid_quad4_full,
    "trias": solid_tria3,

    # 1D Elements (beams, trusses, springs, multi-strand cables)
    "trusses": truss,
    "springs": spring,
    "spring_advanced": spring_advanced,
    "springs_advanced": spring_advanced,
    "spring_mat": spring_mat,
    "springs_mat": spring_mat,
    "nstrand": nstrand,
    "nstrands": nstrand,
    "beams": beam_type3,
    "beams_fiber": beam_fiber,
}

#: /PROP/TYPE18 -> dedicated integrated fiber beam formulation group.
BEAM_PROP_GROUPS = {
    18: "beams_fiber",
}

#: /PROP/SHELL Ishell -> dedicated element-technology group.
SHELL_ISHELL_GROUPS = {
    12: "shells_qbat",
    22: "shells_qeph",
    23: "shells_qeph",
    24: "shells_qeph",
}

#: /PROP/SHELL Ish3n -> dedicated element-technology group for 3-node / 6-node shells.
#: 2 = DKT18 (Discrete Kirchhoff Triangle — cdkforc3.F).
#: 3 = DKT6 (Rotation-free 6-node DKT macro-patch — cdk6forc3.F).
SH3N_ISHELL_GROUPS = {
    2: "sh3n_dkt18",
    3: "shells_dkt6",
}

#: /PROP/SOLID Isolid -> dedicated element-technology group. Read by
#: ``hm_read_prop06.F:285`` (which accepts {1,2,14,17,18,24}), stored as
#: IPARG(23,NG)=JHBE and dispatched by ``forint.F``: 2 FULL (s8forc3.F),
#: 14/15 TSHELL (scforc3.F), 16 HA8 (s8sforc3.F), 17 EAS (s8eforc3.F),
#: 21 COHESIVE (szforc3.F), 24 HEPH. Every value NOT listed here is refused,
#: with its upstream citation, by ``pyradioss.input.checks.unsupported_
#: isolid()`` — never skipped silently. Isolid 0 and 1 are deliberately
#: absent: both reach SFORC3 (forint.F:1156), which *is* the residual
#: ``bricks`` group, so naming one here would partition those elements out.
SOLID_ISOLID_GROUPS = {
    2: "bricks_full",
    14: "tshells",
    15: "tshells",
    16: "solid_shells_ha8",
    17: "bricks_eas",
    21: "cohesives",
    24: "bricks_heph",
}

#: /PROP/SOLID Itetra4 -> dedicated tetrahedral formulation group.
#: 3 = SFEM / NS-FEM (node-based smoothed finite element tetra — s4lagsfem.F)
TETRA4_ITETRA4_GROUPS = {
    3: "tetras_sfem",
}

#: /PROP/SOLID / /PROP/TYPE14 for 6-node wedges:
#: 24 = HEPH (physically-stabilized 6-node wedge — s6zforc3.F90)
PENTA_ISOLID_GROUPS = {
    24: "penta6s_heph",
}

#: /PROP/QUAD / /PROP/TYPE15 Iquad:
#: 2 = Full 2x2 Gauss quadrilateral with B-bar (q4forc2.F)
QUAD_IQUAD_GROUPS = {
    2: "quads_full",
}


