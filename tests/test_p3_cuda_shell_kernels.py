"""Task P3.11 — the shell CUDA kernels: ported or formally declined.

WHAT THIS FILE IS FOR
---------------------
``plan/04_phase3_elements_shell.md`` §Task P3.11 asks for a decision on the
four upstream CUDA files under ``engine/source/elements/shell/coque/``
(``shell_geometry_kernel.cu``, ``shell_strain_material_kernel.cu``,
``shell_force_assembly_kernel.cu``, ``shell_gpu_driver.cu``, dispatched from
``shell_gpu_mod.F90`` via ``resol.F``).  Two outcomes are legitimate:

* (a) a CuPy port of the three kernels, or
* (b) a formal decline recorded in ``docs/PORT_EXTENSIONS.md`` — with the
  host-side dispatch semantics ported, because *when* a GPU path is taken
  and what the fallback is is solver logic, not throughput.

What is not acceptable is silence: a ``.cu`` file upstream with no port and
no record reads as an oversight.

The gate test below is the plan's, modulo its namespace: either the
registry names all three kernels, or the port ships the GPU path
(``gpu_shell_available()``).  It resolves through
``pyradioss.accel.shell_gpu_dispatch`` rather than a top-level
``accel.gpu_shell_available`` alias because
``pyradioss/accel/__init__.py`` is pinned at its size-budget ceiling
(``tests/test_p1_module_size.py``), so even a one-line alias would fail
the gate.  Today the answer is (b) — declined, dispatch
ported in ``pyradioss/accel/shell_gpu_dispatch.py``.

The remaining tests pin the ported dispatch semantics themselves, so the
predicate cannot drift away from ``forintc_prepare_gpu`` /
``shell_gpu_mod.F90`` / ``resol.F``: the group-eligibility condition, the
super-group homogeneity key, and the ``compute_sti`` / ``reduce_elem_dt``
rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyradioss.accel import shell_gpu_dispatch


def test_gpu_shell_path_is_either_ported_or_declined():
    text = Path("docs/PORT_EXTENSIONS.md").read_text()
    either = all(k in text for k in ("shell_geometry_kernel",
                                     "shell_strain_material_kernel",
                                     "shell_force_assembly_kernel"))
    assert either or shell_gpu_dispatch.gpu_shell_available()


# ---------------------------------------------------------------------------
# The decline decision, pinned
# ---------------------------------------------------------------------------


def test_gpu_shell_available_is_false_because_the_kernels_are_declined():
    """The port ships no CUDA shell path: all three kernels are declined.

    If a later task ports the CuPy mirrors, ``gpu_shell_available`` flips to
    reflecting them automatically (it checks the mirrors, not this
    assertion) — and this test is the record that the decision was (b).
    """
    assert shell_gpu_dispatch.gpu_shell_available() is False


# ---------------------------------------------------------------------------
# The host-side dispatch predicate, ported from forintc_prepare_gpu
# ---------------------------------------------------------------------------


def _law2_group(overrides=None):
    """An IPARG row (1-based Fortran positions as ``iparg[k-1]``) for a
    group that satisfies every GPU eligibility condition by default;
    ``overrides`` maps a 1-based IPARG position to its value."""
    row = [0] * 91
    row[0] = 2     # IPARG(1)  MLW   = 2  -> Johnson-Cook
    row[4] = 3     # IPARG(5)  ITY   = 3  -> 4-node shell
    row[5] = 3     # IPARG(6)  NPT   = 3  -> through-thickness integration
    row[8] = 1     # IPARG(9)  ISMSTR = 1
    row[22] = 1    # IPARG(23) JHBE  = 1  -> standard formulation
    row[27] = 1    # IPARG(28) ITHK  = 1  -> thickness update on
    row[28] = 0    # IPARG(29) IPLA  = 0
    row[37] = 1    # IPARG(38) IGTYP = 1  -> standard property
    for position, value in (overrides or {}).items():
        row[position - 1] = value
    return row


def test_standard_law2_shell_group_takes_the_gpu_path_when_enabled():
    assert shell_gpu_dispatch.shell_group_uses_gpu(_law2_group()) is True


@pytest.mark.parametrize("position, value, why", [
    (1, 0, "MLW=0 void groups never reach the GPU branch"),
    (1, 13, "MLW=13 rigid groups never reach the GPU branch"),
    (1, 3, "MLW=3 is not Johnson-Cook"),
    (5, 4, "ITY=4 is not a 4-node shell"),
    (8, 1, "IPARG(8)=1 inactive group"),
    (6, 0, "NPT=0 global (no through-thickness integration)"),
    (23, 11, "JHBE=11 is BATOZ/QEPH, not the standard formulation"),
    (38, 29, "IGTYP=29 is a user property"),
    (54, 1, "IXFEM=1 XFEM group"),
    (43, 1, "IFAILURE=1 failure model"),
    (71, 1, "ISUBSTACK=1 substack group"),
    (91, 1, "ISEATBELT=1 seatbelt group"),
])
def test_each_disqualifying_flag_keeps_the_group_on_cpu(position, value, why):
    assert shell_gpu_dispatch.shell_group_uses_gpu(
        _law2_group({position: value})) is False, why


def test_no_gpu_flag_means_no_gpu_path():
    """resol.F gates every GPU call on IF(GPU == 1) (-GPU, execargcheck.F)."""
    assert shell_gpu_dispatch.shell_group_uses_gpu(_law2_group(),
                                                   gpu_enabled=False) is False


@pytest.mark.parametrize("kwarg", [
    "nsect", "nmexmad", "nslipring", "isens_energy",
])
def test_each_conditional_exclusion_keeps_the_group_on_cpu(kwarg):
    assert shell_gpu_dispatch.shell_group_uses_gpu(
        _law2_group(), **{kwarg: 1}) is False


def test_mlw_other_than_2_stays_on_cpu():
    assert shell_gpu_dispatch.shell_group_uses_gpu(_law2_group({1: 3})) is False


# ---------------------------------------------------------------------------
# Super-group homogeneity (forintc_prepare_gpu): groups sharing the key share
# one CUDA handle
# ---------------------------------------------------------------------------


def test_super_group_key_is_the_five_gpu_relevant_iparg_positions():
    row = _law2_group()
    assert shell_gpu_dispatch.super_group_key(row) == (3, 1, 1, 1, 0)


def test_groups_differing_only_in_irrelevant_positions_share_a_super_group():
    a = _law2_group()
    b = _law2_group()
    b[18] = 9    # IPARG(19) NVC — not part of the homogeneity key
    b[41] = 7    # IPARG(42) ISORTH — not part of the homogeneity key
    assert shell_gpu_dispatch.super_group_key(a) == \
        shell_gpu_dispatch.super_group_key(b)


def test_groups_differing_in_a_key_position_split_super_groups():
    a = _law2_group()
    b = _law2_group({6: 5})   # IPARG(6) NPT differs -> different super-group
    assert shell_gpu_dispatch.super_group_key(a) != \
        shell_gpu_dispatch.super_group_key(b)


# ---------------------------------------------------------------------------
# compute_sti / reduce_elem_dt (forintc_prepare_gpu): the GPU element dt may
# reduce DT2T only where the CPU CDT3 would
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nodadt, idt1sh, idtmins, idtmin3, mode", [
    (0, 0, 0, 1, 2),   # NODADT=0, IDTMIN3/=0 -> CDT3 formula
    (0, 0, 0, 0, 0),   # nothing set -> CDT3 returns early
    (1, 0, 0, 1, 1),   # NODADT/=0 -> CPXPY3+CHSTI3 formula
    (0, 1, 0, 1, 1),   # IDT1SH=1 -> CPXPY3+CHSTI3 formula
    (0, 0, 2, 1, 1),   # IDTMINS=2 -> CPXPY3+CHSTI3 formula
])
def test_compute_sti_mode_matches_the_upstream_branches(nodadt, idt1sh,
                                                        idtmins, idtmin3, mode):
    assert shell_gpu_dispatch.compute_sti_mode(
        nodadt, idt1sh, idtmins, idtmin3) == mode


@pytest.mark.parametrize("nodadt, idt1sh, idtmins, idtmin3, reduce", [
    (0, 0, 0, 1, True),    # compute_sti==2 -> reduce_elem_dt
    (0, 0, 0, 0, False),
    (1, 0, 0, 1, False),   # compute_sti==1 -> no reduction
    (0, 1, 0, 1, False),
    (0, 0, 2, 1, False),
])
def test_reduce_elem_dt_only_where_cpu_cdt3_would_reduce(nodadt, idt1sh,
                                                         idtmins, idtmin3,
                                                         reduce):
    assert shell_gpu_dispatch.reduce_elem_dt(
        nodadt, idt1sh, idtmins, idtmin3) is reduce
