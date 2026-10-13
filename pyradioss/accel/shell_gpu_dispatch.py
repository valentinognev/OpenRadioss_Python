# Copyright (C) 2026 the pyradioss contributors.
#
# This file is part of pyradioss (OpenRadioss_Python), a Python port of the
# OpenRadioss explicit FEM solver, and is licensed under the GNU Affero
# General Public License, version 3 or later (AGPL-3.0-or-later).
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.  See LICENSE for the full text.
#
# Derived from OpenRadioss (Copyright (C) 2026 Siemens, AGPL-3.0-or-later):
#   engine/source/elements/shell/coque/shell_internal_forces.F90 (modified:
#     the FORINTC_PREPARE_GPU group-eligibility condition at lines 522-534,
#     the super-group homogeneity key at lines 547-566, and the compute_sti
#     / reduce_elem_dt rule at lines 641-655, transcribed here as plain
#     predicates; gpu_shell_launch_async / gpu_shell_sync_scatter / the
#     resol.F IF(GPU==1) pipelining are recorded in the module docstring,
#     not transcribed — there is no device code to hide latency behind)
#   engine/source/elements/shell/coque/shell_gpu_mod.F90 (the dispatch
#     lifecycle this module mirrors: data_create -> allocate -> set_mat/hg
#     params -> upload -> step, with the CPU path as the fallback)
#   engine/source/engine/resol.F (the IF(GPU==1) gates at lines 3675-3682
#     and 4287-4308: launch_async BEFORE the CPU element work, sync_scatter
#     AFTER, so the GPU pipeline overlaps the CPU one)
#   engine/source/engine/execargcheck.F (the -GPU command-line flag that
#     sets GPU=1 at line 227)

"""Host-side dispatch for the upstream shell CUDA kernels — DECLINED.

Task P3.11 (``plan/04_phase3_elements_shell.md``) asked for the four
upstream CUDA files under ``engine/source/elements/shell/coque/``
(``shell_geometry_kernel.cu``, ``shell_strain_material_kernel.cu``,
``shell_force_assembly_kernel.cu``, ``shell_gpu_driver.cu``, dispatched
from ``shell_gpu_mod.F90`` via ``resol.F``) to be either ported to CuPy or
formally declined.  **They are declined** — option (b), recorded in
``docs/PORT_EXTENSIONS.md``.  This module ports the one part of that
pipeline that is solver logic rather than throughput: **when** a shell
group is routed to the GPU path, **what** the grouping rule is, and
**what the fallback is**.

Why declined
------------
The port's stated goal is readability, not GPU throughput, and the NumPy
and numba backends already cover performance.  The CUDA pipeline is
~3,900 lines of device code (551 + 994 + 708 + 1,315 + 212 + 1,082 + 232
including the driver, its header, the Fortran shim and the data struct)
whose only ground truth would be a CUDA device: the open-source oracle
cannot run it, so a transcription could never be parity-verified, and an
unverifiable kernel is exactly what this repository's licence and
provenance rules exist to keep out.

What is ported here
-------------------
* :func:`shell_group_uses_gpu` — the group-eligibility condition of
  ``forintc_prepare_gpu`` (``shell_internal_forces.F90:522-534``), the
  single place upstream decides that a 4-node LAW2 shell group runs on
  the GPU instead of the CPU kernels;
* :func:`super_group_key` — the five IPARG positions
  (NPT, ISMSTR, JHBE, ITHK, IPLA) whose equality batches consecutive
  groups into one GPU super-group (one CUDA handle, one stream, one
  device allocation);
* :func:`compute_sti_mode` / :func:`reduce_elem_dt` — the rule that the
  GPU element time step may lower DT2T only where the CPU ``cdt3.F``
  would (``compute_sti == 2``);
* :func:`gpu_shell_available` — ``False``: no CuPy mirrors of the three
  kernels exist, so there is no GPU path to make available.

The fallback, which is the whole story in this port
---------------------------------------------------
Every LAW2 shell group runs the CPU Fortran-faithful kernels in
``pyradioss/elements/`` (``shell_bt4.py`` and its QBAT/QEPH siblings).
Upstream, ``resol.F`` gates all GPU work on ``IF (GPU == 1)`` (the
``-GPU`` flag, ``execargcheck.F:227``): ``gpu_shell_launch_async`` is
called BEFORE the CPU element work and ``gpu_shell_sync_scatter`` AFTER,
so the GPU pipeline hides behind the CPU one.  With the kernels declined
there is no async launch to make and no latency to hide — the CPU path
is not a degraded mode here, it is the only mode.

Do not confuse this with the existing CuPy mirrors ``shell_pre`` and
``shell_post`` in ``accel/gpu_kernels``: those mirror the CPU *Fortran*
kernels (``ccoor3.F`` / ``cdefo3.F`` / ``czforc3.F`` / ``chvis3.F``) and
exist independently of the declined CUDA pipeline.
"""

from __future__ import annotations

__all__ = [
    "GPU_KERNEL_NAMES",
    "IPARG_ACTIVE",
    "IPARG_IFAILURE",
    "IPARG_IGTYP",
    "IPARG_ISMSTR",
    "IPARG_ISEATBELT",
    "IPARG_ISUBSTACK",
    "IPARG_ITHK",
    "IPARG_ITY",
    "IPARG_IXFEM",
    "IPARG_JHBE",
    "IPARG_MLW",
    "IPARG_NPT",
    "IPARG_IPLA",
    "compute_sti_mode",
    "gpu_shell_available",
    "reduce_elem_dt",
    "shell_group_uses_gpu",
    "super_group_key",
]

# ---------------------------------------------------------------------------
# IPARG positions read by the GPU route (1-based, as upstream writes
# IPARG(k, NG); a row is passed as a sequence with IPARG(k, NG) at
# iparg[k-1]).  From shell_internal_forces.F90 forintc_prepare_gpu.
# ---------------------------------------------------------------------------

IPARG_MLW = 1        # material law word; 2 = Johnson-Cook, the only GPU law
IPARG_ITY = 5        # element type; 3 = 4-node shell
IPARG_NPT = 6        # through-thickness integration points; 0 = global (no GPU)
IPARG_ACTIVE = 8     # 1 = group inactive (skipped entirely)
IPARG_ISMSTR = 9     # small-strain flag
IPARG_JHBE = 23      # hourglass formulation; >= 11 = BATOZ/QEPH (no GPU)
IPARG_ITHK = 28      # thickness update flag
IPARG_IPLA = 29      # plasticity algorithm
IPARG_IGTYP = 38     # property type; >= 29 = user property (no GPU)
IPARG_IFAILURE = 43  # failure model; 0 = none (GPU mockup simplification)
IPARG_IXFEM = 54     # XFEM; 0 = none
IPARG_ISUBSTACK = 71  # substack (laminate); 0 = none
IPARG_ISEATBELT = 91  # seatbelt; 0 = none

#: The three upstream CUDA kernel entry points.  ``gpu_shell_available()``
#: reports ``True`` only when all three have CuPy mirrors in
#: ``accel.gpu_kernels`` — none do (declined, see the module docstring).
GPU_KERNEL_NAMES = (
    "shell_geometry_kernel",
    "shell_strain_material_kernel",
    "shell_force_assembly_kernel",
)


def shell_group_uses_gpu(
    iparg,
    *,
    gpu_enabled: bool = True,
    nsect: int = 0,
    nmexmad: int = 0,
    nslipring: int = 0,
    isens_energy: int = 0,
) -> bool:
    """The ``forintc_prepare_gpu`` group-eligibility condition, transcribed.

    ``iparg`` is a group's IPARG row as a sequence with Fortran's
    ``IPARG(k, NG)`` at ``iparg[k-1]`` (see the ``IPARG_*`` constants).
    ``gpu_enabled`` models ``GPU == 1`` (the ``-GPU`` flag); ``nsect``,
    ``nmexmad``, ``nslipring`` and ``isens_energy`` are the group-loop
    scalars of the same name.  Returns ``True`` only when upstream would
    route this group to the CUDA pipeline; every ``False`` is a group the
    CPU kernels own.  The evaluation order follows the upstream IF
    (``shell_internal_forces.F90:522-534``) — a pure conjunction, so the
    order is for the reader, not for the result.
    """
    if iparg[IPARG_ITY - 1] != 3:      # 4-node shell only
        return False
    if iparg[IPARG_MLW - 1] != 2:      # Johnson-Cook only
        return False
    if iparg[IPARG_ACTIVE - 1] == 1:   # group inactive
        return False
    if iparg[IPARG_NPT - 1] <= 0:      # through-thickness integration only
        return False
    if iparg[IPARG_JHBE - 1] >= 11:    # standard formulation (not BATOZ/QEPH)
        return False
    if iparg[IPARG_IGTYP - 1] >= 29:   # not a user property
        return False
    if iparg[IPARG_IXFEM - 1] != 0:    # no XFEM
        return False
    if iparg[IPARG_IFAILURE - 1] != 0:  # no failure model
        return False
    if nsect != 0:                     # no section (mockup simplification)
        return False
    if nmexmad != 0:                   # no explicit mass scaling
        return False
    if nslipring != 0:                 # no slipring
        return False
    if iparg[IPARG_ISEATBELT - 1] != 0:  # no seatbelt
        return False
    if not gpu_enabled:                # IF (GPU == 1) in resol.F
        return False
    if iparg[IPARG_ISUBSTACK - 1] != 0:  # no substack
        return False
    if isens_energy != 0:              # no energy sensor
        return False
    return True


def super_group_key(iparg):
    """The five IPARG positions ``forintc_prepare_gpu`` compares to decide
    whether consecutive groups share one GPU super-group (lines 547-566):
    NPT, ISMSTR, JHBE, ITHK, IPLA.  Groups with equal keys are batched into
    a single ``GPU_SHELL_LAW2`` — one CUDA handle, one stream, one device
    allocation.  Anything else about two groups may differ freely.
    """
    return (
        iparg[IPARG_NPT - 1],
        iparg[IPARG_ISMSTR - 1],
        iparg[IPARG_JHBE - 1],
        iparg[IPARG_ITHK - 1],
        iparg[IPARG_IPLA - 1],
    )


def compute_sti_mode(nodadt, idt1sh, idtmins, idtmin3):
    """The STI-mode ladder of ``forintc_prepare_gpu`` (lines 641-647).

    ``0`` = no STI (CDT3 returns early), ``1`` = the CPXPY3+CHSTI3 formula
    (NODADT/=0 or IDT1SH=1 or IDTMINS=2), ``2`` = the CDT3 formula
    (NODADT=0, IDT1SH/=1, IDTMINS/=2, IDTMIN3/=0).
    """
    if nodadt != 0 or idt1sh == 1 or idtmins == 2:
        return 1
    if idtmin3 != 0:
        return 2
    return 0


def reduce_elem_dt(nodadt, idt1sh, idtmins, idtmin3):
    """Whether the GPU element time step may lower DT2T.

    ``forintc_prepare_gpu`` (lines 649-655): only when the CPU ``cdt3.F``
    would do the same — i.e. ``compute_sti == 2`` (NODADT=0, IDT1SH/=1,
    IDTMINS/=2, IDTMIN3/=0).  The GPU min-dt reduction must never lower
    DT2T where the CPU path would leave it alone.
    """
    return compute_sti_mode(nodadt, idt1sh, idtmins, idtmin3) == 2


def gpu_shell_available() -> bool:
    """Whether this port ships the upstream CUDA shell path.

    ``True`` only when all three upstream CUDA kernels have CuPy mirrors
    in ``accel.gpu_kernels`` AND a CUDA device is usable.  All three are
    declined (Task P3.11 option (b); ``docs/PORT_EXTENSIONS.md``), so this
    is ``False`` and every LAW2 shell group takes the CPU fallback.

    A future CuPy port flips this by adding the three mirrors — the
    dispatch above needs no change.
    """
    from . import gpu_kernels

    if not all(hasattr(gpu_kernels, name) for name in GPU_KERNEL_NAMES):
        return False
    return bool(gpu_kernels.HAS_CUPY)
