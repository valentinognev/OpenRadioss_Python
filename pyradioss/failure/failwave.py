# Ported from $OR_SRC/engine/source/materials/fail/failwave/update_failwave.F
# Ported from $OR_SRC/engine/source/materials/fail/failwave/set_failwave_nod3.F
# Ported from $OR_SRC/engine/source/materials/fail/failwave/set_failwave_nod4.F
# Ported from $OR_SRC/engine/source/materials/fail/failwave/upd_failwave_sh3n.F
# Ported from $OR_SRC/engine/source/materials/fail/failwave/upd_failwave_sh4n.F
# Ported from $OR_SRC/engine/source/materials/fail/failwave/seg_intersect.F
# Ported from $OR_SRC/common_source/modules/failwave_mod.F
# Ported from $OR_SRC/starter/source/materials/fail/failwave_init.F
"""
Failure wave propagation (/FAILWAVE).

Propagates crack and failure front information across neighboring shell
elements via shared nodes:
- Mode 1: Isotropic propagation (any cracked element tags its corner nodes,
  which immediately activates failure wave in all adjacent elements).
- Mode 2: Directional propagation through element edges only.
- Mode 3: Directional propagation through element edges and diagonals.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_TINY = 1e-20


def seg_intersect(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
) -> tuple[bool, float, float]:
    """Calculate intersection of two 2D line segments (x1,y1)-(x2,y2) and (x3,y3)-(x4,y4).

    Ported from $OR_SRC/engine/source/materials/fail/failwave/seg_intersect.F
    Subroutine: SEG_INTERSECT

    Parameters
    ----------
    x1, y1 : float
        Start point of segment 1.
    x2, y2 : float
        End point of segment 1.
    x3, y3 : float
        Start point of segment 2.
    x4, y4 : float
        End point of segment 2.

    Returns
    -------
    tuple[bool, float, float]
        (ok, xint, yint) where ok is True if segments intersect, with coordinates (xint, yint).
    """
    ax = x2 - x1
    ay = y2 - y1
    bx = x4 - x3
    by = y4 - y3

    dm = ay * bx - ax * by
    if abs(dm) > _TINY:
        cx = x3 - x1
        cy = y3 - y1
        alpha = (bx * cy - by * cx) / dm
        beta = (ax * cy - ay * cx) / dm

        if 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0:
            xint = x1 + alpha * ax
            yint = y1 + alpha * ay
            return True, float(xint), float(yint)

    return False, 0.0, 0.0


class Failwave:
    """Failure wave propagation manager.

    Ported from OpenRadioss FAILWAVE_MOD, FAILWAVE_INIT, and UPDATE_FAILWAVE.
    """

    def __init__(
        self,
        wave_mod: int = 1,
        num_nodes: Optional[int] = None,
        node_ids: Optional[Sequence[int]] = None,
        max_size: Optional[int] = None,
    ) -> None:
        """Initialize failure wave structure.

        Parameters
        ----------
        wave_mod : int
            1 = isotropic, 2 = directional edges, 3 = directional edges & diagonals.
        num_nodes : int, optional
            Number of nodes in the mesh.
        node_ids : sequence of int, optional
            Explicit global node IDs.
        max_size : int, optional
            Maximum stack levels (default 1 for mode 1, 10 for mode 2/3).
        """
        self.wave_mod = int(wave_mod)
        if self.wave_mod == 1:
            self.nddl = 1
            self.size = 1 if max_size is None else max_size
        else:
            self.nddl = 2
            self.size = 10 if max_size is None else max_size

        self.idxi: dict[int, int] = {}
        self.indx: list[int] = []

        if node_ids is not None:
            for idx, nid in enumerate(node_ids):
                self.idxi[int(nid)] = idx
                self.indx.append(int(nid))
            self.nnod = len(node_ids)
        elif num_nodes is not None:
            self.nnod = int(num_nodes)
            for i in range(self.nnod):
                self.idxi[i + 1] = i
                self.indx.append(i + 1)
        else:
            self.nnod = 0

        # Arrays: maxlev (nnod), maxlev_stack (nnod)
        # fwave_nod (nddl, nnod, size), fwave_nod_stack (nddl, nnod, size)
        self.maxlev = np.zeros(self.nnod, dtype=int)
        self.maxlev_stack = np.zeros(self.nnod, dtype=int)
        self.fwave_nod = np.zeros((self.nddl, self.nnod, self.size), dtype=int)
        self.fwave_nod_stack = np.zeros((self.nddl, self.nnod, self.size), dtype=int)

    def _ensure_nodes(self, node_list: Sequence[int]) -> None:
        """Ensure all nodes in node_list are registered."""
        needed = [int(n) for n in node_list if int(n) not in self.idxi]
        if not needed:
            return
        curr_len = len(self.indx)
        for i, nid in enumerate(needed):
            self.idxi[nid] = curr_len + i
            self.indx.append(nid)

        new_total = len(self.indx)
        old_total = self.nnod
        self.nnod = new_total

        new_maxlev = np.zeros(new_total, dtype=int)
        new_maxlev_stack = np.zeros(new_total, dtype=int)
        new_fwave_nod = np.zeros((self.nddl, new_total, self.size), dtype=int)
        new_fwave_nod_stack = np.zeros((self.nddl, new_total, self.size), dtype=int)

        if old_total > 0:
            new_maxlev[:old_total] = self.maxlev
            new_maxlev_stack[:old_total] = self.maxlev_stack
            new_fwave_nod[:, :old_total, :] = self.fwave_nod
            new_fwave_nod_stack[:, :old_total, :] = self.fwave_nod_stack

        self.maxlev = new_maxlev
        self.maxlev_stack = new_maxlev_stack
        self.fwave_nod = new_fwave_nod
        self.fwave_nod_stack = new_fwave_nod_stack

    def update(self) -> None:
        """Commit pending stacked failure wave info to active nodal table.

        Ported from $OR_SRC/engine/source/materials/fail/failwave/update_failwave.F
        Subroutine: UPDATE_FAILWAVE
        """
        self.maxlev[:] = self.maxlev_stack[:]
        self.maxlev_stack[:] = 0

        for i in range(self.nnod):
            level = min(self.maxlev[i], self.size)
            if level > 0:
                self.fwave_nod[:, i, :level] = self.fwave_nod_stack[:, i, :level]

    def set_failwave_nod3(
        self,
        fwave_el: np.ndarray,
        elem_nodes: np.ndarray,
        crkdir: Optional[np.ndarray] = None,
        dir_a: Optional[np.ndarray] = None,
        xl2: Optional[np.ndarray] = None,
        xl3: Optional[np.ndarray] = None,
        yl2: Optional[np.ndarray] = None,
        yl3: Optional[np.ndarray] = None,
    ) -> None:
        """Set failure wave on 3-node triangular shell elements.

        Ported from $OR_SRC/engine/source/materials/fail/failwave/set_failwave_nod3.F
        Subroutine: SET_FAILWAVE_NOD3

        Parameters
        ----------
        fwave_el : np.ndarray (nel,)
            Element failure flags: < 0 indicates newly cracked/damaged element.
            -1 = direction 1, -2 = direction 2, -3 = both directions.
        elem_nodes : np.ndarray (nel, 3)
            Global node identifiers for the elements.
        crkdir : np.ndarray (nel, 2), optional
            Crack direction vector (cos, sin).
        dir_a : np.ndarray (nel, 2), optional
            Element coordinate system rotation (cos, sin).
        xl2, xl3, yl2, yl3 : np.ndarray (nel,), optional
            Local coordinates of node 2 and node 3 relative to node 1 at origin.
        """
        fwave_el = np.asarray(fwave_el, dtype=int)
        elem_nodes = np.asarray(elem_nodes, dtype=int)
        nel = len(fwave_el)
        if nel == 0:
            return

        self._ensure_nodes(elem_nodes.ravel())

        if self.wave_mod == 1:
            # Isotropic propagation
            for i in range(nel):
                if fwave_el[i] < 0:
                    n1 = self.idxi[elem_nodes[i, 0]]
                    n2 = self.idxi[elem_nodes[i, 1]]
                    n3 = self.idxi[elem_nodes[i, 2]]
                    self.fwave_nod_stack[0, n1, 0] = 1
                    self.fwave_nod_stack[0, n2, 0] = 1
                    self.fwave_nod_stack[0, n3, 0] = 1
                    self.maxlev_stack[n1] = 1
                    self.maxlev_stack[n2] = 1
                    self.maxlev_stack[n3] = 1
            return

        # Directional propagation (wave_mod in (2, 3))
        rat1 = 0.5 * math.tan(math.pi / 6.0)
        rat2 = 1.0 - rat1

        # Process direction 1 and direction 2
        for dir_idx in (1, 2):
            for i in range(nel):
                flag = fwave_el[i]
                if dir_idx == 1 and flag not in (-1, -3):
                    continue
                if dir_idx == 2 and flag not in (-2, -3):
                    continue

                n1_id = elem_nodes[i, 0]
                n2_id = elem_nodes[i, 1]
                n3_id = elem_nodes[i, 2]
                nod_nn = [self.idxi[n1_id], self.idxi[n2_id], self.idxi[n3_id]]
                nod_id = [n1_id, n2_id, n3_id]

                c_dir = crkdir[i] if crkdir is not None else np.array([1.0, 0.0])
                if dir_idx == 1:
                    cd1, cd2 = -c_dir[1], c_dir[0]
                else:
                    cd1, cd2 = c_dir[0], c_dir[1]

                if dir_a is not None and dir_a.shape[1] >= 2:
                    cosx, sinx = dir_a[i, 0], dir_a[i, 1]
                    dir11 = cosx * cd1 - sinx * cd2
                    dir22 = cosx * cd2 + sinx * cd1
                else:
                    dir11, dir22 = cd1, cd2

                x2_val = xl2[i] if xl2 is not None else 1.0
                x3_val = xl3[i] if xl3 is not None else 0.5
                y2_val = yl2[i] if yl2 is not None else 0.0
                y3_val = yl3[i] if yl3 is not None else 1.0

                xm = (x2_val + x3_val) / 3.0
                ym = (y2_val + y3_val) / 3.0
                lmax = max(x2_val**2 + y2_val**2, x3_val**2 + y3_val**2)
                lmax = math.sqrt(max(lmax, _TINY)) * 2.0

                dx1 = xm - dir11 * lmax
                dy1 = ym - dir22 * lmax
                dx2 = xm + dir11 * lmax
                dy2 = ym + dir22 * lmax

                p1 = (x2_val * rat1, y2_val * rat1)
                p2 = (x2_val * rat2, y2_val * rat2)
                rx = x3_val - x2_val
                ry = y3_val - y2_val
                p3 = (x2_val + rx * rat1, y2_val + ry * rat1)
                p4 = (x2_val + rx * rat2, y2_val + ry * rat2)
                rx_b = -x3_val
                ry_b = -y3_val
                p5 = (x3_val + rx_b * rat1, y3_val + ry_b * rat1)
                p6 = (x3_val + rx_b * rat2, y3_val + ry_b * rat2)

                int1, _, _ = seg_intersect(p6[0], p6[1], p1[0], p1[1], dx1, dy1, dx2, dy2)
                int2, _, _ = seg_intersect(p2[0], p2[1], p3[0], p3[1], dx1, dy1, dx2, dy2)
                int3, _, _ = seg_intersect(p4[0], p4[1], p5[0], p5[1], dx1, dy1, dx2, dy2)
                int4, _, _ = seg_intersect(p1[0], p1[1], p2[0], p2[1], dx1, dy1, dx2, dy2)
                int5, _, _ = seg_intersect(p3[0], p3[1], p4[0], p4[1], dx1, dy1, dx2, dy2)
                int6, _, _ = seg_intersect(p5[0], p5[1], p6[0], p6[1], dx1, dy1, dx2, dy2)

                idf1 = [0, 0, 0]
                idf2 = [0, 0, 0]
                has_inter = True
                if int1 or int5:
                    idf1 = [nod_id[2], nod_id[2], nod_id[1]]
                    idf2 = [nod_id[1], 0, 0]
                elif int2 or int6:
                    idf1 = [nod_id[2], nod_id[0], nod_id[0]]
                    idf2 = [0, nod_id[2], 0]
                elif int3 or int4:
                    idf1 = [nod_id[1], nod_id[0], nod_id[1]]
                    idf2 = [0, 0, nod_id[0]]
                else:
                    has_inter = False

                if has_inter:
                    for k in range(3):
                        ncurr = nod_nn[k]
                        lev = self.maxlev_stack[ncurr] + 1
                        if lev > self.size:
                            lev = self.size
                        self.maxlev_stack[ncurr] = lev
                        self.fwave_nod_stack[0, ncurr, lev - 1] = idf1[k]
                        self.fwave_nod_stack[1, ncurr, lev - 1] = idf2[k]

    def set_failwave_nod4(
        self,
        fwave_el: np.ndarray,
        elem_nodes: np.ndarray,
        crkdir: Optional[np.ndarray] = None,
        dir_a: Optional[np.ndarray] = None,
        xl2: Optional[np.ndarray] = None,
        xl3: Optional[np.ndarray] = None,
        xl4: Optional[np.ndarray] = None,
        yl2: Optional[np.ndarray] = None,
        yl3: Optional[np.ndarray] = None,
        yl4: Optional[np.ndarray] = None,
    ) -> None:
        """Set failure wave on 4-node quad shell elements.

        Ported from $OR_SRC/engine/source/materials/fail/failwave/set_failwave_nod4.F
        Subroutine: SET_FAILWAVE_NOD4
        """
        fwave_el = np.asarray(fwave_el, dtype=int)
        elem_nodes = np.asarray(elem_nodes, dtype=int)
        nel = len(fwave_el)
        if nel == 0:
            return

        self._ensure_nodes(elem_nodes.ravel())

        if self.wave_mod == 1:
            for i in range(nel):
                if fwave_el[i] < 0:
                    for k in range(4):
                        n = self.idxi[elem_nodes[i, k]]
                        self.fwave_nod_stack[0, n, 0] = 1
                        self.maxlev_stack[n] = 1
            return

        # Directional propagation
        for dir_idx in (1, 2):
            for i in range(nel):
                flag = fwave_el[i]
                if dir_idx == 1 and flag not in (-1, -3):
                    continue
                if dir_idx == 2 and flag not in (-2, -3):
                    continue

                nod_id = [int(elem_nodes[i, k]) for k in range(4)]
                nod_nn = [self.idxi[nod_id[k]] for k in range(4)]

                c_dir = crkdir[i] if crkdir is not None else np.array([1.0, 0.0])
                if dir_idx == 1:
                    cd1, cd2 = -c_dir[1], c_dir[0]
                else:
                    cd1, cd2 = c_dir[0], c_dir[1]

                if dir_a is not None and dir_a.shape[1] >= 2:
                    cosx, sinx = dir_a[i, 0], dir_a[i, 1]
                    dir11 = cosx * cd1 - sinx * cd2
                    dir22 = cosx * cd2 + sinx * cd1
                else:
                    dir11, dir22 = cd1, cd2

                x2_val = xl2[i] if xl2 is not None else 1.0
                x3_val = xl3[i] if xl3 is not None else 1.0
                x4_val = xl4[i] if xl4 is not None else 0.0
                y2_val = yl2[i] if yl2 is not None else 0.0
                y3_val = yl3[i] if yl3 is not None else 1.0
                y4_val = yl4[i] if yl4 is not None else 1.0

                xm = (x2_val + x3_val + x4_val) * 0.25
                ym = (y2_val + y3_val + y4_val) * 0.25
                lmax = math.sqrt(max(xm**2 + ym**2, _TINY)) * 5.0

                dx1 = xm - dir11 * lmax
                dy1 = ym - dir22 * lmax
                dx2 = xm + dir11 * lmax
                dy2 = ym + dir22 * lmax

                idf1 = [0, 0, 0, 0]
                idf2 = [0, 0, 0, 0]
                has_inter = False

                if self.wave_mod == 2:
                    # Directional propagation through edges only
                    # Edge 1 (0,0)-(x2, y2)
                    int1, _, _ = seg_intersect(0.0, 0.0, x2_val, y2_val, dx1, dy1, dx2, dy2)
                    # Edge 2 (x2, y2)-(x3, y3)
                    int2, _, _ = seg_intersect(x2_val, y2_val, x3_val, y3_val, dx1, dy1, dx2, dy2)
                    # Edge 3 (x3, y3)-(x4, y4)
                    int3, _, _ = seg_intersect(x3_val, y3_val, x4_val, y4_val, dx1, dy1, dx2, dy2)
                    # Edge 4 (x4, y4)-(0, 0)
                    int4, _, _ = seg_intersect(x4_val, y4_val, 0.0, 0.0, dx1, dy1, dx2, dy2)

                    if int1 or int3:
                        idf1 = [nod_id[1], nod_id[0], nod_id[3], nod_id[2]]
                        has_inter = True
                    elif int2 or int4:
                        idf1 = [nod_id[3], nod_id[2], nod_id[1], nod_id[0]]
                        has_inter = True

                elif self.wave_mod == 3:
                    # Directional propagation through edges and diagonals
                    rat1 = 0.5 * math.tan(math.pi / 8.0)
                    rat2 = 1.0 - rat1
                    p1 = (x2_val * rat1, y2_val * rat1)
                    p2 = (x2_val * rat2, y2_val * rat2)
                    rx1 = x3_val - x2_val
                    ry1 = y3_val - y2_val
                    p3 = (x2_val + rx1 * rat1, y2_val + ry1 * rat1)
                    p4 = (x2_val + rx1 * rat2, y2_val + ry1 * rat2)
                    rx2 = x4_val - x3_val
                    ry2 = y4_val - y3_val
                    p5 = (x3_val + rx2 * rat1, y3_val + ry2 * rat1)
                    p6 = (x3_val + rx2 * rat2, y3_val + ry2 * rat2)
                    p7 = (x4_val * rat2, y4_val * rat2)
                    p8 = (x4_val * rat1, y4_val * rat1)

                    inter1, _, _ = seg_intersect(p1[0], p1[1], p2[0], p2[1], dx1, dy1, dx2, dy2)
                    inter2, _, _ = seg_intersect(p2[0], p2[1], p3[0], p3[1], dx1, dy1, dx2, dy2)
                    inter3, _, _ = seg_intersect(p3[0], p3[1], p4[0], p4[1], dx1, dy1, dx2, dy2)
                    inter4, _, _ = seg_intersect(p4[0], p4[1], p5[0], p5[1], dx1, dy1, dx2, dy2)
                    inter5, _, _ = seg_intersect(p5[0], p5[1], p6[0], p6[1], dx1, dy1, dx2, dy2)
                    inter6, _, _ = seg_intersect(p6[0], p6[1], p7[0], p7[1], dx1, dy1, dx2, dy2)
                    inter7, _, _ = seg_intersect(p7[0], p7[1], p8[0], p8[1], dx1, dy1, dx2, dy2)
                    inter8, _, _ = seg_intersect(p8[0], p8[1], p1[0], p1[1], dx1, dy1, dx2, dy2)

                    if inter1 or inter5:
                        idf1 = [nod_id[1], nod_id[0], nod_id[3], nod_id[2]]
                        has_inter = True
                    elif inter2 or inter6:
                        idf1 = [0, nod_id[0], 0, nod_id[2]]
                        idf2 = [0, nod_id[2], 0, nod_id[0]]
                        has_inter = True
                    elif inter3 or inter7:
                        idf1 = [nod_id[3], nod_id[2], nod_id[1], nod_id[0]]
                        has_inter = True
                    elif inter4 or inter8:
                        idf1 = [nod_id[3], 0, nod_id[1], 0]
                        idf2 = [nod_id[1], 0, nod_id[3], 0]
                        has_inter = True

                if has_inter:
                    for k in range(4):
                        ncurr = nod_nn[k]
                        lev = self.maxlev_stack[ncurr] + 1
                        if lev > self.size:
                            lev = self.size
                        self.maxlev_stack[ncurr] = lev
                        self.fwave_nod_stack[0, ncurr, lev - 1] = idf1[k]
                        self.fwave_nod_stack[1, ncurr, lev - 1] = idf2[k]

    def upd_failwave_sh3n(
        self,
        fwave_el: np.ndarray,
        offly: np.ndarray,
        dadv: np.ndarray,
        elem_nodes: np.ndarray,
    ) -> None:
        """Update failure flag on triangular shell elements from neighbor wave data.

        Ported from $OR_SRC/engine/source/materials/fail/failwave/upd_failwave_sh3n.F
        Subroutine: SET_FAILWAVE_SH3N
        """
        fwave_el_arr = np.asarray(fwave_el, dtype=int)
        offly_arr = np.asarray(offly, dtype=int)
        dadv_arr = np.asarray(dadv, dtype=float)
        elem_nodes_arr = np.asarray(elem_nodes, dtype=int)
        nel = len(fwave_el_arr)
        if nel == 0:
            return

        self._ensure_nodes(elem_nodes_arr.ravel())

        ndr = [1, 2, 0]  # Fortran 2, 3, 1 (0-based)
        ndl = [2, 0, 1]  # Fortran 3, 1, 2 (0-based previous node)

        for i in range(nel):
            if offly_arr[i] != 1 or dadv_arr[i] != 1.0:
                continue

            nodes = [elem_nodes_arr[i, k] for k in range(3)]
            nod_nn = [self.idxi[nodes[k]] for k in range(3)]

            if self.wave_mod == 1:
                nfail = sum(self.fwave_nod[0, nod_nn[k], 0] for k in range(3))
                if nfail > 0:
                    fwave_el[i] = 1
            else:
                found = False
                for k in range(3):
                    ncurr = nod_nn[k]
                    max_l = self.maxlev[ncurr]
                    if max_l <= 0:
                        continue
                    knext = nodes[ndr[k]]
                    kprev = nodes[ndl[k]]
                    for lev in range(min(max_l, self.size)):
                        fnod1 = self.fwave_nod[0, ncurr, lev]
                        fnod2 = self.fwave_nod[1, ncurr, lev]

                        edge_cond = (fnod2 == 0) and (fnod1 == knext or fnod1 == kprev)
                        diag_cond = (
                            fnod1 > 0
                            and fnod2 > 0
                            and fnod1 != kprev
                            and fnod1 != knext
                            and fnod2 != kprev
                            and fnod2 != knext
                        )
                        if edge_cond or diag_cond:
                            found = True
                            fwave_el[i] = 1
                            break
                    if found:
                        break

    def upd_failwave_sh4n(
        self,
        fwave_el: np.ndarray,
        offly: np.ndarray,
        dadv: np.ndarray,
        elem_nodes: np.ndarray,
    ) -> None:
        """Update failure flag on quad shell elements from neighbor wave data.

        Ported from $OR_SRC/engine/source/materials/fail/failwave/upd_failwave_sh4n.F
        Subroutine: SET_FAILWAVE_SH4N
        """
        fwave_el_arr = np.asarray(fwave_el, dtype=int)
        offly_arr = np.asarray(offly, dtype=int)
        dadv_arr = np.asarray(dadv, dtype=float)
        elem_nodes_arr = np.asarray(elem_nodes, dtype=int)
        nel = len(fwave_el_arr)
        if nel == 0:
            return

        self._ensure_nodes(elem_nodes_arr.ravel())

        ndr = [1, 2, 3, 0]  # Fortran 2, 3, 4, 1 (0-based)
        ndl = [3, 0, 1, 2]  # Fortran 4, 1, 2, 3 (0-based)

        for i in range(nel):
            if offly_arr[i] != 1 or dadv_arr[i] != 1.0:
                continue

            nodes = [elem_nodes_arr[i, k] for k in range(4)]
            nod_nn = [self.idxi[nodes[k]] for k in range(4)]

            if self.wave_mod == 1:
                nfail = sum(self.fwave_nod[0, nod_nn[k], 0] for k in range(4))
                if nfail > 0:
                    fwave_el[i] = 1
            elif self.wave_mod == 2:
                found = False
                for k in range(4):
                    ncurr = nod_nn[k]
                    max_l = self.maxlev[ncurr]
                    if max_l <= 0:
                        continue
                    knext = nodes[ndr[k]]
                    for lev in range(min(max_l, self.size)):
                        fnod1 = self.fwave_nod[0, ncurr, lev]
                        fnod2 = self.fwave_nod[1, ncurr, lev]
                        if fnod1 == knext and fnod2 == 0:
                            found = True
                            fwave_el[i] = 1
                            break
                    if found:
                        break
            elif self.wave_mod == 3:
                found = False
                for k in range(4):
                    ncurr = nod_nn[k]
                    max_l = self.maxlev[ncurr]
                    if max_l <= 0:
                        continue
                    knext = nodes[ndr[k]]
                    kprev = nodes[ndl[k]]
                    for lev in range(min(max_l, self.size)):
                        fnod1 = self.fwave_nod[0, ncurr, lev]
                        fnod2 = self.fwave_nod[1, ncurr, lev]
                        edge_cond = (fnod2 == 0) and (fnod1 == knext or fnod1 == kprev)
                        diag_cond = (
                            fnod1 > 0
                            and fnod2 > 0
                            and fnod1 != kprev
                            and fnod1 != knext
                            and fnod2 != kprev
                            and fnod2 != knext
                        )
                        if edge_cond or diag_cond:
                            found = True
                            fwave_el[i] = 1
                            break
                    if found:
                        break
