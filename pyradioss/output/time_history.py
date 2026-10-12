"""
Time-history output (the T01 file, /TFILE + /TH).

Fortran origin: ``engine/source/output/th/`` — the original writes a
binary T01 read by post-processors. The port writes CSV with a commented
header: column 1 is time, then the global variables, then one column per
requested /TH variable, named ``<kind><id>_<VAR>``.

Global columns (always present, matching the original's global T-file
variables): IE internal energy, KE kinetic energy (translational),
HE hourglass energy, CE contact energy, EN numerical dissipation of the
element dampers (M6, see the engine's step-6c ledger), DE /DAMP
dissipation (M6), EW external work, ERR energy error %, MASS (grows
under /DT/NODA/CST mass scaling), MOMX/Y/Z momentum components.

Per-node variables: DX DY DZ (displacement), VX VY VZ (velocity),
AX AY AZ (acceleration is not stored — approximated by force/mass).
Per-part variables: IE (internal energy of the part's elements),
KE (kinetic energy of the part's element masses, computed from their
nodes' velocities).
Per-section variables (M5, /TH/SECT): FX FY FZ MX MY MZ — the section
force/moment resultants computed by pyradioss/engine/sections.py.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..elements.shell_lump import corner_ke
from ..model.model import Model


_STRESS_MAP = {
    "SIGXX": 0, "SIGX": 0, "SX": 0, "SXX": 0, "SIG11": 0, "S11": 0,
    "SIGYY": 1, "SIGY": 1, "SY": 1, "SYY": 1, "SIG22": 1, "S22": 1,
    "SIGZZ": 2, "SIGZ": 2, "SZ": 2, "SZZ": 2, "SIG33": 2, "S33": 2,
    "SIGXY": 3, "SXY": 3, "SIG12": 3, "S12": 3, "SIGYX": 3, "SYX": 3, "SIG21": 3, "S21": 3,
    "SIGYZ": 4, "SYZ": 4, "SIG23": 4, "S23": 4, "SIGZY": 4, "SZY": 4, "SIG32": 4, "S32": 4,
    "SIGZX": 5, "SZX": 5, "SIGXZ": 5, "SXZ": 5, "SIG31": 5, "S31": 5, "SIG13": 5, "S13": 5,
}


class TimeHistory:
    def __init__(self, path: str, model: Model, log):
        self.path = path
        self.model = model
        self._fh = open(path, "w", encoding="utf-8")
        self._cols: List[str] = [
            "TIME", "IE", "KE", "HE", "CE", "EN", "DE", "EW", "ERR%",
            "MASS", "MOMX", "MOMY", "MOMZ"]

        # resolve /TH requests once (Starter checked the ids)
        self._node_req = []   # (label, node_idx, var)
        self._part_req = []   # (label, part_id, var)
        self._sect_req = []   # (label, sect_id, var)
        self._other_req = []  # (label, id, var)
        for th in model.th_requests:
            for oid in th.ids:
                for var in th.variables:
                    if th.kind == "NODE":
                        try:
                            idx = model.node_index(oid)
                        except (KeyError, ValueError):
                            continue
                        self._node_req.append((f"N{oid}_{var}", idx, var))
                    elif th.kind in ("SECT", "SECTIO", "SECTION"):
                        self._sect_req.append((f"S{oid}_{var}", oid, var))
                    elif th.kind == "PART":
                        self._part_req.append((f"P{oid}_{var}", oid, var))
                    else:
                        self._other_req.append((f"{th.kind[:2]}{oid}_{var}", th.kind, oid, var))
        self._cols += [r[0] for r in self._node_req]
        self._cols += [r[0] for r in self._part_req]
        self._cols += [r[0] for r in self._sect_req]
        self._cols += [r[0] for r in self._other_req]
        self._fh.write("# pyradioss time history (T01 equivalent)\n")
        self._fh.write(",".join(self._cols) + "\n")

    # ------------------------------------------------------------------
    def _part_value(self, pid: int, var: str) -> float:
        """IE/KE of one part, summed over its element groups."""
        model = self.model
        val = 0.0
        var_upper = (var or "").strip().upper()
        for _, group in model.element_groups():
            part_ids = group.state.get("part_ids")
            if part_ids is None:
                continue
            mask = part_ids == pid
            if not np.any(mask):
                continue
            if var_upper == "IE":
                eint = group.state.get("eint")
                if eint is not None:
                    val += float(eint[mask].sum())
            elif var_upper == "KE":
                # kinetic energy of the element masses: 1/2 m_e <v^2>_nodes
                # (beams store a reduced 'mass_conn' — their 3rd node is
                # orientation only and carries no mass)
                conn = group.state.get("mass_conn", group.conn)
                if conn is None or conn.size == 0 or conn.shape[1] == 0:
                    continue
                safe_conn = np.clip(conn[mask], 0, max(0, len(model.v) - 1))
                ve = model.v[safe_conn]
                vre = None
                vr = getattr(model, "vr", None)
                if vr is not None and len(vr) > 0 and len(vr) >= len(model.v):
                    vre = vr[safe_conn]
                # Corner shares honour a kernel's PTG angle table when
                # present (shell_lump); otherwise the uniform formulas.
                ke_trans, ke_rot = corner_ke(group.state, ve, vre, mask)
                val += float(ke_trans + ke_rot)
        return val

    def _other_value(self, kind: str, oid: int, var: str) -> float:
        model = self.model
        var_upper = var.upper()
        comp = {"X": 0, "Y": 1, "Z": 2}.get(var_upper[-1], 0)

        if kind == "RBODY":
            rbs = getattr(model, "rigid_bodies", None)
            rb = rbs.get(oid) if rbs is not None else None
            if rb is not None:
                if var_upper.startswith("D"):
                    if hasattr(rb, "x_cg0"):
                        disp = rb.x_cg - rb.x_cg0
                        if var_upper[-1] in ("X", "Y", "Z"):
                            return float(disp[comp])
                        return float(np.linalg.norm(disp))
                    return 0.0
                elif var_upper.startswith("V"):
                    if hasattr(rb, "v_cg"):
                        if var_upper[-1] in ("X", "Y", "Z"):
                            return float(rb.v_cg[comp])
                        return float(np.linalg.norm(rb.v_cg))
                    return 0.0
                elif var_upper.startswith("W") or var_upper.startswith("ROT"):
                    if hasattr(rb, "w"):
                        if var_upper[-1] in ("X", "Y", "Z"):
                            return float(rb.w[comp])
                        return float(np.linalg.norm(rb.w))
                    return 0.0
                elif var_upper.startswith("F"):
                    if hasattr(rb, "f_res"):
                        if var_upper[-1] in ("X", "Y", "Z"):
                            return float(rb.f_res[comp])
                        return float(np.linalg.norm(rb.f_res))
                    return 0.0
                elif var_upper.startswith("M"):
                    if hasattr(rb, "m_res"):
                        if var_upper[-1] in ("X", "Y", "Z"):
                            return float(rb.m_res[comp])
                        return float(np.linalg.norm(rb.m_res))
                    return 0.0

        elif kind in ("SPRING", "SPRI"):
            g = getattr(model, "springs", None)
            if g is not None and oid in g.ids:
                row = np.where(g.ids == oid)[0]
                if len(row):
                    r = row[0]
                    st = g.state
                    if var_upper in ("F", "FORCE"):
                        if "fres" in st:
                            f = st["fres"][r]
                            return float(np.linalg.norm(f) if hasattr(f, "__len__") else f)
                        return float(st.get("force", np.zeros(len(g.ids)))[r])
                    elif var_upper in ("FX", "FY", "FZ"):
                        c = {"FX": 0, "FY": 1, "FZ": 2}[var_upper]
                        if "fres" in st and st["fres"].ndim > 1 and st["fres"].shape[1] > c:
                            return float(st["fres"][r, c])
                        if "force" in st:
                            n1, n2 = g.conn[r, 0], g.conn[r, 1]
                            dx = model.x[n2] - model.x[n1]
                            L = np.linalg.norm(dx)
                            if L > 1e-20:
                                return float(st["force"][r] * (dx[c] / L))
                        return 0.0
                    elif var_upper in ("D", "DISP", "DL"):
                        if "disp" in st:
                            return float(st["disp"][r])
                        if "L" in st and "L0" in st:
                            return float(st["L"][r] - st["L0"][r])
                        n1, n2 = g.conn[r, 0], g.conn[r, 1]
                        L = np.linalg.norm(model.x[n2] - model.x[n1])
                        L0 = np.linalg.norm(model.x0[n2] - model.x0[n1])
                        return float(L - L0)
                    elif var_upper in ("DX", "DY", "DZ"):
                        c = {"DX": 0, "DY": 1, "DZ": 2}[var_upper]
                        n1, n2 = g.conn[r, 0], g.conn[r, 1]
                        dx = (model.x[n2, c] - model.x0[n2, c]) - (model.x[n1, c] - model.x0[n1, c])
                        return float(dx)
                    elif var_upper in ("E", "IE", "ENERGY"):
                        return float(st["eint"][r]) if "eint" in st else 0.0

        elif kind in ("SHEL", "SHELL", "BRIC", "BRICK", "SH3N", "TETR", "TETRA", "QUAD", "BEAM", "TRUS", "TRUSS", "SOLID", "TSHELL", "PENTA", "PENTA6", "WEDGE"):
            for attr in ("shells", "shells_qbat", "shells_qeph", "bricks", "bricks_heph", "tshells", "tetras", "sh3n", "sh3n_dkt18", "bric20s", "penta6s", "shel16s", "tetra10s", "quads", "beams", "trusses"):
                g = getattr(model, attr, None)
                if g is not None and oid in g.ids:
                    row = np.where(g.ids == oid)[0]
                    if len(row):
                        r = row[0]
                        st = g.state
                        if var_upper in ("IE", "EINT", "ENERGY"):
                            return float(st["eint"][r]) if "eint" in st else 0.0
                        elif var_upper in ("EPSP", "PLASTIC_STRAIN"):
                            ep = st.get("epsp")
                            if ep is not None:
                                return float(ep[r].max() if ep[r].ndim > 0 else ep[r])
                        elif var_upper in ("VM", "VONM", "VON_MISES"):
                            from .anim_vtk import _von_mises
                            return float(_von_mises(attr, g)[r])
                        elif var_upper in ("N", "FX", "FY", "FZ"):
                            if "fres" in st:
                                f = st["fres"]
                                c = {"N": 0, "FX": 0, "FY": 1, "FZ": 2}[var_upper]
                                if f.ndim > 1 and f.shape[1] > c:
                                    return float(f[r, c])
                                elif f.ndim == 1 and c == 0:
                                    return float(f[r])
                            return 0.0
                        elif var_upper in ("MX", "MY", "MZ"):
                            c = {"MX": 0, "MY": 1, "MZ": 2}[var_upper]
                            if "mres" in st:
                                m = st["mres"]
                                if m.ndim > 1 and m.shape[1] > c:
                                    return float(m[r, c])
                            elif "fres" in st:
                                f = st["fres"]
                                if f.ndim > 1 and f.shape[1] > (3 + c):
                                    return float(f[r, 3 + c])
                            return 0.0
                        elif var_upper in ("F", "FORCE"):
                            if "fres" in st:
                                f = st["fres"][r]
                                return float(np.linalg.norm(f) if hasattr(f, "__len__") else f)
                            return 0.0
                        elif var_upper in ("M", "MOM", "MOMENT"):
                            if "mres" in st:
                                m = st["mres"][r]
                                return float(np.linalg.norm(m) if hasattr(m, "__len__") else m)
                            return 0.0
                        elif var_upper in _STRESS_MAP:
                            c_idx = _STRESS_MAP[var_upper]
                            sig = st.get("sig")
                            if sig is not None:
                                s_elem = sig[r]
                                if s_elem.ndim > 1:
                                    s_elem = s_elem.mean(axis=0)
                                if np.ndim(s_elem) == 0:
                                    if var_upper in ("P", "PRESSURE"):
                                        return float(-s_elem / 3.0)
                                    return float(s_elem) if c_idx == 0 else 0.0
                                if len(s_elem) == 3:
                                    shell_map = {0: 0, 1: 1, 3: 2}
                                    if c_idx in shell_map:
                                        return float(s_elem[shell_map[c_idx]])
                                    return 0.0  # SIGZZ, SIGYZ, SIGZX are 0.0 in plane stress
                                else:
                                    return float(s_elem[c_idx])
                        elif var_upper in ("P", "PRESSURE"):
                            sig = st.get("sig")
                            if sig is not None:
                                s_elem = sig[r]
                                if s_elem.ndim > 1:
                                    s_elem = s_elem.mean(axis=0)
                                if len(s_elem) == 3:
                                    return float(-(s_elem[0] + s_elem[1]) / 3.0)
                                return float(-np.mean(s_elem[:3]))
        return 0.0

    def write(self, t, energies, mass, momentum, sect_values=None) -> None:
        """``sect_values``: {sect_id: (F (3,), M (3,))} from
        SectionForces.compute — required only when /TH/SECT was asked."""
        model = self.model
        row = [t, energies["IE"], energies["KE"], energies["HE"],
               energies["CE"], energies["EN"], energies["DE"],
               energies["EW"], energies["ERR"], mass,
               momentum[0], momentum[1], momentum[2]]
        disp = None
        for _, idx, var in self._node_req:
            var_upper = (var or "").strip().upper()
            comp = {"X": 0, "Y": 1, "Z": 2}.get(var_upper[-1] if var_upper else "", 0)
            if var_upper.startswith("D"):
                if disp is None:
                    disp = model.x - model.x0
                row.append(disp[idx, comp])
            elif var_upper.startswith("V"):
                row.append(model.v[idx, comp])
            elif var_upper.startswith("A"):
                if hasattr(model, "a") and model.a is not None:
                    row.append(model.a[idx, comp])
                else:
                    m = model.mass[idx]
                    fext_val = model.fext[idx, comp] if hasattr(model, "fext") and model.fext is not None else 0.0
                    fint_val = model.fint[idx, comp] if hasattr(model, "fint") and model.fint is not None else 0.0
                    row.append((fext_val + fint_val) / m if m > 0 else 0.0)
            elif var_upper.startswith("F"):
                row.append(model.fint[idx, comp] if hasattr(model, "fint") and model.fint is not None else 0.0)
            elif var_upper in ("X", "Y", "Z", "COORDX", "COORDY", "COORDZ"):
                row.append(float(model.x[idx, comp]))
            else:
                row.append(0.0)
        for _, pid, var in self._part_req:
            row.append(self._part_value(pid, var))
        for _, sid, var in self._sect_req:
            var_upper = (var or "").strip().upper()
            comp = {"X": 0, "Y": 1, "Z": 2}.get(var_upper[-1] if var_upper else "", 0)
            if sect_values is None or sid not in sect_values:
                row.append(0.0)
            else:
                F, M = sect_values[sid]
                if var_upper.startswith("F"):
                    row.append(F[comp])
                elif var_upper.startswith("M"):
                    row.append(M[comp])
                else:
                    row.append(0.0)
        for _, kind, oid, var in self._other_req:
            row.append(self._other_value(kind, oid, var))
        self._fh.write(",".join(f"{x:.9E}" for x in row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if hasattr(self, "_fh") and not self._fh.closed:
            self._fh.close()
