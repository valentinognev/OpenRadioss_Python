"""
Fortran-compatible binary restart I/O for OpenRadioss Python.

Ported from OpenRadioss upstream files:
- engine/source/output/restart/wrrestp.F (ENGINE restart output)
- engine/source/output/restart/rdresa.F (ENGINE restart header reading)
- engine/source/output/restart/rdresb.F (ENGINE restart array reading)
- starter/source/restart/ddsplit/wrrest.F (STARTER restart output)
- starter/source/output/tools/wrtsqi.F (Sequential unformatted integer writer)
- starter/source/output/tools/wrtsqr.F (Sequential unformatted real writer)
- common_source/tools/input_output/write_routines.c (Record markers & low-level I/O)
- engine/share/includes/scr03_c.inc (IRADIOS=2143942393, IRESFIL=101)

Fortran unformatted sequential files use 4-byte integer record markers:
each WRITE statement outputs:
  [4-byte int: record_length] [record_bytes] [4-byte int: record_length]
"""

from __future__ import annotations

import os
import pickle
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np

from .. import __version__
from ..model.model import ElementGroup, Model

# Upstream Fortran magic constants from engine/share/includes/scr03_c.inc:59,61
# Ported from $OR_SRC/engine/share/includes/scr03_c.inc
IRADIOS: int = 2143942393
IRESFIL: int = 101

# Binary restart file identifier (exactly 20 bytes for Fortran record)
# Ported from $OR_SRC/engine/source/output/restart/wrrestp.F
_MAGIC_BINARY: bytes = b"OPENRADIOSS-RST-BIN\x00"

_ELEMENT_GROUP_NAMES = {
    "bricks", "bricks_full", "bricks_eas", "bricks_heph", "solid_shells_ha8", "cohesives",
    "tshells", "bric20s", "penta6s", "penta6s_heph", "pyra5s",
    "quads", "quads_full", "trias",
    "tetras", "tetras_sfem", "tetra10s",
    "shel16s", "thickshell_wedges", "thickshell_composites",
    "shells", "shells_qbat", "shells_qeph",
    "sh3n", "sh3n_dkt18", "shells_dkt6",
    "trusses", "springs", "beams", "beams_fiber",
}

_EXCLUDED_ATTRS = {
    "node_ids", "x", "x0", "v", "vr", "a", "mass", "mass0", "inertia", "_id2idx",
    "engine_state",
} | _ELEMENT_GROUP_NAMES


class FortranBinaryFile:
    """Reader/writer for Fortran unformatted sequential binary files.

    Each WRITE statement produces:
    - 4-byte integer (record length in bytes, little-endian)
    - the data payload
    - 4-byte integer (record length again)

    Ported from OpenRadioss:
    - starter/source/output/tools/wrtsqi.F (WRTSQI)
    - starter/source/output/tools/wrtsqr.F (WRTSQR)
    - common_source/tools/input_output/write_routines.c (WRITE_I_C, READ_I_C, WRITE_DB_C)
    """

    def __init__(self, file_or_path: Union[str, Path, BinaryIO], mode: str = "rb") -> None:
        self.mode = mode
        if isinstance(file_or_path, (str, Path)):
            b_mode = mode if "b" in mode else mode + "b"
            self._fh: BinaryIO = open(file_or_path, b_mode)
            self._should_close = True
        else:
            self._fh = file_or_path
            self._should_close = False

    def __enter__(self) -> "FortranBinaryFile":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close underlying file stream if opened by this object."""
        if self._should_close and self._fh is not None:
            self._fh.close()
            self._fh = None

    def write_record(self, data: bytes) -> None:
        """Write one Fortran unformatted sequential record with 4-byte markers.

        Ported from $OR_SRC/starter/source/output/tools/wrtsqi.F
        """
        n = len(data)
        hdr = struct.pack("<i", n)
        self._fh.write(hdr)
        if n > 0:
            self._fh.write(data)
        self._fh.write(hdr)

    def read_record(self) -> bytes:
        """Read one Fortran unformatted sequential record and verify record markers.

        Ported from $OR_SRC/common_source/tools/input_output/write_routines.c
        """
        hdr = self._fh.read(4)
        if not hdr:
            raise EOFError("End of file while reading Fortran record header")
        if len(hdr) < 4:
            raise ValueError(f"Truncated Fortran record header: got {len(hdr)} bytes, expected 4")
        n = struct.unpack("<i", hdr)[0]
        if n < 0:
            raise ValueError(f"Invalid negative record length: {n}")
        data = self._fh.read(n) if n > 0 else b""
        if len(data) < n:
            raise ValueError(f"Truncated record data: expected {n} bytes, got {len(data)}")
        ftr = self._fh.read(4)
        if len(ftr) < 4:
            raise ValueError(f"Truncated Fortran record footer: got {len(ftr)} bytes, expected 4")
        n_trailing = struct.unpack("<i", ftr)[0]
        if n != n_trailing:
            raise ValueError(f"Fortran record marker mismatch: header={n}, footer={n_trailing}")
        return data

    def write_array(self, arr: np.ndarray) -> None:
        """Write numpy array as a Fortran record.

        Ported from $OR_SRC/starter/source/output/tools/wrtsqr.F
        """
        data = np.ascontiguousarray(arr).tobytes()
        self.write_record(data)

    def read_array(
        self,
        dtype: Union[str, np.dtype],
        count: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> np.ndarray:
        """Read numpy array from record with optional shape or element count.

        Ported from $OR_SRC/engine/source/output/restart/rdresb.F
        """
        data = self.read_record()
        arr = np.frombuffer(data, dtype=dtype).copy()
        if count is not None:
            if isinstance(count, (tuple, list)):
                arr = arr.reshape(count)
            elif isinstance(count, (int, np.integer)):
                if count >= 0:
                    arr = arr[:count]
        return arr

    def write_int(self, value: int) -> None:
        """Write a 4-byte signed integer as a Fortran record.

        Ported from $OR_SRC/starter/source/output/tools/wrtsqi.F
        """
        self.write_record(struct.pack("<i", int(value)))

    def read_int(self) -> int:
        """Read an integer from a Fortran record.

        Ported from $OR_SRC/engine/source/output/restart/rdresa.F
        """
        data = self.read_record()
        if len(data) == 4:
            return struct.unpack("<i", data)[0]
        if len(data) == 8:
            return struct.unpack("<q", data)[0]
        return int.from_bytes(data, byteorder="little", signed=True)

    def write_float(self, value: float) -> None:
        """Write an 8-byte double precision float as a Fortran record.

        Ported from $OR_SRC/starter/source/output/tools/wrtsqr.F
        """
        self.write_record(struct.pack("<d", float(value)))

    def read_float(self) -> float:
        """Read an 8-byte float from a Fortran record.

        Ported from $OR_SRC/engine/source/output/restart/rdresb.F
        """
        data = self.read_record()
        if len(data) == 8:
            return struct.unpack("<d", data)[0]
        if len(data) == 4:
            return struct.unpack("<f", data)[0]
        raise ValueError(f"Invalid float record length: {len(data)}")

    def write_string(self, s: str, length: Optional[int] = None) -> None:
        """Write a string as a Fortran record with optional fixed length padding.

        Ported from $OR_SRC/common_source/tools/input_output/write_routines.c
        """
        raw = s.encode("utf-8")
        if length is not None:
            raw = raw[:length].ljust(length, b" ")
        self.write_record(raw)

    def read_string(self, length: Optional[int] = None) -> str:
        """Read a string from a Fortran record.

        Ported from $OR_SRC/engine/source/output/restart/rdresa.F
        """
        data = self.read_record()
        s = data.decode("utf-8", errors="replace")
        if length is not None:
            return s[:length]
        return s.rstrip(" \x00")


def is_binary_restart(peek: bytes) -> bool:
    """Check if the initial bytes of a file indicate a Fortran binary restart."""
    if len(peek) < 24:
        return False
    rec_len = struct.unpack("<i", peek[:4])[0]
    return rec_len == len(_MAGIC_BINARY) and peek[4 : 4 + len(_MAGIC_BINARY)] == _MAGIC_BINARY


def _extract_auxiliary_state(model: Model) -> Dict[str, Any]:
    """Extract non-array entity metadata and state attributes for the auxiliary record.

    Ported from OpenRadioss Python serialization in wrrestp.F (PYTHON_SERIALIZE).
    """
    aux: Dict[str, Any] = {}
    for k, v in model.__dict__.items():
        if k not in _EXCLUDED_ATTRS and not k.startswith("_"):
            aux[k] = v

    # Preserve any non-ndarray values in element group state dicts
    group_non_arrays: Dict[str, Dict[str, Any]] = {}
    for name in _ELEMENT_GROUP_NAMES:
        grp = getattr(model, name, None)
        if grp is not None and hasattr(grp, "state"):
            non_arr: Dict[str, Any] = {}
            for sk, sv in grp.state.items():
                if isinstance(sv, np.ndarray):
                    continue
                if isinstance(sv, dict):
                    non_arr[sk] = {
                        subk: subv for subk, subv in sv.items() if not isinstance(subv, np.ndarray)
                    }
                else:
                    non_arr[sk] = sv
            if non_arr:
                group_non_arrays[name] = non_arr
    if group_non_arrays:
        aux["_group_non_arrays"] = group_non_arrays

    return aux


def _apply_auxiliary_state(model: Model, aux: Dict[str, Any]) -> None:
    """Apply non-array entity metadata from auxiliary record onto Model instance."""
    for k, v in aux.items():
        if k not in ("_group_non_arrays", "_engine_extra"):
            setattr(model, k, v)


def write_restart_binary(model: Model, path: str, engine: Optional[dict] = None) -> None:
    """Write model state in Fortran unformatted sequential binary format.

    Ported from OpenRadioss:
    - engine/source/output/restart/wrrestp.F (WRRESTP)
    - starter/source/restart/ddsplit/wrrest.F (WRREST)
    - starter/source/output/tools/wrtsqi.F (WRTSQI)
    - starter/source/output/tools/wrtsqr.F (WRTSQR)
    - common_source/tools/input_output/write_routines.c
    """
    with FortranBinaryFile(path, "wb") as fb:
        # 1. Header block (magic number, version, format flags)
        # Ported from $OR_SRC/engine/source/output/restart/wrrestp.F:475-508
        fb.write_record(_MAGIC_BINARY)
        has_engine = 1 if engine is not None else 0
        has_aux = 1
        header_ints = np.array([1, 0, has_engine, has_aux, IRADIOS, IRESFIL], dtype=np.int32)
        fb.write_array(header_ints)
        fb.write_string(__version__)
        title = getattr(model, "title", "") or "OPENRADIOSS_MODEL"
        fb.write_string(title)

        # 2. Node block: coordinates, velocities, accelerations, masses
        # Ported from $OR_SRC/engine/source/output/restart/wrrestp.F:523-543
        n_nodes = int(model.numnod) if hasattr(model, "numnod") and model.numnod > 0 else len(getattr(model, "node_ids", []))
        fb.write_int(n_nodes)
        if n_nodes > 0:
            fb.write_array(model.node_ids.astype(np.int64))
            fb.write_array(np.ascontiguousarray(model.x, dtype=np.float64))
            x0 = model.x0 if getattr(model, "x0", None) is not None and len(model.x0) == n_nodes else model.x
            fb.write_array(np.ascontiguousarray(x0, dtype=np.float64))
            v = model.v if getattr(model, "v", None) is not None and len(model.v) == n_nodes else np.zeros((n_nodes, 3))
            fb.write_array(np.ascontiguousarray(v, dtype=np.float64))
            vr = model.vr if getattr(model, "vr", None) is not None and len(model.vr) == n_nodes else np.zeros((n_nodes, 3))
            fb.write_array(np.ascontiguousarray(vr, dtype=np.float64))
            a = getattr(model, "a", None)
            if a is None or len(a) != n_nodes:
                a = np.zeros((n_nodes, 3), dtype=np.float64)
            fb.write_array(np.ascontiguousarray(a, dtype=np.float64))
            mass = model.mass if getattr(model, "mass", None) is not None and len(model.mass) == n_nodes else np.zeros(n_nodes)
            fb.write_array(np.ascontiguousarray(mass, dtype=np.float64))
            mass0 = getattr(model, "mass0", None)
            if mass0 is None or len(mass0) != n_nodes:
                mass0 = mass
            fb.write_array(np.ascontiguousarray(mass0, dtype=np.float64))
            inertia = getattr(model, "inertia", None)
            if inertia is None or len(inertia) != n_nodes:
                inertia = np.zeros(n_nodes, dtype=np.float64)
            fb.write_array(np.ascontiguousarray(inertia, dtype=np.float64))

        # 3. Element block: connectivity, stresses, strains, internal energy
        # Ported from $OR_SRC/engine/source/output/restart/wrrestp.F:550-600
        groups = list(model.element_groups()) if hasattr(model, "element_groups") else []
        fb.write_int(len(groups))
        for name, group in groups:
            fb.write_string(name)
            fb.write_int(int(group.n))
            # Shape of connectivity
            fb.write_array(np.array(group.conn.shape, dtype=np.int64))
            fb.write_array(group.ids.astype(np.int64))
            fb.write_array(group.conn.astype(np.int64))
            part = group.part if hasattr(group, "part") and group.part is not None else np.zeros(group.n, dtype=np.int64)
            fb.write_array(part.astype(np.int64))

            # 4. Material state: plastic strain, damage variables, etc. in group.state
            flat_state: List[Tuple[str, np.ndarray]] = []
            for sk, sv in group.state.items():
                if isinstance(sv, np.ndarray):
                    flat_state.append((sk, sv))
                elif isinstance(sv, dict):
                    for subk, subv in sv.items():
                        if isinstance(subv, np.ndarray):
                            flat_state.append((f"{sk}/{subk}", subv))

            fb.write_int(len(flat_state))
            for sname, sarr in flat_state:
                fb.write_string(sname)
                fb.write_string(str(sarr.dtype))
                fb.write_array(np.array(sarr.shape, dtype=np.int64))
                fb.write_array(sarr)

        # 5. Engine state: time, cycle, dt, energy balances, rigid bodies, sensors
        # Ported from $OR_SRC/engine/source/output/restart/wrrestp.F:360-424
        fb.write_int(has_engine)
        if has_engine and engine is not None:
            t = float(engine.get("t", 0.0))
            cycle = int(engine.get("cycle", 0))
            dt = float(engine.get("dt", 0.0))
            dt_prev = float(engine.get("dt_prev", 0.0) if engine.get("dt_prev") is not None else -1.0)
            wext = float(engine.get("wext", 0.0))
            econt = float(engine.get("econt", 0.0))
            epeak = float(engine.get("epeak", 0.0))
            ndel = int(engine.get("ndel", 0))
            e_num = float(engine.get("e_num", 0.0))
            e_madd = float(engine.get("e_madd", 0.0))
            e_damp = float(engine.get("e_damp", 0.0))
            e0 = float(engine.get("e0", 0.0))
            next_th = float(engine.get("next_th", 0.0))
            next_anim = float(engine.get("next_anim", 0.0))
            anim_no = int(engine.get("anim_no", 0))
            next_state = float(engine.get("next_state", 0.0))

            float_scalars = np.array(
                [t, dt, dt_prev, wext, econt, epeak, e_num, e_madd, e_damp, e0, next_th, next_anim, next_state],
                dtype=np.float64,
            )
            int_scalars = np.array([cycle, ndel, anim_no], dtype=np.int64)
            fb.write_array(float_scalars)
            fb.write_array(int_scalars)

            # Rigid body state: orientation R, angular momentum L, reference velocities
            rbodies = engine.get("rbodies", {}) or {}
            fb.write_int(len(rbodies))
            for rb_id, rb_dict in rbodies.items():
                fb.write_int(int(rb_id))
                fb.write_array(np.ascontiguousarray(rb_dict["R"], dtype=np.float64))
                fb.write_array(np.ascontiguousarray(rb_dict["L"], dtype=np.float64))
                fb.write_array(np.ascontiguousarray(rb_dict["v_ref"], dtype=np.float64))
                fb.write_array(np.ascontiguousarray(rb_dict["w"], dtype=np.float64))
                fb.write_array(np.ascontiguousarray(rb_dict["x_ref"], dtype=np.float64))
                fb.write_array(np.ascontiguousarray(rb_dict["xg"], dtype=np.float64))

            # Sensor state: fire times and status
            sensors = engine.get("sensors", {}) or {}
            sensors_status = engine.get("sensors_status", {}) or {}
            all_sids = sorted(set(sensors.keys()) | set(sensors_status.keys()))
            fb.write_int(len(all_sids))
            for sid in all_sids:
                fb.write_int(int(sid))
                fb.write_float(float(sensors.get(sid, 0.0)))
                fb.write_int(1 if sensors_status.get(sid, False) else 0)

            # Noda (mass scaling) state
            noda_dict = engine.get("noda")
            if noda_dict:
                fb.write_int(1)
                fb.write_float(float(noda_dict.get("mass_added", 0.0)))
                fb.write_float(float(noda_dict.get("iner_added", 0.0)))
                fb.write_float(float(noda_dict.get("e_madd", 0.0)))
                fb.write_float(float(noda_dict.get("mass0", 0.0)))
                fb.write_int(1 if noda_dict.get("reported", False) else 0)
                mom = noda_dict.get("mom_added", np.zeros(3))
                fb.write_array(np.ascontiguousarray(mom, dtype=np.float64))
            else:
                fb.write_int(0)

            # Dynamic relaxation state
            dr_dict = engine.get("dyn_relax")
            if dr_dict:
                fb.write_int(1)
                fb.write_float(float(dr_dict.get("ke_prev", 0.0)))
            else:
                fb.write_int(0)

        # 6. Auxiliary model entities block (analogous to Fortran PYTHON_SERIALIZE in wrrestp.F:545)
        aux = _extract_auxiliary_state(model)
        if has_engine and engine is not None:
            # Preserve any non-standard engine keys
            known_eng_keys = {
                "t", "cycle", "dt", "dt_prev", "wext", "econt", "epeak", "ndel",
                "e_num", "e_madd", "e_damp", "e0", "next_th", "next_anim", "anim_no",
                "next_state", "rbodies", "sensors", "sensors_status", "noda", "dyn_relax",
            }
            extra_eng = {k: v for k, v in engine.items() if k not in known_eng_keys}
            if extra_eng:
                aux["_engine_extra"] = extra_eng

        aux_bytes = pickle.dumps(aux, protocol=pickle.HIGHEST_PROTOCOL)
        fb.write_record(aux_bytes)


def read_restart_binary(path: str) -> Tuple[Model, Optional[dict]]:
    """Read back Model and engine state from a Fortran binary restart file.

    Ported from OpenRadioss:
    - engine/source/output/restart/rdresa.F (RDRESA)
    - engine/source/output/restart/rdresb.F (RDRESB)
    """
    with FortranBinaryFile(path, "rb") as fb:
        # 1. Header block
        magic = fb.read_record()
        if magic != _MAGIC_BINARY:
            raise ValueError(f"{path} is not a valid OpenRadioss binary restart file (magic mismatch)")
        header_ints = fb.read_array(np.int32)
        v_maj, v_min, has_engine, has_aux, iradios, iresfil = header_ints[:6]
        code_ver = fb.read_string()
        title = fb.read_string()

        # 2. Node block
        n_nodes = fb.read_int()
        if n_nodes > 0:
            node_ids = fb.read_array(np.int64, count=n_nodes)
            x = fb.read_array(np.float64, count=(n_nodes, 3))
            x0 = fb.read_array(np.float64, count=(n_nodes, 3))
            v = fb.read_array(np.float64, count=(n_nodes, 3))
            vr = fb.read_array(np.float64, count=(n_nodes, 3))
            a = fb.read_array(np.float64, count=(n_nodes, 3))
            mass = fb.read_array(np.float64, count=n_nodes)
            mass0 = fb.read_array(np.float64, count=n_nodes)
            inertia = fb.read_array(np.float64, count=n_nodes)
        else:
            node_ids = np.zeros(0, dtype=np.int64)
            x = np.zeros((0, 3), dtype=np.float64)
            x0 = np.zeros((0, 3), dtype=np.float64)
            v = np.zeros((0, 3), dtype=np.float64)
            vr = np.zeros((0, 3), dtype=np.float64)
            a = np.zeros((0, 3), dtype=np.float64)
            mass = np.zeros(0, dtype=np.float64)
            mass0 = np.zeros(0, dtype=np.float64)
            inertia = np.zeros(0, dtype=np.float64)

        # 3. Element block
        n_groups = fb.read_int()
        loaded_groups: Dict[str, ElementGroup] = {}
        for _ in range(n_groups):
            gname = fb.read_string()
            n_elem = fb.read_int()
            conn_shape = tuple(fb.read_array(np.int64).tolist())
            g_ids = fb.read_array(np.int64, count=n_elem)
            g_conn = fb.read_array(np.int64, count=conn_shape)
            g_part = fb.read_array(np.int64, count=n_elem)

            n_states = fb.read_int()
            g_state: Dict[str, Any] = {}
            for _ in range(n_states):
                sname = fb.read_string()
                dtype_str = fb.read_string()
                shape = tuple(fb.read_array(np.int64).tolist())
                arr = fb.read_array(dtype_str, count=shape)
                if "/" in sname:
                    parent_k, sub_k = sname.split("/", 1)
                    if parent_k not in g_state or not isinstance(g_state[parent_k], dict):
                        g_state[parent_k] = {}
                    g_state[parent_k][sub_k] = arr
                else:
                    g_state[sname] = arr

            loaded_groups[gname] = ElementGroup(ids=g_ids, conn=g_conn, part=g_part, state=g_state)

        # 4. Engine state block
        engine_flag = fb.read_int()
        engine_dict: Optional[dict] = None
        if engine_flag != 0:
            float_scalars = fb.read_array(np.float64)
            int_scalars = fb.read_array(np.int64)

            t, dt, dt_prev_f, wext, econt, epeak, e_num, e_madd, e_damp, e0, next_th, next_anim, next_state = float_scalars[:13]
            cycle, ndel, anim_no = int_scalars[:3]
            dt_prev = dt_prev_f if dt_prev_f >= 0.0 else None

            # Rigid bodies
            n_rb = fb.read_int()
            rbodies: Dict[int, Dict[str, np.ndarray]] = {}
            for _ in range(n_rb):
                rb_id = fb.read_int()
                R = fb.read_array(np.float64, count=(3, 3))
                L = fb.read_array(np.float64, count=3)
                v_ref = fb.read_array(np.float64, count=3)
                w = fb.read_array(np.float64, count=3)
                x_ref = fb.read_array(np.float64, count=3)
                xg = fb.read_array(np.float64, count=3)
                rbodies[rb_id] = {
                    "R": R, "L": L, "v_ref": v_ref, "w": w, "x_ref": x_ref, "xg": xg
                }

            # Sensors
            n_sensors = fb.read_int()
            sensors: Dict[int, float] = {}
            sensors_status: Dict[int, bool] = {}
            for _ in range(n_sensors):
                sid = fb.read_int()
                ftime = fb.read_float()
                status_int = fb.read_int()
                sensors[sid] = ftime
                sensors_status[sid] = bool(status_int)

            # Noda
            has_noda = fb.read_int()
            noda_dict: Optional[dict] = None
            if has_noda:
                mass_added = fb.read_float()
                iner_added = fb.read_float()
                e_madd_n = fb.read_float()
                mass0_n = fb.read_float()
                rep = fb.read_int()
                mom_added = fb.read_array(np.float64, count=3)
                noda_dict = {
                    "mass_added": mass_added,
                    "iner_added": iner_added,
                    "e_madd": e_madd_n,
                    "mass0": mass0_n,
                    "reported": bool(rep),
                    "mom_added": mom_added,
                }

            # Dyn relax
            has_dr = fb.read_int()
            dr_dict: Optional[dict] = None
            if has_dr:
                ke_prev = fb.read_float()
                dr_dict = {"ke_prev": ke_prev}

            engine_dict = {
                "t": t, "cycle": cycle, "dt": dt, "dt_prev": dt_prev,
                "wext": wext, "econt": econt, "epeak": epeak, "ndel": ndel,
                "e_num": e_num, "e_madd": e_madd, "e_damp": e_damp, "e0": e0,
                "next_th": next_th, "next_anim": next_anim, "anim_no": anim_no,
                "next_state": next_state,
                "rbodies": rbodies,
                "sensors": sensors,
                "sensors_status": sensors_status,
                "noda": noda_dict,
                "dyn_relax": dr_dict or {"ke_prev": 0.0},
            }

        # 5. Auxiliary block
        aux: Dict[str, Any] = {}
        try:
            aux_bytes = fb.read_record()
            aux = pickle.loads(aux_bytes)
        except (EOFError, Exception):
            pass

        # Reconstruct Model
        model = Model()
        model.title = title
        _apply_auxiliary_state(model, aux)

        # Apply binary node data
        model.node_ids = node_ids
        model.x = x
        model.x0 = x0
        model.v = v
        model.vr = vr
        model.a = a
        model.mass = mass
        model.mass0 = mass0
        model.inertia = inertia
        model._id2idx = {int(nid): i for i, nid in enumerate(node_ids)}

        # Apply binary element groups
        group_non_arrays = aux.get("_group_non_arrays", {})
        for gname, grp in loaded_groups.items():
            if gname in group_non_arrays:
                for k, v in group_non_arrays[gname].items():
                    if isinstance(v, dict):
                        grp.state.setdefault(k, {}).update(v)
                    else:
                        grp.state[k] = v
            setattr(model, gname, grp)

        # Merge extra engine keys if any
        if engine_dict is not None and "_engine_extra" in aux:
            engine_dict.update(aux["_engine_extra"])

        return model, engine_dict
