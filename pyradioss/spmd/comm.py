"""
The SPMD communicator layer — the port of ``engine/source/mpi/spmd_mod.F90``
and the generic wrappers around it (``engine/source/mpi/generic/`` and the
``spmd_send/recv/isend/irecv/wait/allreduce/bcast/gather/barrier`` .F90
files next to ``spmd_mod.F90``).

Fortran origin
--------------
OpenRadioss wraps every MPI call in a thin ``SPMD_*`` layer so the whole
engine can be compiled with or without ``-DMPI``:

* ``spmd_mod.F90``          — the module re-exporting the point-to-point
  (``spmd_isend``, ``spmd_irecv``, ``spmd_wait``...) and collective
  (``spmd_allreduce``, ``spmd_bcast``, ``spmd_gather``, ``spmd_barrier``
  ...) wrappers, plus ``spmd_comm_size``/``spmd_comm_rank``;
* ``spmd_comm_world.F90``   — ``SPMD_COMM_WORLD``, the communicator every
  exchange runs on (an ``MPI_COMM_SPLIT`` of ``MPI_COMM_WORLD`` by the
  ``MPI_APPNUM`` colour, so a coupled multi-program launch keeps its own
  world — ``init/inipar.F`` ICAS=1);
* ``spmd_operator.F90``     — the ``SPMD_MAX/MIN/SUM/PROD`` operator ids;
* ``init/inipar.F``         — process bootstrap: ``ISPMD`` (0-based rank),
  ``NSPMD`` (domain count), ``IT_SPMD(I) = I-1`` (rank of domain I), the
  ``NSPMD /= NNODES`` "REQUIRED (number of .rst files) NSPMD" check, the
  hostname gather that computes ``L_SPMD``/``NSPMD_PER_NODE``, and the
  ICAS=2 ``MPI_FINALIZE``;
* ``generic/glob_min.F``    — the user-defined ``MPI_OP`` behind
  ``spmd_glob_min5.F`` (the per-cycle global time-step reduction): slot 1
  is MINimised and slots 2-3 (critical element type/number) follow the
  slot-1 winner, slots 4, 5 and 7 are SUMmed, slot 6 MINimised and slots
  8-10 MAXimised (the ``MSTOP`` stop flags).

The three backends
------------------
The Fortran has exactly two build flavours: ``-DMPI`` (real MPI) and the
serial build, where every ``SPMD_*`` routine is an empty ``#else`` branch
and ``NSPMD`` must be 1 (``inipar.F``'s "NON HYBRID EXECUTABLE ONLY
SUPPORTS ONE SPMD DOMAIN"). The port keeps both and adds a third:

* :class:`SerialComm`   — size 1, every collective is the identity, every
  point-to-point is a programming error (there is nobody to talk to).
  This is what a plain ``pyradioss-engine`` run uses, so a serial run
  pays NOTHING for the SPMD layer (``NSPMD == 1`` short-circuits every
  hook in the engine, exactly like the ``IF(NSPMD>1)`` guards in
  ``resol.F``).
* :class:`Mpi4pyComm`   — the real thing over ``mpi4py``, one Python
  process per domain launched with ``mpirun -np N python -m
  pyradioss.engine -i RunName_0001.rad``.  Buffered (upper-case)
  ``Isend/Irecv/Allreduce`` on NumPy arrays for the per-cycle frontier
  traffic; pickled (lower-case) collectives for the small control
  packets.
* :class:`ThreadComm`   — the in-process fallback: the N domains run as
  N Python threads of ONE process, exchanging through queues and a
  barrier.  It exists so a decomposed run works on a machine without an
  MPI library (``-np N`` without ``mpirun``) and so the SPMD algebra can
  be tested deterministically inside pytest; it is NOT a performance
  path (the GIL serialises the Python parts of the cycle, only the NumPy
  kernels overlap).  There is no Fortran counterpart — the serial build
  simply refuses ``NSPMD > 1``.

Message tags
------------
Every Fortran exchange routine carries its own ``MSGOFF`` message-type
base (``spmd_exch_a.F`` 120, ``spmd_exch_v.F`` 7000, ``spmd_fl_sum.F``
186/187, ``spmd_exch_a_rb6.F`` 165 ...) so concurrent exchanges never
match each other's receives.  The port keeps the same discipline:
``exchange.py`` passes the Fortran tag of the routine it ports.
"""

from __future__ import annotations

import threading
import queue
from typing import Any, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Operator ids — spmd_operator.F90 (SPMD_MAX / SPMD_MIN / SPMD_SUM / SPMD_PROD)
# ---------------------------------------------------------------------------
SPMD_MAX = "max"
SPMD_MIN = "min"
SPMD_SUM = "sum"
SPMD_PROD = "prod"

_NP_OPS = {
    SPMD_MAX: np.maximum,
    SPMD_MIN: np.minimum,
    SPMD_SUM: np.add,
    SPMD_PROD: np.multiply,
}


class Request:
    """A non-blocking operation handle (the ``REQ_S``/``REQ_R`` integers
    of the Fortran, ``spmd_wait.F90`` consumes them)."""

    __slots__ = ("_wait", "_done", "_result")

    def __init__(self, wait_fn):
        self._wait = wait_fn
        self._done = False
        self._result = None

    def wait(self):
        if not self._done:
            self._result = self._wait()
            self._done = True
        return self._result


class Comm:
    """Abstract communicator: the subset of ``spmd_mod.F90`` the engine
    needs. ``rank`` is ``ISPMD`` (0-based), ``size`` is ``NSPMD``."""

    rank: int = 0
    size: int = 1
    kind: str = "serial"

    # ---- point-to-point (spmd_isend.F90 / spmd_irecv.F90 / spmd_wait.F90)
    def isend(self, buf: np.ndarray, dest: int, tag: int) -> Request:
        raise NotImplementedError

    def irecv(self, buf: np.ndarray, src: int, tag: int) -> Request:
        """Post a receive INTO ``buf`` (a preallocated contiguous array,
        like the Fortran ``RBUF(L:L+SIZ-1)`` slice). ``wait()`` returns
        ``buf`` once filled."""
        raise NotImplementedError

    def send(self, buf: np.ndarray, dest: int, tag: int) -> None:
        self.isend(buf, dest, tag).wait()

    def recv(self, buf: np.ndarray, src: int, tag: int) -> np.ndarray:
        return self.irecv(buf, src, tag).wait()

    # ---- collectives ------------------------------------------------------
    def allreduce(self, arr: np.ndarray, op: str = SPMD_SUM) -> np.ndarray:
        """``spmd_allreduce.F90``: element-wise reduction of a NumPy array,
        result returned on every rank (a NEW array; the input is not
        modified — the Fortran uses ``MPI_IN_PLACE`` in
        ``spmd_glob_rsum_poff.F``, the port keeps inputs immutable)."""
        raise NotImplementedError

    def allgather(self, obj: Any) -> list:
        """``spmd_allgather.F90`` on an arbitrary picklable object:
        returns ``[obj_rank0, obj_rank1, ...]`` on every rank."""
        raise NotImplementedError

    def bcast(self, obj: Any, root: int = 0) -> Any:
        """``spmd_bcast.F90`` (pickled object flavour)."""
        raise NotImplementedError

    def gather(self, obj: Any, root: int = 0) -> Optional[list]:
        """``spmd_gather.F90`` on picklable objects: the list on ``root``,
        ``None`` elsewhere."""
        raise NotImplementedError

    def barrier(self) -> None:
        """``spmd_barrier.F90``."""
        raise NotImplementedError

    def abort(self, code: int = 2) -> None:
        """``MPI_ABORT`` (``inipar.F`` ICAS=3 mismatch path, ``spmd_kill``)."""
        raise SystemExit(code)

    # ---- conveniences -----------------------------------------------------
    @property
    def active(self) -> bool:
        """``NSPMD > 1`` — the guard on every engine hook."""
        return self.size > 1

    @property
    def is_root(self) -> bool:
        """``ISPMD == 0`` — the domain that owns the listing and the
        outputs (``spmd_chkw.F``/``spmd_wiout.F`` route every other
        domain's writes through it)."""
        return self.rank == 0

    def allreduce_scalar(self, value: float, op: str = SPMD_SUM) -> float:
        return float(self.allreduce(np.asarray([value], dtype=np.float64), op)[0])

    def glob_min(self, packet: Sequence[float]) -> np.ndarray:
        """The ``GLOB_MIN`` user operator of ``generic/glob_min.F`` applied
        to one 10-slot packet (``spmd_glob_min5.F`` sends exactly one).

        Slot semantics (1-based in the Fortran, 0-based here):

        ==== ================ ===========================================
        slot Fortran           reduction
        ==== ================ ===========================================
        0    DT2              MIN
        1    ITYPTS           follows the slot-0 minimum (critical element
        2    NELTS            type and number of the winning domain)
        3    IEXICODT         SUM
        4    IMSCH            SUM
        5    TSTOP            MIN
        6    IWIOUT           SUM
        7    MSTOP1           MAX
        8    MSTOP2           MAX
        9    ISMSCH           MAX
        ==== ================ ===========================================

        The reduction is performed in RANK ORDER on every domain (an
        ``allgather`` of the packets followed by the same serial loop),
        which makes the result bitwise identical everywhere — the
        property the Fortran relies on implicitly (every domain must take
        the same next step, MPI_ALLREDUCE guarantees it for the built-in
        ops and the commutative user op).  Ties on slot 0 resolve to the
        HIGHEST rank: the Fortran's ``IF(RIN(1) == RINOUT(1))`` test runs
        AFTER the MIN, so an exact tie takes the incoming packet, and the
        port visits the ranks in increasing order (see _glob_min_pair)."""
        pk = np.asarray(packet, dtype=np.float64)
        if pk.shape != (10,):
            raise ValueError("glob_min packet must have 10 slots")
        if self.size == 1:
            return pk.copy()
        allp = self.allgather(pk)
        out = allp[0].copy()
        for rin in allp[1:]:
            _glob_min_pair(rin, out)
        return out


def _glob_min_pair(rin: np.ndarray, rinout: np.ndarray) -> None:
    """One application of ``GLOB_MIN(RIN, RINOUT, 1, TYPE)`` — the exact
    slot-by-slot rule of ``generic/glob_min.F`` (VLEN = 10)."""
    rinout[0] = min(rinout[0], rin[0])
    if rin[0] == rinout[0]:
        # the incoming packet holds the (new or equal) minimum: its
        # critical element wins.  NOTE the Fortran compares AFTER the
        # min, so an exact tie takes the incoming (later-visited) one;
        # MPI applies the op in an unspecified order, the port visits
        # ranks in increasing order — a tie therefore resolves to the
        # HIGHEST rank here.  Documented; ties on a float dt are
        # measure-zero events in practice.
        rinout[1] = rin[1]
        rinout[2] = rin[2]
    rinout[3] += rin[3]
    rinout[4] += rin[4]
    rinout[5] = min(rinout[5], rin[5])
    rinout[6] += rin[6]
    rinout[7] = max(rinout[7], rin[7])
    rinout[8] = max(rinout[8], rin[8])
    rinout[9] = max(rinout[9], rin[9])


# ===========================================================================
class SerialComm(Comm):
    """``NSPMD == 1``: the serial build.  Every ``SPMD_*`` routine of the
    Fortran compiles to nothing in this configuration; here the
    collectives are the identity and point-to-point is refused."""

    rank = 0
    size = 1
    kind = "serial"

    def isend(self, buf, dest, tag):
        raise RuntimeError("SerialComm: no peer to send to")

    def irecv(self, buf, src, tag):
        raise RuntimeError("SerialComm: no peer to receive from")

    def allreduce(self, arr, op=SPMD_SUM):
        return np.array(arr, dtype=np.asarray(arr).dtype, copy=True)

    def allgather(self, obj):
        return [obj]

    def bcast(self, obj, root=0):
        return obj

    def gather(self, obj, root=0):
        return [obj]

    def barrier(self):
        return None


SERIAL = SerialComm()


# ===========================================================================
class _ThreadWorld:
    """The shared state of one ``ThreadComm`` group (the analogue of
    ``SPMD_COMM_WORLD`` for the in-process backend): per-(src, dst, tag)
    message queues and the collective rendezvous."""

    def __init__(self, size: int):
        self.size = size
        self._lock = threading.Lock()
        self._queues = {}
        self._barrier = threading.Barrier(size)
        self._slots: List[Any] = [None] * size
        self.error: Optional[BaseException] = None

    def q(self, src: int, dst: int, tag: int) -> queue.Queue:
        key = (src, dst, tag)
        with self._lock:
            qq = self._queues.get(key)
            if qq is None:
                qq = self._queues[key] = queue.Queue()
        return qq

    def collective(self, rank: int, payload: Any, reduce_fn):
        """Rendezvous: every thread deposits ``payload`` in its slot, all
        wait, every thread computes ``reduce_fn(slots)`` on the SAME
        sequence (deterministic on every rank), all wait again before the
        slots are reused.  A thread that died breaks the barrier for the
        others (``threading.BrokenBarrierError``) so a crash in one
        domain aborts the whole run instead of hanging it — the role of
        ``MPI_ABORT`` in ``spmd_kill.F``."""
        self._slots[rank] = payload
        self._barrier.wait()
        try:
            result = reduce_fn(list(self._slots))
        finally:
            self._barrier.wait()
        return result

    def abort_all(self):
        self._barrier.abort()


class ThreadComm(Comm):
    """One domain of an in-process SPMD run (see module docstring).

    Create the whole group with :meth:`ThreadComm.group` and give each
    thread its member."""

    kind = "threads"

    def __init__(self, world: _ThreadWorld, rank: int):
        self._w = world
        self.rank = rank
        self.size = world.size

    @classmethod
    def group(cls, size: int) -> List["ThreadComm"]:
        world = _ThreadWorld(size)
        return [cls(world, r) for r in range(size)]

    # ---- point-to-point ---------------------------------------------------
    def isend(self, buf, dest, tag):
        # a COPY is queued: the Fortran ISEND semantics let the caller
        # reuse SBUF only after the wait, but the port's callers pack a
        # fresh buffer per exchange anyway; copying makes the queue own
        # its data whatever the caller does next.
        self._w.q(self.rank, dest, tag).put(np.array(buf, copy=True))
        return Request(lambda: None)

    def irecv(self, buf, src, tag):
        qq = self._w.q(src, self.rank, tag)

        def _wait():
            data = qq.get()
            if data.shape != buf.shape:
                raise RuntimeError(
                    f"ThreadComm rank {self.rank}: message from {src} tag "
                    f"{tag} has shape {data.shape}, expected {buf.shape}")
            buf[...] = data
            return buf
        return Request(_wait)

    # ---- collectives ------------------------------------------------------
    def allreduce(self, arr, op=SPMD_SUM):
        a = np.array(arr, copy=True)
        fn = _NP_OPS[op]

        def _reduce(slots):
            out = np.array(slots[0], copy=True)
            for s in slots[1:]:
                out = fn(out, s)
            return out
        return self._w.collective(self.rank, a, _reduce)

    def allgather(self, obj):
        return self._w.collective(self.rank, obj, lambda slots: list(slots))

    def bcast(self, obj, root=0):
        return self._w.collective(self.rank, obj, lambda slots: slots[root])

    def gather(self, obj, root=0):
        res = self._w.collective(self.rank, obj, lambda slots: list(slots))
        return res if self.rank == root else None

    def barrier(self):
        self._w.collective(self.rank, None, lambda slots: None)

    def abort(self, code=2):
        self._w.abort_all()
        raise SystemExit(code)


# ===========================================================================
class Mpi4pyComm(Comm):
    """``-DMPI`` build over ``mpi4py``.  Imports mpi4py lazily so the
    package stays importable without it (``pyproject`` optional extra
    ``mpi``)."""

    kind = "mpi"

    def __init__(self, comm=None):
        try:
            from mpi4py import MPI
        except (ImportError, OSError, RuntimeError) as exc:
            # P0.7: mpi4py is LAZY — `import mpi4py` succeeds with no MPI
            # runtime at all; libmpi is resolved when `mpi4py.MPI` is first
            # imported.  With mpi4py 4.x and no MPI library that import
            # raises RuntimeError("cannot load MPI library") (measured on
            # the 4.1.2 wheel, 2026-10-03); mpi4py 3.x raises ImportError
            # from the missing/stub extension.  All three are "no usable
            # MPI here", so turn them into ONE actionable error instead of
            # an opaque traceback out of driver.run_engine_spmd:89-91.
            raise RuntimeError(
                "mpi4py's MPI module is not usable here "
                f"({type(exc).__name__}: {exc}): either mpi4py is not "
                "installed, or it is installed with no MPI library "
                "reachable. Install an MPI implementation (e.g. "
                "`apt-get install mpich`, or "
                "`conda install -c conda-forge mpich`), or launch without "
                "mpirun so the N domains run as threads of one process "
                "(spmd.comm.ThreadComm).") from exc
        self._MPI = MPI
        if comm is None:
            # inipar.F ICAS=1: SPMD_COMM_WORLD = MPI_COMM_SPLIT(WORLD, colour=
            # MPI_APPNUM) so a multi-program mpirun (coupling) keeps the
            # Radioss processes in their own world.
            world = MPI.COMM_WORLD
            colour = world.Get_attr(MPI.APPNUM)
            colour = int(colour) if colour is not None else 0
            comm = world.Split(colour, world.Get_rank())
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self._ops = {SPMD_MAX: MPI.MAX, SPMD_MIN: MPI.MIN,
                     SPMD_SUM: MPI.SUM, SPMD_PROD: MPI.PROD}

    def isend(self, buf, dest, tag):
        buf = np.ascontiguousarray(buf)
        req = self.comm.Isend(buf, dest=dest, tag=tag)
        # keep the buffer alive until the wait (MPI reads it lazily)
        return Request(lambda b=buf, r=req: r.Wait())

    def irecv(self, buf, src, tag):
        if not buf.flags["C_CONTIGUOUS"]:
            raise ValueError("irecv needs a C-contiguous receive buffer")
        req = self.comm.Irecv(buf, source=src, tag=tag)

        def _wait():
            req.Wait()
            return buf
        return Request(_wait)

    def allreduce(self, arr, op=SPMD_SUM):
        a = np.ascontiguousarray(arr)
        out = np.empty_like(a)
        self.comm.Allreduce(a, out, op=self._ops[op])
        return out

    def allgather(self, obj):
        return self.comm.allgather(obj)

    def bcast(self, obj, root=0):
        return self.comm.bcast(obj, root=root)

    def gather(self, obj, root=0):
        return self.comm.gather(obj, root=root)

    def barrier(self):
        self.comm.Barrier()

    def abort(self, code=2):
        self.comm.Abort(code)

    def hostname_layout(self):
        """``inipar.F`` ICAS=3: the hostname gather that fills ``L_SPMD``
        (local index of the domain on its host) and ``NSPMD_PER_NODE``.
        Returned as (l_spmd list, per_node count) — informational only
        in the port (the Fortran uses them for the hybrid MPI/OpenMP
        thread pinning)."""
        MPI = self._MPI
        names = self.comm.allgather(MPI.Get_processor_name())
        order = sorted(range(self.size), key=lambda i: (names[i], i))
        l_spmd = [0] * self.size
        host_of = [0] * self.size
        prev, local, host = None, 0, 0
        for i in order:
            if names[i] != prev:
                prev, local, host = names[i], 0, host + 1
            l_spmd[i] = local
            host_of[i] = host
            local += 1
        per_node = sum(1 for h in host_of if h == host_of[self.rank])
        return l_spmd, per_node


# ===========================================================================
def mpi_world_size() -> int:
    """Size of ``MPI_COMM_WORLD`` if this process was started under
    mpirun with mpi4py importable, else 1 — without initializing MPI
    when it is not needed (importing mpi4py.MPI initializes it).

    P0.7: "no usable MPI" is three different exception types depending on
    the mpi4py major version — ImportError (not installed, or 3.x's missing
    extension), OSError (a libmpi that will not dlopen) and RuntimeError
    (4.x's ABI finder: "cannot load MPI library").  All of them mean the
    same thing HERE, namely "not running under a working MPI" — so all of
    them fall back to 1, which makes driver.run_engine_spmd:89 take the
    ThreadComm branch.  A bare ImportError handler leaked the other two.
    """
    import os
    # the launchers set one of these before the process starts
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MPI_LOCALNRANKS",
                "PMIX_RANK", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        if var in os.environ:
            try:
                from mpi4py import MPI  # noqa: F401
            except (ImportError, OSError, RuntimeError):
                return 1
            return MPI.COMM_WORLD.Get_size()
    return 1
