"""
/ACCEL — 4th-order Butterworth digital filter for accelerometers (SAE J211 / ISO 6487).

Upstream Fortran reference:
  - $OR_SRC/engine/source/tools/accele/accel1.F
    Subroutine ACCEL1 (lines 28-117)
  - $OR_SRC/starter/source/tools/accele/lecacc.F
    Subroutine LECACC (lines 43-162)
  - $OR_SRC/engine/source/engine/resol.F
    Accelerometer integration loop (lines 7741-7762)

Theory & Formulation
--------------------
SAE J211-1 and ISO 6487 specify instrumentation for impact tests, defining
Channel Frequency Classes (CFC) for low-pass filtering of acceleration time histories.
OpenRadioss implements this as an in-situ digital filter in SUBROUTINE ACCEL1.

1. Filter Topology:
   Cascaded two-stage second-order Butterworth low-pass digital filter (4th order total):
       H(z) = H1(z) * H2(z)
   Where each stage H_k(z) is a second-order IIR biquad section derived via the
   bilinear transformation with prewarping:
       s = (1 / d) * (1 - z^(-1)) / (1 + z^(-1))
       d = tan(pi * f_c * dt)

2. Cutoff Frequency & Nyquist Capping:
   f_c is the cutoff frequency in Hz.
   To prevent numerical distortion and unstable warping near the Nyquist frequency
   f_nyq = 1 / (2 * dt), OpenRadioss caps the effective cutoff frequency to 80% of
   Nyquist (ZEP4 = 0.4 in OpenRadioss constant_mod.F):
       f_eff = min(f_c, 0.4 / dt)

3. Filter Coefficients (accel1.F lines 56-89):
       d  = tan(pi * f_eff * dt)
       dd = d^2
       d2 = 2 * d
       dp = 1 + dd

   Stage 1 (poles at +/- pi/8 from the real s-axis):
       e1 = d2 * sin(pi / 8)
       g1 = 1 / (e1 + dp)
       c0 = dd * g1
       c1 = 2 * c0
       c2 = c0
       c3 = 2 * g1 - c1
       c4 = (e1 - dp) * g1

   Stage 2 (poles at +/- 3*pi/8 from the real s-axis):
       e2 = d2 * sin(3 * pi / 8)
       g2 = 1 / (e2 + dp)
       c5 = dd * g2
       c6 = 2 * c5
       c7 = c5
       c8 = 2 * g2 - c6
       c9 = (e2 - dp) * g2

4. Direct Form IIR Difference Equations (accel1.F lines 93-110):
   For each spatial acceleration component j in {1, 2, 3}:
       x1 = A0[j, 1]  ! x[n-2]
       x2 = A0[j, 0]  ! x[n-1]
       x3 = a[j]      ! x[n] (current raw acceleration)

       Stage 1 output y3:
       y1 = A1[j, 1]  ! y[n-2]
       y2 = A1[j, 0]  ! y[n-1]
       y3 = c0*x3 + c1*x2 + c2*x1 + c3*y2 + c4*y1

       Stage 2 output z3:
       z1 = A2[j, 1]  ! z[n-2]
       z2 = A2[j, 0]  ! z[n-1]
       z3 = c5*y3 + c6*y2 + c7*y1 + c8*z2 + c9*z1

       State history rotation:
       A0[j, 1] = x2;  A0[j, 0] = x3
       A1[j, 1] = y2;  A1[j, 0] = y3
       A2[j, 1] = z2;  A2[j, 0] = z3

5. Skew Frame Transformation (accel1.F lines 48-50 and 112-114):
   Local coordinate system defined by transformation matrix R (shape (3, 3)):
       a_local = R @ a_global

6. Integrated Velocity (accel1.F lines 51-53):
       v_local += a_local * dt12
   Where dt12 is the time step for velocity integration.

7. Standard SAE J211 Channel Frequency Classes:
   - CFC 60   : f_c = 100.0 Hz  (structural members, vehicle body)
   - CFC 180  : f_c = 300.0 Hz  (thorax, pelvis, vehicle crush zone)
   - CFC 600  : f_c = 1000.0 Hz (head, femur, neck forces)
   - CFC 1000 : f_c = 1650.0 Hz (head acceleration for HIC)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

# Constant capping ratio matching ZEP4 = 0.4 in OpenRadioss constant_mod.F
_ZEP4 = 0.4
_EM20 = 1.0e-20


# ============================================================================
# SAE J211 / ISO 6487 Standard Cutoff Frequencies
# ============================================================================

# SAE J211 / ISO 6487 specification maps Channel Frequency Class (CFC)
# to low-pass 4-pole Butterworth cutoff frequency f_c (Hz):
#   CFC 60   -> 100 Hz
#   CFC 180  -> 300 Hz
#   CFC 600  -> 1000 Hz
#   CFC 1000 -> 1650 Hz
CFC_CUTOFF_FREQUENCIES: Dict[Union[int, str], float] = {
    60: 100.0,
    "60": 100.0,
    "CFC60": 100.0,
    "CFC_60": 100.0,
    180: 300.0,
    "180": 300.0,
    "CFC180": 300.0,
    "CFC_180": 300.0,
    600: 1000.0,
    "600": 1000.0,
    "CFC600": 1000.0,
    "CFC_600": 1000.0,
    1000: 1650.0,
    "1000": 1650.0,
    "CFC1000": 1650.0,
    "CFC_1000": 1650.0,
}


def parse_cfc(cfc: Union[int, str, float]) -> float:
    """Resolve a Channel Frequency Class (CFC) to its cutoff frequency in Hz.

    Parameters
    ----------
    cfc : int, str, or float
        Channel Frequency Class (e.g. 60, 180, 600, 1000, "CFC60", etc.)

    Returns
    -------
    float
        Cutoff frequency f_c in Hz.
    """
    if isinstance(cfc, str):
        key = cfc.strip().upper()
        if key in CFC_CUTOFF_FREQUENCIES:
            return CFC_CUTOFF_FREQUENCIES[key]
        try:
            val = float(key)
            if int(val) in CFC_CUTOFF_FREQUENCIES:
                return CFC_CUTOFF_FREQUENCIES[int(val)]
            return val
        except ValueError:
            raise ValueError(f"Unknown CFC specification: {cfc!r}. Expected one of {list(CFC_CUTOFF_FREQUENCIES.keys())}")
    elif isinstance(cfc, (int, float)):
        int_key = int(cfc)
        if int_key in CFC_CUTOFF_FREQUENCIES:
            return CFC_CUTOFF_FREQUENCIES[int_key]
        return float(cfc)
    raise TypeError(f"Unsupported CFC type: {type(cfc)}")


# ============================================================================
# Filter Coefficients Structure & Calculation
# ============================================================================

class FilterCoefficients(NamedTuple):
    """Filter coefficients for 4th-order cascaded Butterworth filter.

    Matches variables in $OR_SRC/engine/source/tools/accele/accel1.F
    lines 56-89.
    """
    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    c6: float
    c7: float
    c8: float
    c9: float
    fc_eff: float
    dt: float


def compute_filter_coefficients(fc: float, dt: float) -> FilterCoefficients:
    """Compute cascaded 4th-order Butterworth digital filter coefficients.

    Follows SUBROUTINE ACCEL1 (lines 56-89) in:
    $OR_SRC/engine/source/tools/accele/accel1.F

    Parameters
    ----------
    fc : float
        Cut-off frequency in Hz (e.g. 100.0 for CFC60).
    dt : float
        Time step in seconds (dt > 0).

    Returns
    -------
    FilterCoefficients
        Coefficients (c0..c4 for Stage 1, c5..c9 for Stage 2, fc_eff, dt).
    """
    if dt <= 0.0:
        raise ValueError(f"Time step dt must be strictly positive, got {dt}")
    if fc <= 0.0:
        return FilterCoefficients(
            c0=1.0, c1=0.0, c2=0.0, c3=0.0, c4=0.0,
            c5=1.0, c6=0.0, c7=0.0, c8=0.0, c9=0.0,
            fc_eff=0.0, dt=dt,
        )

    # Effective cutoff capped at ZEP4 / dt = 0.4 / dt (80% of Nyquist)
    f_eff = min(float(fc), _ZEP4 / dt)

    pi = math.pi
    pi8 = pi / 8.0
    pi38 = 3.0 * pi8
    spi8 = math.sin(pi8)
    spi38 = math.sin(pi38)

    d = math.tan(pi * f_eff * dt)
    dd = d * d
    d2 = 2.0 * d
    dp = 1.0 + dd

    # Stage 1
    e1 = d2 * spi8
    g1 = 1.0 / (e1 + dp)
    c0 = dd * g1
    c1 = 2.0 * c0
    c2 = c0
    c3 = 2.0 * g1 - c1
    c4 = (e1 - dp) * g1

    # Stage 2
    e2 = d2 * spi38
    g2 = 1.0 / (e2 + dp)
    c5 = dd * g2
    c6 = 2.0 * c5
    c7 = c5
    c8 = 2.0 * g2 - c6
    c9 = (e2 - dp) * g2

    return FilterCoefficients(
        c0=c0, c1=c1, c2=c2, c3=c3, c4=c4,
        c5=c5, c6=c6, c7=c7, c8=c8, c9=c9,
        fc_eff=f_eff, dt=dt,
    )


# ============================================================================
# Fortran Subroutine ACCEL1 Parity Wrapper
# ============================================================================

def accel1(
    a: Sequence[float] | np.ndarray,
    ff: float,
    a2: np.ndarray,
    a1: np.ndarray,
    a0: np.ndarray,
    as_out: np.ndarray,
    vs_out: np.ndarray,
    skew: Optional[Sequence[float] | np.ndarray] = None,
    dt: float = 1.0e-5,
    dt12: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute OpenRadioss SUBROUTINE ACCEL1 step.

    Upstream Fortran reference:
    $OR_SRC/engine/source/tools/accele/accel1.F
    lines 28-117.

    Parameters
    ----------
    a : array_like of shape (3,)
        Raw acceleration vector in global coordinates [ax, ay, az].
    ff : float
        Filter cut-off frequency in Hz. If 0.0, raw acceleration is rotated
        into the skew system and integrated without filtering.
    a2 : np.ndarray of shape (3, 2)
        Output Stage 2 history: a2[j, 0] = z[n-1], a2[j, 1] = z[n-2]. (Modified in-place)
    a1 : np.ndarray of shape (3, 2)
        Stage 1 intermediate history: a1[j, 0] = y[n-1], a1[j, 1] = y[n-2]. (Modified in-place)
    a0 : np.ndarray of shape (3, 2)
        Input history: a0[j, 0] = x[n-1], a0[j, 1] = x[n-2]. (Modified in-place)
    as_out : np.ndarray of shape (3,)
        Output filtered acceleration in skew frame. (Modified in-place)
    vs_out : np.ndarray of shape (3,)
        Output integrated velocity in skew frame. (Modified in-place)
    skew : array_like of shape (9,) or (3, 3), optional
        Transformation matrix from global to local skew frame.
        If None, identity matrix is used.
    dt : float, default 1.0e-5
        Time step DT2 for filter coefficients calculation.
    dt12 : float, optional
        Time step for velocity integration. Defaults to dt.

    Returns
    -------
    (as_out, vs_out) : tuple of np.ndarray
        Updated filtered acceleration and integrated velocity in skew frame.
    """
    dt_int = dt if dt12 is None else dt12

    if skew is None:
        r_mat = np.eye(3, dtype=float)
    else:
        skew_arr = np.asarray(skew, dtype=float)
        if skew_arr.size == 9:
            r_mat = skew_arr.reshape(3, 3)
        else:
            r_mat = np.eye(3, dtype=float)

    a_arr = np.asarray(a, dtype=float)

    # Initial rotation and integration of raw acceleration (lines 48-53)
    # AS(1) = A(1)*SKEW(1) + A(2)*SKEW(2) + A(3)*SKEW(3), etc.
    as_raw = r_mat @ a_arr
    as_out[:] = as_raw
    vs_out += as_raw * dt_int

    if ff == 0.0:
        return as_out, vs_out

    # Compute filter coefficients (lines 56-89)
    coeffs = compute_filter_coefficients(ff, dt)

    c0, c1, c2, c3, c4 = coeffs.c0, coeffs.c1, coeffs.c2, coeffs.c3, coeffs.c4
    c5, c6, c7, c8, c9 = coeffs.c5, coeffs.c6, coeffs.c7, coeffs.c8, coeffs.c9

    # Filter each coordinate component j in {0, 1, 2} (lines 93-110)
    for j in range(3):
        x1 = a0[j, 1]  # X1 = A0(J, 2)
        x2 = a0[j, 0]  # X2 = A0(J, 1)
        x3 = a_arr[j]  # X3 = A(J)

        y1 = a1[j, 1]  # Y1 = A1(J, 2)
        y2 = a1[j, 0]  # Y2 = A1(J, 1)
        y3 = c0 * x3 + c1 * x2 + c2 * x1 + c3 * y2 + c4 * y1

        z1 = a2[j, 1]  # Z1 = A2(J, 2)
        z2 = a2[j, 0]  # Z2 = A2(J, 1)
        z3 = c5 * y3 + c6 * y2 + c7 * y1 + c8 * z2 + c9 * z1

        a0[j, 1] = x2
        a0[j, 0] = x3
        a1[j, 1] = y2
        a1[j, 0] = y3
        a2[j, 1] = z2
        a2[j, 0] = z3

    # Transform filtered acceleration to skew frame (lines 112-114):
    # AS(1) = A2(1, 1)*SKEW(1) + A2(2, 1)*SKEW(2) + A2(3, 1)*SKEW(3)
    z_vec = a2[:, 0]
    as_filtered = r_mat @ z_vec
    as_out[:] = as_filtered

    return as_out, vs_out


# ============================================================================
# Object-Oriented AccelFilter Class
# ============================================================================

class AccelFilter:
    """Cascaded 4th-order Butterworth digital filter for /ACCEL accelerometer sensors.

    Corresponds to OpenRadioss `/ACCEL` card and engine evaluation via `accel1.F`.
    Conforms to SAE J211-1 / ISO 6487 Channel Frequency Classes (CFC).

    Attributes
    ----------
    id : int, optional
        Accelerometer sensor ID.
    title : str, optional
        Sensor description title.
    node_id : int, optional
        Tracked node ID.
    skew_id : int, optional
        Local skew coordinate frame ID.
    cfc : int or str, optional
        SAE J211 channel frequency class (60, 180, 600, 1000).
    fc : float
        Filter cut-off frequency in Hz.
    dt : float, optional
        Simulation time step.
    skew : np.ndarray
        (3, 3) rotation matrix relating global to local coordinates.
    """

    def __init__(
        self,
        cfc: Optional[Union[int, str, float]] = None,
        fc: Optional[float] = None,
        dt: Optional[float] = None,
        skew: Optional[Sequence[float] | np.ndarray] = None,
        id: int = 1,
        title: str = "ACCELEROMETER",
        node_id: Optional[int] = None,
        skew_id: int = 0,
        initial_accel: Optional[Sequence[float] | np.ndarray] = None,
    ) -> None:
        self.id = int(id)
        self.title = str(title)
        self.node_id = node_id
        self.skew_id = skew_id

        # Determine cut-off frequency
        if cfc is not None:
            self.cfc: Optional[Union[int, str, float]] = cfc
            self.fc = parse_cfc(cfc)
        elif fc is not None:
            self.cfc = None
            self.fc = float(fc)
        else:
            # Default to CFC 1000 if neither specified
            self.cfc = 1000
            self.fc = 1650.0

        self.dt = float(dt) if dt is not None else 1.0e-5

        # Local skew frame transformation matrix
        if skew is None:
            self.skew = np.eye(3, dtype=np.float64)
        else:
            s_arr = np.asarray(skew, dtype=np.float64)
            if s_arr.shape == (3, 3):
                self.skew = s_arr.copy()
            elif s_arr.size == 9:
                self.skew = s_arr.reshape((3, 3)).copy()
            else:
                raise ValueError(f"Skew transformation matrix must have shape (3, 3) or (9,), got {s_arr.shape}")

        # State history arrays: shape (3, 2)
        # Column 0: x[n-1], Column 1: x[n-2]
        self.a0 = np.zeros((3, 2), dtype=np.float64)  # raw input history
        self.a1 = np.zeros((3, 2), dtype=np.float64)  # stage 1 history
        self.a2 = np.zeros((3, 2), dtype=np.float64)  # stage 2 output history

        # Output states in skew frame
        self._as = np.zeros(3, dtype=np.float64)       # filtered acceleration
        self._vs = np.zeros(3, dtype=np.float64)       # integrated velocity
        self._a_raw = np.zeros(3, dtype=np.float64)   # latest raw acceleration

        # Cached coefficients
        self._cached_coeffs: Optional[FilterCoefficients] = None
        self._last_dt: Optional[float] = None

        if initial_accel is not None:
            self.reset(initial_a=initial_accel)

    @property
    def a_filtered(self) -> np.ndarray:
        """Current filtered acceleration vector in local skew frame (3,)."""
        return self._as.copy()

    @property
    def v_filtered(self) -> np.ndarray:
        """Current integrated velocity vector in local skew frame (3,)."""
        return self._vs.copy()

    @property
    def a_raw(self) -> np.ndarray:
        """Latest raw input acceleration vector in global coordinates (3,)."""
        return self._a_raw.copy()

    @property
    def a_global_filtered(self) -> np.ndarray:
        """Current filtered acceleration vector transformed back into global coordinates (3,)."""
        # Since self.skew is orthogonal (R @ a_global = a_local => a_global = R.T @ a_local)
        return self.skew.T @ self._as

    def reset(
        self,
        initial_a: Optional[Sequence[float] | np.ndarray] = None,
        initial_v: Optional[Sequence[float] | np.ndarray] = None,
    ) -> None:
        """Reset internal history buffers.

        Parameters
        ----------
        initial_a : array_like of shape (3,), optional
            If provided, steady-state initial acceleration to preload into
            history buffers (preventing startup transient).
        initial_v : array_like of shape (3,), optional
            Initial velocity vector in skew coordinates. Defaults to zero.
        """
        if initial_a is not None:
            init_arr = np.asarray(initial_a, dtype=np.float64)
            if init_arr.size == 1:
                init_arr = np.array([init_arr.item(), 0.0, 0.0], dtype=np.float64)
            self._a_raw[:] = init_arr
            # In steady state: x = y = z = init_arr
            self.a0[:, 0] = init_arr
            self.a0[:, 1] = init_arr
            self.a1[:, 0] = init_arr
            self.a1[:, 1] = init_arr
            self.a2[:, 0] = init_arr
            self.a2[:, 1] = init_arr
            self._as[:] = self.skew @ init_arr
        else:
            self.a0.fill(0.0)
            self.a1.fill(0.0)
            self.a2.fill(0.0)
            self._a_raw.fill(0.0)
            self._as.fill(0.0)

        if initial_v is not None:
            v_init = np.asarray(initial_v, dtype=np.float64)
            if v_init.size == 1:
                self._vs[:] = np.array([v_init.item(), 0.0, 0.0], dtype=np.float64)
            else:
                self._vs[:] = v_init
        else:
            self._vs.fill(0.0)

    def get_coefficients(self, dt: Optional[float] = None) -> FilterCoefficients:
        """Retrieve or compute filter coefficients for time step dt."""
        use_dt = self.dt if dt is None else float(dt)
        if self._cached_coeffs is None or self._last_dt != use_dt:
            self._cached_coeffs = compute_filter_coefficients(self.fc, use_dt)
            self._last_dt = use_dt
        return self._cached_coeffs

    def update(
        self,
        a_global: Sequence[float] | np.ndarray | float,
        dt: Optional[float] = None,
        dt12: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a single time-step filter update.

        Parameters
        ----------
        a_global : array_like of shape (3,) or float
            Current raw acceleration vector in global coordinates.
        dt : float, optional
            Time step for the current step. If None, uses self.dt.
        dt12 : float, optional
            Time step for velocity integration. If None, uses dt.

        Returns
        -------
        a_local : np.ndarray of shape (3,)
            Filtered acceleration vector in local skew frame.
        v_local : np.ndarray of shape (3,)
            Integrated velocity vector in local skew frame.
        """
        step_dt = self.dt if dt is None else float(dt)
        step_dt12 = step_dt if dt12 is None else float(dt12)

        a_arr = np.asarray(a_global, dtype=np.float64)
        if a_arr.ndim == 0:
            a_vec = np.array([float(a_arr), 0.0, 0.0], dtype=np.float64)
        elif a_arr.shape == (1,):
            a_vec = np.array([float(a_arr[0]), 0.0, 0.0], dtype=np.float64)
        elif a_arr.shape == (3,):
            a_vec = a_arr.copy()
        else:
            raise ValueError(f"Input acceleration must have shape (3,), got {a_arr.shape}")

        self._a_raw[:] = a_vec

        # Call Fortran parity routine
        accel1(
            a=a_vec,
            ff=self.fc,
            a2=self.a2,
            a1=self.a1,
            a0=self.a0,
            as_out=self._as,
            vs_out=self._vs,
            skew=self.skew,
            dt=step_dt,
            dt12=step_dt12,
        )

        return self._as.copy(), self._vs.copy()

    def step(
        self,
        a_global: Sequence[float] | np.ndarray | float,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Alias for update()."""
        return self.update(a_global, dt=dt)

    def filter_series(
        self,
        time: Sequence[float] | np.ndarray,
        accel: Sequence[float] | np.ndarray,
        skew: Optional[Sequence[float] | np.ndarray] = None,
        reset_state: bool = True,
        preload_steady_state: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter an entire acceleration time series history.

        Parameters
        ----------
        time : array_like of shape (N,)
            Monotonically increasing time values.
        accel : array_like of shape (N,) or (N, 3)
            Raw acceleration signal history.
        skew : array_like of shape (3, 3) or (9,), optional
            Skew rotation matrix. Defaults to self.skew.
        reset_state : bool, default True
            Whether to reset internal history before processing.
        preload_steady_state : bool, default False
            If True, initializes filter state to steady state matching first sample
            accel[0] to prevent initial settling transients.

        Returns
        -------
        a_filt : np.ndarray of shape (N, 3) or (N,)
            Filtered acceleration signal.
        v_filt : np.ndarray of shape (N, 3) or (N,)
            Integrated velocity signal.
        """
        t_arr = np.asarray(time, dtype=np.float64)
        a_arr = np.asarray(accel, dtype=np.float64)

        if len(t_arr) < 1:
            raise ValueError("Time array must contain at least 1 sample")
        if len(t_arr) != len(a_arr):
            raise ValueError(f"Length mismatch: time ({len(t_arr)}) vs accel ({len(a_arr)})")

        is_1d = (a_arr.ndim == 1)
        if is_1d:
            a_3d = np.zeros((len(a_arr), 3), dtype=np.float64)
            a_3d[:, 0] = a_arr
        else:
            if a_arr.shape[1] != 3:
                raise ValueError(f"accel must have 3 columns, got {a_arr.shape}")
            a_3d = a_arr.copy()

        if skew is not None:
            orig_skew = self.skew.copy()
            s_arr = np.asarray(skew, dtype=np.float64)
            self.skew = s_arr.reshape((3, 3)) if s_arr.size == 9 else s_arr
        else:
            orig_skew = None

        if reset_state:
            init_a = a_3d[0] if preload_steady_state else None
            self.reset(initial_a=init_a)

        n_pts = len(t_arr)
        a_out = np.zeros((n_pts, 3), dtype=np.float64)
        v_out = np.zeros((n_pts, 3), dtype=np.float64)

        for i in range(n_pts):
            if i == 0:
                dt_step = float(t_arr[1] - t_arr[0]) if n_pts > 1 else self.dt
            else:
                dt_step = float(t_arr[i] - t_arr[i - 1])
            dt_step = max(dt_step, 1.0e-15)

            as_i, vs_i = self.update(a_3d[i], dt=dt_step, dt12=dt_step)
            a_out[i] = as_i
            v_out[i] = vs_i

        if orig_skew is not None:
            self.skew = orig_skew

        if is_1d:
            return a_out[:, 0], v_out[:, 0]
        return a_out, v_out

    def transfer_function(self, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the discrete-time transfer function polynomials B(z) and A(z).

        H(z) = B(z) / A(z) for the cascaded 4th-order filter:
        B(z) = (c0 + c1*z^-1 + c2*z^-2) * (c5 + c6*z^-1 + c7*z^-2)
        A(z) = (1 - c3*z^-1 - c4*z^-2) * (1 - c8*z^-1 - c9*z^-2)

        Parameters
        ----------
        dt : float, optional
            Time step. Defaults to self.dt.

        Returns
        -------
        (b, a) : tuple of np.ndarray
            Numerator b of length 5 and denominator a of length 5.
        """
        c = self.get_coefficients(dt)
        b1 = np.array([c.c0, c.c1, c.c2], dtype=np.float64)
        a1 = np.array([1.0, -c.c3, -c.c4], dtype=np.float64)

        b2 = np.array([c.c5, c.c6, c.c7], dtype=np.float64)
        a2 = np.array([1.0, -c.c8, -c.c9], dtype=np.float64)

        # Convolution of 2nd-order stages yields 4th-order polynomial
        b = np.convolve(b1, b2)
        a = np.convolve(a1, a2)
        return b, a

    def frequency_response(
        self,
        frequencies: Sequence[float] | np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Evaluate the complex frequency response H(e^(j 2 pi f dt)) at specified frequencies.

        Parameters
        ----------
        frequencies : array_like
            Frequencies in Hz at which to evaluate the frequency response.
        dt : float, optional
            Time step. Defaults to self.dt.

        Returns
        -------
        np.ndarray of complex128
            Complex frequency response H(f).
        """
        use_dt = self.dt if dt is None else float(dt)
        c = self.get_coefficients(use_dt)

        freqs = np.asarray(frequencies, dtype=np.float64)
        omega = 2.0 * np.pi * freqs * use_dt
        z = np.exp(1j * omega)
        z1 = 1.0 / z
        z2 = z1 * z1

        # Stage 1 frequency response
        num1 = c.c0 + c.c1 * z1 + c.c2 * z2
        den1 = 1.0 - c.c3 * z1 - c.c4 * z2
        h1 = num1 / den1

        # Stage 2 frequency response
        num2 = c.c5 + c.c6 * z1 + c.c7 * z2
        den2 = 1.0 - c.c8 * z1 - c.c9 * z2
        h2 = num2 / den2

        return h1 * h2

    def __repr__(self) -> str:
        return (
            f"AccelFilter(id={self.id}, cfc={self.cfc!r}, fc={self.fc:.1f}Hz, "
            f"dt={self.dt:.3e}s, node_id={self.node_id})"
        )
