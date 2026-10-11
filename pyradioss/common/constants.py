"""
Numerical constants used throughout the solver.

Fortran origin: ``common_source/comm/constant.inc`` (and the ``ZERO``,
``EM20``, ``EP30`` … parameters sprinkled through every Fortran file), which
defines named constants mostly to guarantee identical literals everywhere and
to make the "guard" values greppable.  We keep the same *names* so that a
reader coming from the Fortran immediately recognizes e.g. ``EM20`` as the
tiny value used to avoid division by zero.
"""

# Tiny value used to protect divisions (Fortran: EM20). Any length, area or
# velocity magnitude smaller than this is treated as zero.
EM15 = 1.0e-15
EM20 = 1.0e-20
EM30 = 1.0e-30

# Huge values used to initialize minima (Fortran: EP30/EP20): EP30 starts the
# search for the smallest element time step, EP20 stands for "no bound" where
# a minimum is not searched (dtcdk_reg.F's ``DTNL_TH`` at NDDL == 1).
EP30 = 1.0e30
EP20 = 1.0e30

# Default time-step *scale factor* applied to the critical (Courant) time
# step. The pure central-difference stability limit dt <= 2/omega_max is an
# upper bound computed from linearized element stiffness; non-linearity,
# contact and hourglass forces erode it, so Radioss multiplies by a safety
# factor. Fortran default: 0.9 (see engine /DT card documentation and
# engine/source/time_step/).
DEFAULT_DT_SCALE = 0.9

# Default bulk-viscosity coefficients for solid elements (quadratic qa and
# linear qb terms) — same defaults as the Radioss /PROP/SOLID card.
# They damp shock oscillations by adding a viscous pressure
#   q = rho * l * (qa^2 * l * tr(D)^2  -  qb * c * tr(D))   when tr(D) < 0.
DEFAULT_QA = 1.1
DEFAULT_QB = 0.05

# Default hourglass resistance coefficient for one-point-integrated elements
# (Radioss /PROP/SOLID "h" and /PROP/SHELL "hm/hf/hr" default 0.01–0.1 range;
# we use 0.1, the common practical value, LS-DYNA's QM default is 0.10 too).
DEFAULT_HOURGLASS = 0.10

# Shear correction factor for Mindlin–Reissner shells (5/6): the transverse
# shear stress distribution through the thickness is parabolic, not constant,
# so the constant-shear-strain kinematics overestimates shear stiffness by
# exactly 6/5 for a homogeneous section.
SHEAR_FACTOR = 5.0 / 6.0
