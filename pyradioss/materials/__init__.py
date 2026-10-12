"""
pyradioss.materials — constitutive laws.

Fortran origin: ``engine/source/materials/mat/matXXX/sigepsXX.F``. In
OpenRadioss every material law is a routine ``SIGEPS<law>`` receiving the
strain-rate components and the old stress of a *group* of elements and
returning the new stress (plus updated internal variables and the sound
speed). The element kernels call it once per integration point per cycle.

The port keeps that contract exactly (vectorized over the group):

    law01_elastic.solid_update / shell_update        (LAW1)
    law02_johnson_cook.solid_update / shell_update   (LAW2)
    law36_tabulated.solid_update / shell_update      (LAW36)
    law27_brittle.shell_update                       (LAW27, shells only)
    law42_ogden.solid_update                         (LAW42, solids only)

Stress storage convention (Voigt, engineering shear):
    solids : (n, 6)  = [xx, yy, zz, xy, yz, zx]
    shells : (n, 3)  = [xx, yy, xy]   (in-plane; transverse shear handled
                                       elastically by the shell kernel)
Strain increments use the same ordering, with *engineering* shear
(gamma = 2*eps) exactly like the Fortran EPSPXX/EPSPXY arguments.

The objective (Jaumann) rotation of the old stress is done by the element
kernel BEFORE calling the law — same split as the Fortran, where SROTA3.F
rotates the stress and SIGEPS only integrates the constitutive rate.
(Hyperelastic laws — LAW42 — overwrite the stress from the deformation
gradient instead, so the pre-rotation is harmless for them.)

The ``extra`` argument (M3)
---------------------------
Some laws carry state beyond (sig, epsp), or need kinematic context
beyond the strain increment. The Fortran passes dozens of UVAR/extra
arguments; the port passes one ``extra`` dict of per-slice array views:

* ``extra_shapes(mat, nip)`` tells the element kernels which persistent
  per-element (per-layer for shells) arrays a law needs; the kernels
  allocate them in the group state under ``mat_extra`` and slice them
  into ``extra`` at call time.
* ``needs_defgrad(mat)`` asks for the deformation gradient: the solid
  kernels then compute F exactly from the stored initial shape-function
  gradients each cycle and pass it as ``extra["F"]``.
* shell kernels always add ``extra["layfail"]`` (the shared layer-failure
  array — see pyradioss.failure) so a law can break layers (LAW27).

``solid_update`` returns (sig, epsp, c) where c is a per-element sound
speed array, or None to keep the kernel's constant elastic estimate.
This is the Fortran SOUNDSP output: laws whose tangent stiffness can
exceed the ground-state one (LAW42 at large stretch!) MUST return the
true current sound speed or the Courant time step is not a bound.
"""

from typing import Any, Optional

import numpy as np

from . import (detonation, eos, law01_elastic, law02_johnson_cook, law03_plas_bost,  # noqa: F401
               law04_hyd_jcook, law06_hyd_visc, law10_soil,
               law12_comp3d, law14_compso, law15_chang,
               law19_fabric, law21_dprag, law22_dama, law24_concrete, law25_composite, law27_brittle,
               law28_honeycomb, law33_foamplas, law34_boltzmann, law35_kelvinmax, law36_tabulated,
               law37_biphas,
               law40_kelvinmax, law42_ogden, law44_cowper,
               law48_zhao, law49_steinb, law50_visc_honey,
               law52_gurson,
               law60_plast3,
               law62_hypervisco, law69_hyperelastic, law70_tabfoam, law79_john_holm, law81_druckerprager,
               law82_ogden,
               law83_spotweld, law114_seatbelt, law119_seatbelt, law120_advanced,
               law120_tapo, law121_plas_rate,
               law58_fabr_a,
               law57_barlat,
               law66_plas_tab,
               law71_nitinol,
               law73_hill_therm,
               law74_hill_3d,
               law87_barlat2000,
               law88_tab_hyp,
               law92_arruda_boyce,
               law93_orth_hill,
               law94_yeoh,
               law95_bergstrom_boyce,
               law100_multi_network,
               law101_plas_poly,
               law102_dprag2,
               law103_hensel_spittel,
               law104_drucker,
               law105_powder_burn,
               law106_jcook_alm,
               law107_paper_light,
               law109_tab_plas,
               law110_vegter,
               law117_cohesive,
               law123_daimler,
               law124_cdpm2,
               law126_hjc,
               law169_arup,
               law163_crush_foam,
               law76_samp,
               law90_foam,
               law190_dubois,
               mat_gas, mat_void, shell_moduli,
               law00_void, law11_fluid_visc, law13_fluid_ale, law16_gray_ewing, law17_orth_elastic,
               law18_plas_iso, law20_rigid, law23_user_mat, law26_honeycomb_sesame, law41_jwl_burn,
               law45_fabric_membrane, law45_orth_fabric, law46_kin_hard, law51_granular_soil, law53_tab_foam, law54_hyd_visc,
               law55_shell_orth, law56_orth_fail, law59_spring_rate, law63_composite_shell, law64_rate_comp,
               law65_thermo_visc, law68_tab_plas, law72_hill_mmc, law75_thermo_trans, law77_visc_poly, law77_polymer,
               law78_composite_dmg, law78_yoshida, law80_ramberg, law84_mooney_rivlin, law85_void_pinch, law86_honeycomb_shell,
               law91_pinch_shell, law96_thermo_visc, law96_thermovp, law97_orth_nonlinear, law98_loss_optim, law108_yield_fit,
               law111_barlat_yld2004, law112_xia_damage, law113_yield_curve_fit, law115_perzyna, law116_spotweld,
               law118_reserved, law122_chaboche, law125_tab_aniso_rate, law127_mohr_coulomb, law128_hill_rate_shell,
               law129_barlat_yld2000, law130_nonlocal_damage, law131_modular_elastoplas, law132_rate_modifier,
               law133_microplane_concrete, law134_fabric_wrinkle, law135_modular_user, law151_multimat_ale,
               law158_progressive_composite, law187_samp1_polymer,
               inimap, inivol)
from .law120_tapo import (
    Law120Params,
    build_law120,
    solid_update as law120_solid_update,
    shell_update as law120_shell_update,
    sound_speed as law120_sound_speed,
    solid_tangent as law120_solid_tangent,
    consistent_solid_tangent as law120_consistent_solid_tangent,
    consistent_shell_tangent as law120_consistent_shell_tangent,
    shell_membrane_tangent as law120_shell_membrane_tangent,
    extra_shapes as law120_extra_shapes,
    resolve as law120_resolve,
)
from .law121_plas_rate import (
    Law121Params,
    build_law121,
    solid_update as law121_solid_update,
    shell_update as law121_shell_update,
    sound_speed as law121_sound_speed,
    sound_speed_solid as law121_sound_speed_solid,
    sound_speed_shell as law121_sound_speed_shell,
    solid_tangent as law121_solid_tangent,
    consistent_solid_tangent as law121_consistent_solid_tangent,
    consistent_shell_tangent as law121_consistent_shell_tangent,
    shell_tangent as law121_shell_tangent,
    extra_shapes as law121_extra_shapes,
    resolve as law121_resolve,
)
from .law123_daimler import (
    Law123Params,
    Law132Params,
    build_law123,
    build_law132,
    solid_step as law123_solid_step,
    shell_step as law123_shell_step,
    solid_update as law123_solid_update,
    shell_update as law123_shell_update,
    sound_speed as law123_sound_speed,
    solid_tangent as law123_solid_tangent,
    shell_tangent as law123_shell_tangent,
    consistent_solid_tangent as law123_consistent_solid_tangent,
    consistent_shell_tangent as law123_consistent_shell_tangent,
    extra_shapes as law123_extra_shapes,
)
from .law190_dubois import (
    Law190Params,
    build_law190,
    solid_step as law190_solid_step,
    solid_update as law190_solid_update,
    sound_speed as law190_sound_speed,
    solid_tangent as law190_solid_tangent,
    consistent_solid_tangent as law190_consistent_solid_tangent,
    shell_update as law190_shell_update,
    extra_shapes as law190_extra_shapes,
    resolve as law190_resolve,
)
from .law117_cohesive import (
    Law117Params,
    build_law117,
    solid_step as law117_solid_step,
    shell_step as law117_shell_step,
    solid_update as law117_solid_update,
    shell_update as law117_shell_update,
    sound_speed as law117_sound_speed,
    solid_tangent as law117_solid_tangent,
    shell_tangent as law117_shell_tangent,
    consistent_solid_tangent as law117_consistent_solid_tangent,
    consistent_shell_tangent as law117_consistent_shell_tangent,
    shell_membrane_tangent as law117_shell_membrane_tangent,
    extra_shapes as law117_extra_shapes,
)
from .law90_foam import (
    Law90Params,
    build_law90,
    solid_step as law90_solid_step,
    solid_update as law90_solid_update,
    sound_speed as law90_sound_speed,
    solid_tangent as law90_solid_tangent,
    consistent_solid_tangent as law90_consistent_solid_tangent,
    shell_update as law90_shell_update,
    extra_shapes as law90_extra_shapes,
)
from .law124_cdpm2 import (
    Law124Params,
    build_law124,
    solid_step as law124_solid_step,
    solid_update as law124_solid_update,
    sound_speed as law124_sound_speed,
    solid_tangent as law124_solid_tangent,
    consistent_solid_tangent as law124_consistent_solid_tangent,
    extra_shapes as law124_extra_shapes,
    resolve as law124_resolve,
)
from .law126_hjc import (
    Law126Params,
    build_law126,
    solid_step as law126_solid_step,
    solid_update as law126_solid_update,
    sound_speed as law126_sound_speed,
    solid_tangent as law126_solid_tangent,
    consistent_solid_tangent as law126_consistent_solid_tangent,
    extra_shapes as law126_extra_shapes,
)
from .law169_arup import (
    Law169Params,
    build_law169,
    solid_step as law169_solid_step,
    shell_step as law169_shell_step,
    solid_update as law169_solid_update,
    shell_update as law169_shell_update,
    sound_speed as law169_sound_speed,
    solid_tangent as law169_solid_tangent,
    shell_tangent as law169_shell_tangent,
    consistent_solid_tangent as law169_consistent_solid_tangent,
    consistent_shell_tangent as law169_consistent_shell_tangent,
    extra_shapes as law169_extra_shapes,
)
from .law76_samp import (
    Law76Params,
    build_law76,
    solid_step as law76_solid_step,
    shell_step as law76_shell_step,
    solid_update as law76_solid_update,
    shell_update as law76_shell_update,
    sound_speed as law76_sound_speed,
    solid_tangent as law76_solid_tangent,
    shell_tangent as law76_shell_tangent,
    consistent_solid_tangent as law76_consistent_solid_tangent,
    consistent_shell_tangent as law76_consistent_shell_tangent,
    shell_membrane_tangent as law76_shell_membrane_tangent,
    extra_shapes as law76_extra_shapes,
)
from .law71_nitinol import (
    Law71Params,
    build_law71,
    solid_update as law71_solid_update,
    shell_update as law71_shell_update,
    sound_speed as law71_sound_speed,
    consistent_solid_tangent as law71_solid_tangent,
    consistent_shell_tangent as law71_shell_tangent,
    shell_membrane_tangent as law71_shell_membrane_tangent,
    extra_shapes as law71_extra_shapes,
)
from .law110_vegter import (
    VegterParams,
    build_law110,
    shell_update as law110_shell_update,
    sound_speed as law110_sound_speed,
    consistent_shell_tangent as law110_consistent_shell_tangent,
    extra_shapes as law110_extra_shapes,
)
from .law109_tab_plas import (
    Law109Params,
    build_law109,
    solid_update as law109_solid_update,
    shell_update as law109_shell_update,
    sound_speed as law109_sound_speed,
    solid_tangent as law109_solid_tangent,
    consistent_shell_tangent as law109_consistent_shell_tangent,
    extra_shapes as law109_extra_shapes,
)
from .law107_paper_light import (
    PaperLightParams,
    PaperLightConstants,
    build_law107,
    compute_paper_light_constants,
    compute_paper_light_normals,
    compute_paper_light_yield_surface,
    mat107_newton_update_single,
    mat107_nice_update_single,
    solid_update as law107_solid_update,
    shell_update as law107_shell_update,
    sound_speed as law107_sound_speed,
    sound_speed_solid as law107_sound_speed_solid,
    sound_speed_shell as law107_sound_speed_shell,
    solid_tangent as law107_solid_tangent,
    consistent_shell_tangent as law107_consistent_shell_tangent,
    extra_shapes as law107_extra_shapes,
)
from .law105_powder_burn import (
    PowderBurnParams,
    build_law105,
    solid_update as law105_solid_update,
    sound_speed as law105_sound_speed,
    solid_tangent as law105_solid_tangent,
)
from .law106_jcook_alm import (
    JCookAlmParams,
    build_law106,
    compute_temperature_elastic_moduli as law106_compute_temperature_elastic_moduli,
    compute_jcook_alm_yield_stress as law106_compute_jcook_alm_yield_stress,
    solid_update as law106_solid_update,
    shell_update as law106_shell_update,
    sound_speed as law106_sound_speed,
    sound_speed_solid as law106_sound_speed_solid,
    sound_speed_shell as law106_sound_speed_shell,
    solid_tangent as law106_solid_tangent,
    consistent_shell_tangent as law106_consistent_shell_tangent,
)
from .law104_drucker import (
    DruckerParams,
    build_law104,
    compute_drucker_constants,
    compute_drucker_equivalent_stress,
    compute_drucker_yield_stress,
    solid_update as law104_solid_update,
    shell_update as law104_shell_update,
    sound_speed as law104_sound_speed,
    sound_speed_solid as law104_sound_speed_solid,
    sound_speed_shell as law104_sound_speed_shell,
    solid_tangent as law104_solid_tangent,
    consistent_shell_tangent as law104_consistent_shell_tangent,
)
from .law103_hensel_spittel import (
    HenselSpittelParams,
    build_law103,
    compute_hensel_spittel_flow_stress,
    compute_hensel_spittel_hardening_modulus,
    solid_update as law103_solid_update,
    solid_update_single as law103_solid_update_single,
    solid_update_array as law103_solid_update_array,
    sound_speed as law103_sound_speed,
    solid_tangent as law103_solid_tangent,
)
from .law102_dprag2 import (
    DPrag2Params,
    build_law102,
    compute_dprag2_constants,
    solid_update as law102_solid_update,
    solid_update_single as law102_solid_update_single,
    solid_update_array as law102_solid_update_array,
    sound_speed as law102_sound_speed,
    solid_tangent as law102_solid_tangent,
)
from .law101_plas_poly import (
    BouvardParams,
    build_law101,
    solid_update as law101_solid_update,
    sound_speed as law101_sound_speed,
    solid_tangent as law101_solid_tangent,
    extra_shapes as law101_extra_shapes,
)
from .law100_multi_network import (
    MultiNetworkParams,
    SecondaryNetworkParams,
    build_law100,
    solid_update as law100_solid_update,
    sound_speed as law100_sound_speed,
    solid_tangent as law100_solid_tangent,
)
from .law95_bergstrom_boyce import (
    BergstromBoyceParams,
    build_law95,
    solid_update as law95_solid_update,
    sound_speed as law95_sound_speed,
    consistent_tangent as law95_consistent_tangent,
    extra_shapes as law95_extra_shapes,
)
from .law93_orth_hill import (
    OrthHillParams,
    build_law93,
    solid_update as law93_solid_update,
    shell_update as law93_shell_update,
    sound_speed as law93_sound_speed,
    sound_speed_shell as law93_sound_speed_shell,
    consistent_tangent as law93_consistent_tangent,
    extra_shapes as law93_extra_shapes,
)
from .law94_yeoh import (
    YeohParams,
    build_law94,
    solid_update as law94_solid_update,
    shell_update as law94_shell_update,
    sound_speed as law94_sound_speed,
    sound_speed_shell as law94_sound_speed_shell,
    consistent_tangent as law94_consistent_tangent,
    extra_shapes as law94_extra_shapes,
)
from .law92_arruda_boyce import (
    ArrudaBoyceParams,
    build_law92,
    solid_update as law92_solid_update,
    shell_update as law92_shell_update,
    sound_speed as law92_sound_speed,
    sound_speed_shell as law92_sound_speed_shell,
    consistent_tangent as law92_consistent_tangent,
    extra_shapes as law92_extra_shapes,
)
from .law88_tab_hyp import (
    Law88Params,
    build_law88,
    solid_update_law88,
    shell_update_law88,
    sound_speed_solid_law88,
    sound_speed_shell_law88,
    solid_update as law88_solid_update,
    shell_update as law88_shell_update,
    sound_speed_solid as law88_solid_sound_speed,
    sound_speed_shell as law88_shell_sound_speed,
    sound_speed as law88_sound_speed,
    consistent_solid_tangent as law88_solid_tangent,
    consistent_shell_tangent as law88_shell_tangent,
    shell_membrane_tangent as law88_shell_membrane_tangent,
    extra_shapes as law88_extra_shapes,
)
from .law87_barlat2000 import (
    Law87Params,
    build_law87,
    solid_update_law87,
    shell_update_law87,
    sound_speed_shell_law87,
    barlat2000_equivalent_stress,
    barlat2000_yield_surface,
    barlat2000_gradient,
    barlat2000_yield_and_gradient,
    solid_update as law87_solid_update,
    shell_update as law87_shell_update,
    sound_speed_shell as law87_sound_speed,
    consistent_shell_tangent as law87_shell_tangent,
    shell_membrane_tangent as law87_shell_membrane_tangent,
    extra_shapes as law87_extra_shapes,
)
from .law74_hill_3d import (
    Law74Params,
    build_law74,
    solid_update_law74,
    shell_update_law74,
    sound_speed as law74_sound_speed,
    sound_speed_solid as law74_solid_sound_speed,
    solid_update as law74_solid_update,
    shell_update as law74_shell_update,
    consistent_solid_tangent as law74_solid_tangent,
    extra_shapes as law74_extra_shapes,
)
from .law66_plas_tab import (
    Law66Params,
    build_law66,
    solid_update_law66,
    shell_update_law66,
    sound_speed as law66_sound_speed,
    sound_speed_solid as law66_solid_sound_speed,
    sound_speed_shell as law66_shell_sound_speed,
    solid_update as law66_solid_update,
    shell_update as law66_shell_update,
    consistent_solid_tangent as law66_solid_tangent,
    consistent_shell_tangent as law66_shell_tangent,
    shell_membrane_tangent as law66_shell_membrane_tangent,
    extra_shapes as law66_extra_shapes,
)
from .law73_hill_therm import (
    Law73Params,
    build_law73,
    solid_update_law73,
    shell_update_law73,
    sound_speed as law73_sound_speed,
    tangent_law73_shell,
    solid_update as law73_solid_update,
    shell_update as law73_shell_update,
    consistent_shell_tangent as law73_shell_tangent,
    shell_membrane_tangent as law73_shell_membrane_tangent,
    extra_shapes as law73_extra_shapes,
)
from .law163_crush_foam import (
    Law163Params,
    build_law163,
    solid_update_law163,
    shell_update_law163,
    sound_speed_solid_law163,
    tangent_law163_solid,
    solid_update as law163_solid_update,
    shell_update as law163_shell_update,
    sound_speed_solid as law163_sound_speed,
    consistent_solid_tangent as law163_solid_tangent,
    extra_shapes as law163_extra_shapes,
)
from .law50_visc_honey import (
    Law50Params,
    build_law50,
    solid_update_law50,
    shell_update_law50,
    sound_speed_solid_law50,
    tangent_law50_solid,
    solid_update as law50_solid_update,
    shell_update as law50_shell_update,
    sound_speed_solid as law50_sound_speed,
    consistent_solid_tangent as law50_solid_tangent,
    extra_shapes as law50_extra_shapes,
)
from .law79_john_holm import (
    Law79Params,
    build_law79,
    solid_update_law79,
    shell_update_law79,
    sound_speed_solid_law79,
    tangent_law79_solid,
    solid_update as law79_solid_update,
    shell_update as law79_shell_update,
    sound_speed_solid as law79_sound_speed,
    consistent_solid_tangent as law79_solid_tangent,
    extra_shapes as law79_extra_shapes,
)
from .law49_steinb import (
    Law49Params,
    build_law49,
    solid_update_law49,
    shell_update_law49,
    sound_speed_solid_law49,
    tangent_law49_solid,
    solid_update as law49_solid_update,
    shell_update as law49_shell_update,
    sound_speed_solid as law49_sound_speed,
    consistent_solid_tangent as law49_solid_tangent,
)
from .law57_barlat import (
    Law57Params,
    build_law57,
    solid_update_law57,
    shell_update_law57,
    sound_speed_shell_law57,
    tangent_law57_shell,
    shell_update as law57_shell_update,
    solid_update as law57_solid_update,
    sound_speed_shell as law57_sound_speed,
    consistent_shell_tangent as law57_shell_tangent,
    shell_membrane_tangent as law57_shell_membrane_tangent,
    shell_tangent as law57_tangent_shell,
)
from .law21_dprag import (
    build_law21,
    solid_update_law21,
    shell_update_law21,
    sound_speed_solid_law21,
    tangent_law21_solid,
    solid_update as law21_solid_update,
    shell_update as law21_shell_update,
    sound_speed_solid as law21_sound_speed,
    consistent_solid_tangent as law21_solid_tangent,
)
from .law58_fabr_a import (
    Law58Params,
    FabricAMaterial,
    build_law58,
    solid_update_law58,
    shell_update_law58,
    sound_speed_shell_law58,
    tangent_law58_shell,
    shell_update as law58_shell_update,
    solid_update as law58_solid_update,
    sound_speed_shell as law58_sound_speed,
    tangent_shell as law58_shell_tangent,
)
from .law48_zhao import (
    Law48Params,
    build_law48,
    solid_update_law48,
    shell_update_law48,
    sound_speed_solid_law48,
    sound_speed_shell_law48,
    tangent_law48_solid,
    tangent_law48_shell,
    solid_update as law48_solid_update,
    shell_update as law48_shell_update,
    sound_speed as law48_sound_speed,
    consistent_solid_tangent as law48_solid_tangent,
    consistent_shell_tangent as law48_shell_tangent,
)
from .law52_gurson import (
    Law52Params,
    build_law52,
    solid_update_law52,
    shell_update_law52,
    sound_speed_solid_law52,
    sound_speed_shell_law52,
    tangent_law52_solid,
    tangent_law52_shell,
    shell_membrane_tangent as law52_shell_membrane_tangent,
    solid_update as law52_solid_update,
    shell_update as law52_shell_update,
    sound_speed as law52_sound_speed,
    sound_speed_solid as law52_solid_sound_speed,
    sound_speed_shell as law52_shell_sound_speed,
    consistent_solid_tangent as law52_solid_tangent,
    consistent_shell_tangent as law52_shell_tangent,
)
from .law57_barlat import (
    Law57Params,
    BarlatParams,
    build_law57,
    barlat_params,
    calculp2,
    barlat_equivalent_stress,
    shell_update_law57,
    sound_speed_shell_law57,
    tangent_law57_shell,
    shell_update as law57_shell_update,
    sound_speed as law57_sound_speed,
    consistent_shell_tangent as law57_shell_tangent,
    shell_membrane_tangent as law57_shell_membrane_tangent,
    shell_tangent as law57_tangent_shell,
)
from .law60_plast3 import (
    Law60Params,
    build_law60,
    solid_update as law60_solid_update,
    shell_update as law60_shell_update,
    sound_speed as law60_sound_speed,
    consistent_solid_tangent as law60_consistent_solid_tangent,
    consistent_shell_tangent as law60_consistent_shell_tangent,
    consistent_solid_tangent as law60_solid_tangent,
    consistent_shell_tangent as law60_shell_tangent,
)
from .law24_concrete import (Law24Params,
                             build_conc,
                             build_law24,
                             solid_update as law24_solid_update,
                             consistent_solid_tangent as law24_solid_tangent,
                             sound_speed as law24_sound_speed,
                             peric_damage_update as law24_peric_damage_update)
from .law34_boltzmann import (solid_update as law34_solid_update,
                             shell_update as law34_shell_update,
                             sound_speed as law34_sound_speed,
                             consistent_solid_tangent as law34_solid_tangent,
                             shell_membrane_tangent as law34_shell_tangent,
                             truss_update as law34_truss_update,
                             beam_update as law34_beam_update)
from .law37_biphas import (Law37Params,
                           build_law37,
                           solid_update as law37_solid_update,
                           shell_update as law37_shell_update,
                           sound_speed as law37_sound_speed,
                           consistent_solid_tangent as law37_solid_tangent,
                           tangent_law37_solid)
from .law69_hyperelastic import (
    Law69Params,
    build_law69,
    solid_update as law69_solid_update,
    shell_update as law69_shell_update,
    solid_sound_speed as law69_solid_sound_speed,
    shell_sound_speed as law69_shell_sound_speed,
    consistent_solid_tangent as law69_consistent_solid_tangent,
    consistent_shell_tangent as law69_consistent_shell_tangent,
    sound_speed as law69_sound_speed,
    sound_speed_shell as law69_sound_speed_shell,
    consistent_solid_tangent as law69_solid_tangent,
    consistent_shell_tangent as law69_shell_tangent,
)
from .law82_ogden import (
    OgdenParams,
    build_law82,
    solid_update as law82_solid_update,
    shell_update as law82_shell_update,
    solid_sound_speed as law82_solid_sound_speed,
    shell_sound_speed as law82_shell_sound_speed,
    consistent_solid_tangent as law82_consistent_solid_tangent,
    consistent_shell_tangent as law82_consistent_shell_tangent,
    sound_speed as law82_sound_speed,
    sound_speed_shell as law82_sound_speed_shell,
    consistent_solid_tangent as law82_solid_tangent,
)

try:
    from . import law38_visc_tab
    from .law38_visc_tab import (solid_update as law38_solid_update,
                                sound_speed as law38_sound_speed,
                                consistent_solid_tangent as law38_solid_tangent)
except ImportError:
    law38_visc_tab = None
    law38_solid_update = None
    law38_sound_speed = None
    law38_solid_tangent = None

try:
    from . import law32_hill
    from .law32_hill import (shell_update as law32_shell_update,
                             consistent_shell_tangent as law32_shell_tangent,
                             sound_speed as law32_sound_speed)
except ImportError:
    law32_hill = None
    law32_shell_update = None
    law32_shell_tangent = None
    law32_sound_speed = None


def _get_law32():
    global law32_hill, law32_shell_update, law32_shell_tangent, law32_sound_speed
    if law32_hill is None:
        try:
            from . import law32_hill as _m
            law32_hill = _m
            law32_shell_update = getattr(_m, "shell_update", None)
            law32_shell_tangent = getattr(_m, "consistent_shell_tangent", None)
            law32_sound_speed = getattr(_m, "sound_speed", None)
        except ImportError:
            pass
    return law32_hill


try:
    from . import law43_hill_tab
    from .law43_hill_tab import (solid_update as law43_solid_update,
                                 shell_update as law43_shell_update,
                                 sound_speed as law43_sound_speed,
                                 consistent_solid_tangent as law43_solid_tangent,
                                 consistent_shell_tangent as law43_shell_tangent,
                                 shell_membrane_tangent as law43_membrane_tangent,
                                 build_law43)
except ImportError:
    law43_hill_tab = None
    law43_solid_update = None
    law43_shell_update = None
    law43_sound_speed = None
    law43_solid_tangent = None
    law43_shell_tangent = None
    law43_membrane_tangent = None
    build_law43 = None


def _get_law43():
    global law43_hill_tab, law43_solid_update, law43_shell_update, law43_sound_speed
    global law43_solid_tangent, law43_shell_tangent, law43_membrane_tangent, build_law43
    if law43_hill_tab is None:
        try:
            from . import law43_hill_tab as _m
            law43_hill_tab = _m
            law43_solid_update = getattr(_m, "solid_update", None)
            law43_shell_update = getattr(_m, "shell_update", None)
            law43_sound_speed = getattr(_m, "sound_speed", None)
            law43_solid_tangent = getattr(_m, "consistent_solid_tangent", None)
            law43_shell_tangent = getattr(_m, "consistent_shell_tangent", None)
            law43_membrane_tangent = getattr(_m, "shell_membrane_tangent", None)
            build_law43 = getattr(_m, "build_law43", None)
        except ImportError:
            pass
    return law43_hill_tab


try:
    from . import law25_composite
    from .law25_composite import (solid_update as law25_solid_update,
                                 shell_update as law25_shell_update,
                                 sound_speed as law25_sound_speed,
                                 consistent_solid_tangent as law25_solid_tangent,
                                 shell_membrane_tangent as law25_shell_tangent,
                                 build_law25,
                                 build_comp_plas,
                                 build_compsh,
                                 build_tsai_wu,
                                 build_crasurv,
                                 build_composite_plas)
except ImportError:
    law25_composite = None
    law25_solid_update = None
    law25_shell_update = None
    law25_sound_speed = None
    law25_solid_tangent = None
    law25_shell_tangent = None
    build_law25 = None
    build_comp_plas = None
    build_compsh = None
    build_tsai_wu = None
    build_crasurv = None
    build_composite_plas = None


try:
    from .law22_dama import build_law22
except ImportError:
    build_law22 = None


def _get_law25():
    global law25_composite, law25_solid_update, law25_shell_update, law25_sound_speed, law25_solid_tangent, law25_shell_tangent
    global build_law25, build_comp_plas, build_compsh, build_tsai_wu, build_crasurv, build_composite_plas
    if law25_composite is None:
        try:
            from . import law25_composite as _m
            law25_composite = _m
            law25_solid_update = getattr(_m, "solid_update", None)
            law25_shell_update = getattr(_m, "shell_update", None)
            law25_sound_speed = getattr(_m, "sound_speed", None)
            law25_solid_tangent = getattr(_m, "consistent_solid_tangent", None)
            law25_shell_tangent = getattr(_m, "shell_membrane_tangent", None)
            build_law25 = getattr(_m, "build_law25", None)
            build_comp_plas = getattr(_m, "build_comp_plas", None)
            build_compsh = getattr(_m, "build_compsh", None)
            build_tsai_wu = getattr(_m, "build_tsai_wu", None)
            build_crasurv = getattr(_m, "build_crasurv", None)
            build_composite_plas = getattr(_m, "build_composite_plas", None)
        except ImportError:
            pass
    return law25_composite


try:
    from . import law05_jwl
    from .law05_jwl import (solid_update as law05_solid_update,
                             sound_speed as law05_sound_speed,
                             consistent_solid_tangent as law05_solid_tangent)
except ImportError:
    law05_jwl = None
    law05_solid_update = None
    law05_sound_speed = None
    law05_solid_tangent = None


def _get_law05():
    global law05_jwl, law05_solid_update, law05_sound_speed, law05_solid_tangent
    if law05_jwl is None:
        try:
            from . import law05_jwl as _m
            law05_jwl = _m
            law05_solid_update = getattr(_m, "solid_update", None)
            law05_sound_speed = getattr(_m, "sound_speed", None)
            law05_solid_tangent = getattr(_m, "consistent_solid_tangent", None)
        except ImportError:
            pass
    return law05_jwl


def _register_law05():
    _get_law05()
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law05_jwl is not None:
            builder = getattr(law05_jwl, "build_law05", getattr(law05_jwl, "build_jwl", getattr(law05_jwl, "_builder", None)))
        if builder is None:
            def _dynamic_law05_builder(rec):
                mod = _get_law05()
                if mod is not None:
                    fn = getattr(mod, "build_law05", getattr(mod, "build_jwl", None))
                    if fn is not None:
                        return fn(rec)
                raise NotImplementedError("LAW5 builder not available in law05_jwl")
            builder = _dynamic_law05_builder
        for k in (5, "5", "LAW5", "JWL"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law05()


try:
    from . import law10_soil
    from .law10_soil import (solid_update as law10_solid_update,
                             sound_speed as law10_sound_speed,
                             consistent_solid_tangent as law10_solid_tangent)
except ImportError:
    law10_soil = None
    law10_solid_update = None
    law10_sound_speed = None
    law10_solid_tangent = None


def _get_law10():
    global law10_soil, law10_solid_update, law10_sound_speed, law10_solid_tangent
    if law10_soil is None:
        try:
            from . import law10_soil as _m
            law10_soil = _m
            law10_solid_update = getattr(_m, "solid_update", None)
            law10_sound_speed = getattr(_m, "sound_speed", None)
            law10_solid_tangent = getattr(_m, "consistent_solid_tangent", None)
        except ImportError:
            pass
    return law10_soil


def _register_law10():
    _get_law10()
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law10_soil is not None:
            builder = getattr(law10_soil, "build_law10", getattr(law10_soil, "build_soil", getattr(law10_soil, "_builder", None)))
        if builder is None:
            def _dynamic_law10_builder(rec):
                mod = _get_law10()
                if mod is not None:
                    fn = getattr(mod, "build_law10", getattr(mod, "build_soil", None))
                    if fn is not None:
                        return fn(rec)
                raise NotImplementedError("LAW10 builder not available in law10_soil")
            builder = _dynamic_law10_builder
        for k in (10, "10", "LAW10", "SOIL", "DPRAG", "DPRAG1"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law10()


def _register_law28():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law28_honeycomb, "build_law28", getattr(law28_honeycomb, "build_honeycomb", None))
        if builder is not None:
            for k in (28, "28", "LAW28", "HONEYCOMB", "HONEYCOMB_SOL"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law28()


def _register_law34():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law34_boltzmann, "build_law34", None)
        if builder is not None:
            for k in (34, "34", "LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law34()


def _register_law37():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law37_biphas, "build_law37", None)
        if builder is not None:
            for k in (37, "37", "LAW37", "BIPHAS", "BIPHASIC"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law37()


def _get_law38():
    global law38_visc_tab, law38_solid_update, law38_sound_speed, law38_solid_tangent
    if law38_visc_tab is None:
        try:
            from . import law38_visc_tab as _m
            law38_visc_tab = _m
            law38_solid_update = getattr(_m, "solid_update", None)
            law38_sound_speed = getattr(_m, "sound_speed", None)
            law38_solid_tangent = getattr(_m, "consistent_solid_tangent", None)
        except ImportError:
            pass
    return law38_visc_tab


def _register_law38():
    _get_law38()
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law38_visc_tab is not None:
            builder = getattr(law38_visc_tab, "build_law38", None)
        if builder is None:
            def _dynamic_law38_builder(rec):
                mod = _get_law38()
                if mod is not None:
                    fn = getattr(mod, "build_law38", None)
                    if fn is not None:
                        return fn(rec)
                raise NotImplementedError("LAW38 builder not available in law38_visc_tab")
            builder = _dynamic_law38_builder
        for k in (38, "38", "LAW38", "VISC_TAB", "MAT_LAW38", "MAT_VISC_TAB"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law38()


def _register_law32():
    _get_law32()
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law32_hill is not None:
            builder = getattr(law32_hill, "build_law32", getattr(law32_hill, "build_hill", None))
        if builder is None:
            def _dynamic_law32_builder(rec):
                mod = _get_law32()
                if mod is not None:
                    fn = getattr(mod, "build_law32", getattr(mod, "build_hill", None))
                    if fn is not None:
                        return fn(rec)
                from ..model.entities import Material
                params = dict(rec.params) if hasattr(rec, "params") else {}
                density = getattr(rec, "density", 0.0)
                mid = getattr(rec, "id", 0)
                title = getattr(rec, "title", "")
                return Material(id=mid, law=32, rho0=density, title=title, law_name="LAW32", params=params)
            builder = _dynamic_law32_builder
        for k in (32, "32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law32()


def _register_law43():
    _get_law43()
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law43_hill_tab is not None:
            builder = getattr(law43_hill_tab, "build_law43", None)
        if builder is None:
            def _dynamic_law43_builder(rec):
                mod = _get_law43()
                if mod is not None:
                    fn = getattr(mod, "build_law43", None)
                    if fn is not None:
                        return fn(rec)
                from ..model.entities import Material
                params = dict(rec.params) if hasattr(rec, "params") else {}
                density = getattr(rec, "density", 0.0)
                mid = getattr(rec, "id", 0)
                title = getattr(rec, "title", "")
                return Material(id=mid, law=43, rho0=density, title=title, law_name="LAW43", params=params)
            builder = _dynamic_law43_builder
        for k in (43, "43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law43()


def _register_law82():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law82_ogden, "build_law82", None)
        if builder is not None:
            for k in (82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN", "OGDEN_82", "MAT_OGDEN_82"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law82()


def _register_law69():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law69_hyperelastic, "build_law69", None)
        if builder is not None:
            for k in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law69()


def _get_law60():
    global law60_plast3, law60_solid_update, law60_shell_update, law60_sound_speed
    global law60_consistent_solid_tangent, law60_consistent_shell_tangent, law60_solid_tangent, law60_shell_tangent
    global build_law60, Law60Params
    if law60_plast3 is None:
        try:
            from . import law60_plast3 as _m
            law60_plast3 = _m
            Law60Params = getattr(_m, "Law60Params", None)
            build_law60 = getattr(_m, "build_law60", None)
            law60_solid_update = getattr(_m, "solid_update", None)
            law60_shell_update = getattr(_m, "shell_update", None)
            law60_sound_speed = getattr(_m, "sound_speed", None)
            law60_consistent_solid_tangent = getattr(_m, "consistent_solid_tangent", None)
            law60_consistent_shell_tangent = getattr(_m, "consistent_shell_tangent", None)
            law60_solid_tangent = getattr(_m, "consistent_solid_tangent", None)
            law60_shell_tangent = getattr(_m, "consistent_shell_tangent", None)
        except ImportError:
            pass
    return law60_plast3


def _register_law60():
    _get_law60()
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law60_plast3 is not None:
            builder = getattr(law60_plast3, "build_law60", getattr(law60_plast3, "build_plast3", None))
        if builder is None:
            def _dynamic_law60_builder(rec):
                mod = _get_law60()
                if mod is not None:
                    fn = getattr(mod, "build_law60", getattr(mod, "build_plast3", None))
                    if fn is not None:
                        return fn(rec)
                from ..model.entities import Material
                params = dict(rec.params) if hasattr(rec, "params") else {}
                density = getattr(rec, "density", 0.0)
                mid = getattr(rec, "id", 0)
                title = getattr(rec, "title", "")
                return Material(id=mid, law=60, rho0=density, title=title, law_name="LAW60", params=params)
            builder = _dynamic_law60_builder
        for k in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law60()


def _register_law48():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law48_zhao, "build_law48", None)
        if builder is None:
            def _dynamic_law48_builder(rec):
                from ..model.entities import Material
                params = dict(rec.params) if hasattr(rec, "params") else {}
                density = getattr(rec, "density", 0.0)
                mid = getattr(rec, "id", 0)
                title = getattr(rec, "title", "")
                return Material(id=mid, law=48, rho0=density, title=title, law_name="LAW48", params=params)
            builder = _dynamic_law48_builder
        for k in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO",
                  "MAT_LAW48", "MAT_ZHAO", "PLAS_ZHAO", "LAW48_ZHAO"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law48()


def _register_law58():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law58_fabr_a, "build_law58", None)
        if builder is None:
            def _dynamic_law58_builder(rec):
                from ..model.entities import Material
                params = dict(rec.params) if hasattr(rec, "params") else {}
                density = getattr(rec, "density", 0.0)
                mid = getattr(rec, "id", 0)
                title = getattr(rec, "title", "")
                return Material(id=mid, law=58, rho0=density, title=title, law_name="LAW58", params=params)
            builder = _dynamic_law58_builder
        for k in (58, "58", "LAW58", "FABR_A", "FABRIC_A",
                  "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law58()


def _register_law52():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law52_gurson, "build_law52", None)
        if builder is None:
            def _dynamic_law52_builder(rec):
                from ..model.entities import Material
                params = dict(rec.params) if hasattr(rec, "params") else {}
                density = getattr(rec, "rho0", getattr(rec, "rho", getattr(rec, "density", 0.0)))
                mid = getattr(rec, "id", 0)
                title = getattr(rec, "title", "")
                return Material(id=mid, law=52, rho0=density, title=title, law_name="LAW52", params=params)
            builder = _dynamic_law52_builder
        for k in (52, "52", "LAW52", "GURSON", "PLAS_GURS",
                  "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law52()


def _register_law57():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law57_barlat, "build_law57", None)
        if builder is None:
            def _dynamic_law57_builder(rec):
                from ..model.entities import Material
                params = dict(rec.params) if hasattr(rec, "params") else {}
                density = getattr(rec, "rho0", getattr(rec, "rho", getattr(rec, "density", 0.0)))
                mid = getattr(rec, "id", 0)
                title = getattr(rec, "title", "")
                return Material(id=mid, law=57, rho0=density, title=title, law_name="LAW57", params=params)
            builder = _dynamic_law57_builder
        for k in (57, "57", "LAW57", "BARLAT", "BARLAT3",
                  "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT3"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law57()


def _register_law25():
    _get_law25()
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law25_composite is not None:
            builder = getattr(law25_composite, "build_law25", None)
        if builder is None:
            def _dynamic_law25_builder(rec):
                mod = _get_law25()
                if mod is not None:
                    fn = getattr(mod, "build_law25", None)
                    if fn is not None:
                        return fn(rec)
                from ..model.entities import Material
                params = dict(rec.params) if hasattr(rec, "params") else {}
                density = getattr(rec, "density", 0.0)
                mid = getattr(rec, "id", 0)
                title = getattr(rec, "title", "")
                return Material(id=mid, law=25, rho0=density, title=title, law_name="LAW25", params=params)
            builder = _dynamic_law25_builder
        for k in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS",
                  "MAT_LAW25", "MAT_COMP_PLAS", "MAT_COMPSH", "MAT_TSAI_WU", "MAT_CRASURV", "MAT_COMPOSITE_PLAS"):
            MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law25()


def _register_law15():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law15_chang is not None:
            builder = getattr(law15_chang, "build_law15", None)
        if builder is not None:
            for k in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG",
                      "MAT_LAW15", "MAT_CHANG", "MAT_PLAS_ANISO", "MAT_COMP_CHANG"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law15()


def _register_law22():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law22_dama is not None:
            builder = getattr(law22_dama, "build_law22", None)
        if builder is not None:
            for k in (22, "22", "LAW22", "DAMA", "PLAS_DAMA",
                      "MAT_LAW22", "MAT_DAMA", "MAT_PLAS_DAMA"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law22()


def _register_law21():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law21_dprag is not None:
            builder = getattr(law21_dprag, "build_law21", None)
        if builder is not None:
            for k in (21, "21", "LAW21", "DPRAG",
                      "MAT_LAW21", "MAT_DPRAG", "LAW21_DPRAG"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law21()


def _register_law49():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law49_steinb, "build_law49", None)
        if builder is not None:
            for k in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN",
                      "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law49()


def _register_law79():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law79_john_holm, "build_law79", None)
        if builder is not None:
            for k in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2",
                      "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law79()


def _register_law50():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law50_visc_honey, "build_law50", None)
        if builder is not None:
            for k in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM",
                      "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law50()


def _register_law163():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law163_crush_foam, "build_law163", None)
        if builder is not None:
            for k in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM",
                      "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM", "LAW163_CRUSHABLE_FOAM"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law163()


def _register_law12():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law12_comp3d is not None:
            builder = getattr(law12_comp3d, "build_law12", None)
        if builder is not None:
            for k in (12, "12", "LAW12", "3D_COMP", "COMP_3D",
                      "MAT_LAW12", "MAT_3D_COMP", "MAT_COMP_3D",
                      "3PARBI", "MAT_3PARBI", "LAW12_3PARBI", "LAW12_3D_COMP"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law12()


def _register_law14():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = None
        if law14_compso is not None:
            builder = getattr(law14_compso, "build_law14", None)
        if builder is not None:
            for k in (14, "14", "LAW14", "COMPSO", "COMP_SOL",
                      "MAT_LAW14", "MAT_COMPSO", "MAT_COMP_SOL"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law14()


def _register_law73():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law73_hill_therm, "build_law73", None)
        if builder is not None:
            for k in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL",
                      "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL",
                      "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


def _register_law66():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law66_plas_tab, "build_law66", None)
        if builder is not None:
            for k in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB",
                      "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law66()


_register_law73()


_LAW71_KEYS = (
    71, "71", "LAW71", "SUPER_ELAS", "NITINOL",
    "MAT_71", "MAT_LAW71", "MAT_SUPER_ELAS", "MAT_NITINOL",
    "LAW71_SUPER_ELAS", "LAW71_NITINOL",
)


def _register_law71():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law71_nitinol, "build_law71", None)
        if builder is not None:
            for k in _LAW71_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law71()


_LAW76_KEYS = (
    76, "76", "LAW76", "SAMP", "SAMP-1", "SAMP_1",
    "MAT_76", "MAT_LAW76", "MAT_SAMP", "MAT_SAMP-1", "MAT_SAMP_1",
    "LAW76_SAMP", "LAW76_SAMP-1", "LAW76_SAMP_1",
)


def _register_law76():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law76_samp, "build_law76", None)
        if builder is not None:
            for k in _LAW76_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law76()


def _register_law74():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law74_hill_3d, "build_law74", None)
        if builder is not None:
            for k in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL",
                      "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL",
                      "LAW74_HILL_3D"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law74()


def _register_law87():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law87_barlat2000, "build_law87", None)
        if builder is not None:
            for k in (87, "87", "LAW87", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D",
                      "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT2000", "MAT_BARLAT_2000",
                      "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law87()


def _register_law88():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law88_tab_hyp, "build_law88", None)
        if builder is not None:
            for k in (88, "88", "LAW88", "HYP_TAB", "TAB_HYP", "HYPER_ELAS", "TABULATED_HYPERELASTIC",
                      "TABULATED_HYP", "MAT_LAW88", "MAT_HYP_TAB", "MAT_TAB_HYP", "MAT_HYPER_ELAS",
                      "MAT_TABULATED_HYPERELASTIC", "LAW88_TAB_HYP"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law88()

_LAW88_KEYS = (
    88, "88", "LAW88", "HYP_TAB", "TAB_HYP", "HYPER_ELAS", "TABULATED_HYPERELASTIC",
    "TABULATED_HYP", "MAT_LAW88", "MAT_HYP_TAB", "MAT_TAB_HYP", "MAT_HYPER_ELAS",
    "MAT_TABULATED_HYPERELASTIC", "LAW88_TAB_HYP",
)


def _register_law92():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law92_arruda_boyce, "build_law92", None)
        if builder is not None:
            for k in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE",
                      "MAT_LAW92", "MAT_ARRUDA_BOYCE", "LAW92_ARRUDA_BOYCE"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law92()

_LAW92_KEYS = (
    92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE",
    "MAT_LAW92", "MAT_ARRUDA_BOYCE", "LAW92_ARRUDA_BOYCE",
)


def _register_law94():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law94_yeoh, "build_law94", None)
        if builder is not None:
            for k in (94, "94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH", "LAW94_YEOH"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law94()

_LAW94_KEYS = (
    94, "94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH", "LAW94_YEOH",
)


def _register_law93():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law93_orth_hill, "build_law93", None)
        if builder is not None:
            for k in (93, "93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law93()

_LAW93_KEYS = (
    93, "93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL",
)


def _register_law95():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law95_bergstrom_boyce, "build_law95", None)
        if builder is not None:
            for k in (95, "95", "LAW95", "BERGSTROM_BOYCE", "BERGSTROM-BOYCE", "MAT_LAW95", "MAT_BERGSTROM_BOYCE", "LAW95_BERGSTROM_BOYCE"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law95()

_LAW95_KEYS = (
    95, "95", "LAW95", "BERGSTROM_BOYCE", "BERGSTROM-BOYCE", "MAT_LAW95", "MAT_BERGSTROM_BOYCE", "LAW95_BERGSTROM_BOYCE",
)


def _register_law100():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law100_multi_network, "build_law100", None)
        if builder is not None:
            for k in (100, "100", "LAW100", "VISC_HYP", "MNF", "MAT_LAW100", "MAT_VISC_HYP", "MAT_MNF", "LAW100_VISC_HYP", "LAW100_MNF", "SPOTWELD", "STRUCTURAL_ADHESIVE"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law100()

_LAW100_KEYS = (
    100, "100", "LAW100", "VISC_HYP", "MNF", "MAT_LAW100", "MAT_VISC_HYP", "MAT_MNF", "LAW100_VISC_HYP", "LAW100_MNF", "SPOTWELD", "STRUCTURAL_ADHESIVE",
)


def _register_law101():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law101_plas_poly, "build_law101", None)
        if builder is not None:
            for k in (101, "101", "LAW101", "PP", "MAT_PP", "PLAS_POLY", "MAT_PLAS_POLY", "MAT_LAW101", "LAW101_PP", "LAW101_PLAS_POLY"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law101()

_LAW101_KEYS = (
    101, "101", "LAW101", "PP", "MAT_PP", "PLAS_POLY", "MAT_PLAS_POLY", "MAT_LAW101", "LAW101_PP", "LAW101_PLAS_POLY",
)


def _register_law102():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law102_dprag2, "build_law102", None)
        if builder is not None:
            for k in (102, "102", "LAW102", "DPRAG2", "MAT_DPRAG2", "MAT_LAW102", "LAW102_DPRAG2", "DRUCKER_PRAGER_2"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law102()

_LAW102_KEYS = (
    102, "102", "LAW102", "DPRAG2", "MAT_DPRAG2", "MAT_LAW102", "LAW102_DPRAG2", "DRUCKER_PRAGER_2",
)


def _register_law103():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law103_hensel_spittel, "build_law103", None)
        if builder is not None:
            for k in (103, "103", "LAW103", "HENSEL_SPITTEL", "HENSEL-SPITTEL", "PLAS_HENS", "MAT_HENSEL_SPITTEL", "MAT_PLAS_HENS", "MAT_103", "MAT_LAW103", "HEN"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law103()

_LAW103_KEYS = (
    103, "103", "LAW103", "HENSEL_SPITTEL", "HENSEL-SPITTEL", "PLAS_HENS", "MAT_HENSEL_SPITTEL", "MAT_PLAS_HENS", "MAT_103", "MAT_LAW103", "HEN",
)


def _register_law104():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law104_drucker, "build_law104", None)
        if builder is not None:
            for k in (104, "104", "LAW104", "DRUCKER", "JOHNS_VOCE_DRUCKER", "JOHNS-VOCE-DRUCKER", "PLAS_DRUCK", "MAT_DRUCKER", "MAT_JOHNS_VOCE_DRUCKER", "MAT_PLAS_DRUCK", "MAT_104", "MAT_LAW104", "LAW104_DRUCKER"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law104()

_LAW104_KEYS = (
    104, "104", "LAW104", "DRUCKER", "JOHNS_VOCE_DRUCKER", "JOHNS-VOCE-DRUCKER", "PLAS_DRUCK",
    "MAT_DRUCKER", "MAT_JOHNS_VOCE_DRUCKER", "MAT_PLAS_DRUCK", "MAT_104", "MAT_LAW104", "LAW104_DRUCKER",
)


def _register_law105():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law105_powder_burn, "build_law105", None)
        if builder is not None:
            for k in (105, "105", "LAW105", "POWDER_BURN", "POWDERBURN", "MAT_POWDER_BURN", "MAT_POWDERBURN", "MAT_105", "MAT_LAW105", "LAW105_POWDER_BURN"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law105()

_LAW105_KEYS = (
    105, "105", "LAW105", "POWDER_BURN", "POWDERBURN", "MAT_POWDER_BURN", "MAT_POWDERBURN", "MAT_105", "MAT_LAW105", "LAW105_POWDER_BURN",
)


def _register_law106():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law106_jcook_alm, "build_law106", None)
        if builder is not None:
            for k in (106, "106", "LAW106", "JCOOK_ALM", "JOHNS_COOK_ALM", "MAT_JCOOK_ALM", "MAT_106", "MAT_LAW106", "LAW106_JCOOK_ALM"):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law106()

_LAW106_KEYS = (
    106, "106", "LAW106", "JCOOK_ALM", "JOHNS_COOK_ALM", "MAT_JCOOK_ALM", "MAT_106", "MAT_LAW106", "LAW106_JCOOK_ALM",
)


def _register_law107():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law107_paper_light, "build_law107", None)
        if builder is not None:
            for k in (
                107, "107", "LAW107", "PAPER_LIGHT", "PLAS_PAPER_LIGHT",
                "LAW107_PAPER_LIGHT", "PFEIFFER", "MAT_PFEIFFER",
                "MAT_PAPER_LIGHT", "MAT_PLAS_PAPER_LIGHT", "MAT_107", "MAT_LAW107"
            ):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law107()

_LAW107_KEYS = (
    107, "107", "LAW107", "PAPER_LIGHT", "PLAS_PAPER_LIGHT",
    "LAW107_PAPER_LIGHT", "PFEIFFER", "MAT_PFEIFFER",
    "MAT_PAPER_LIGHT", "MAT_PLAS_PAPER_LIGHT", "MAT_107", "MAT_LAW107"
)


def _register_law109():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law109_tab_plas, "build_law109", None)
        if builder is not None:
            for k in (
                109, "109", "LAW109", "TAB_PLAS", "ELASTO_PLAS_TAB",
                "LAW109_TAB_PLAS", "MLAW109", "MAT_LAW109", "MAT_TAB_PLAS",
                "MAT_ELASTO_PLAS_TAB", "MAT_109",
            ):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law109()

_LAW109_KEYS = (
    109, "109", "LAW109", "TAB_PLAS", "ELASTO_PLAS_TAB",
    "LAW109_TAB_PLAS", "MLAW109", "MAT_LAW109", "MAT_TAB_PLAS",
    "MAT_ELASTO_PLAS_TAB", "MAT_109",
)


def _register_law110():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law110_vegter, "build_law110", None)
        if builder is not None:
            for k in (
                110, "110", "LAW110", "VEGTER", "PLAS_VEGTER",
                "LAW110_VEGTER", "MLAW110", "MAT_LAW110", "MAT_VEGTER",
                "MAT_PLAS_VEGTER", "MAT_110",
            ):
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law110()

_LAW110_KEYS = (
    110, "110", "LAW110", "VEGTER", "PLAS_VEGTER",
    "LAW110_VEGTER", "MLAW110", "MAT_LAW110", "MAT_VEGTER",
    "MAT_PLAS_VEGTER", "MAT_110",
)

_LAW126_KEYS = (
    126, "126", "LAW126", "JOHNSON_HOLMQUIST_CONCRETE", "MAT_LAW126",
    "MAT_JOHNSON_HOLMQUIST_CONCRETE", "LAW126_JOHNSON_HOLMQUIST_CONCRETE", "HJC", "MAT_HJC",
)

_LAW169_KEYS = (
    169, "169", "LAW169", "ARUP_ADHESIVE", "MAT_LAW169", "MAT_ARUP_ADHESIVE",
    "LAW169_ARUP_ADHESIVE", "ARUP", "ADHESIVE",
)

_LAW117_KEYS = (
    117, "117", "LAW117", "COH_MC", "COH_TAB", "COHESIVE",
    "COHESIVE_TABULATED", "MAT_LAW117", "MAT_COH_MC", "MAT_COH_TAB",
    "LAW117_COH_MC", "LAW117_COH_TAB", "MLAW117",
)


def _register_law117():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law117_cohesive, "build_law117", None)
        if builder is not None:
            for k in _LAW117_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law117()

_LAW114_KEYS = (
    114, "114", "LAW114", "SPR_SEATBELT", "MAT_LAW114", "MAT_SPR_SEATBELT", "SEATBELT_SPR",
)


def _register_law114():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law114_seatbelt, "build_law114", None)
        if builder is not None:
            for k in _LAW114_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law114()

_LAW119_KEYS = (
    119, "119", "LAW119", "SH_SEATBELT", "SEATBELT_SH", "SEATBELT_SHELL", "MAT_LAW119", "MAT_SH_SEATBELT", "LAW119_SH_SEATBELT",
)


def _register_law119():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law119_seatbelt, "build_law119", None)
        if builder is not None:
            for k in _LAW119_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law119()

_LAW123_KEYS = (
    123, "123", "LAW123", "DAIMLER_PINHO", "DAIMLER-PINHO",
    "LAMINATED_FRACTURE_DAIMLER_PINHO", "MAT_LAW123", "MAT_DAIMLER_PINHO",
    "LAW123_DAIMLER_PINHO",
)

_LAW132_KEYS = (
    132, "132", "LAW132", "DAIMLER_CAMANHO", "DAIMLER-CAMANHO",
    "LAMINATED_FRACTURE_DAIMLER_CAMANHO", "MAT_LAW132", "MAT_DAIMLER_CAMANHO",
    "LAW132_DAIMLER_CAMANHO",
)


def _register_law123():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder123 = getattr(law123_daimler, "build_law123", None)
        if builder123 is not None:
            for k in _LAW123_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder123
        builder132 = getattr(law123_daimler, "build_law132", None)
        if builder132 is not None:
            for k in _LAW132_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder132
    except Exception:
        pass


_register_law123()

_LAW124_KEYS = (
    124, "124", "LAW124", "CDPM2", "CONCRETE_DAMAGE_PLASTICITY_2",
    "MAT_LAW124", "MAT_CDPM2", "LAW124_CDPM2", "MLAW124",
)


def _register_law124():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law124_cdpm2, "build_law124", None)
        if builder is not None:
            for k in _LAW124_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law124()

_LAW120_KEYS = (
    120, "120", "LAW120", "TAPO", "MAT_LAW120", "MAT_TAPO", "LAW120_TAPO",
    "TAB_PONT_ORTH", "MAT_TAB_PONT_ORTH", "MAT_120",
)


def _register_law120():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law120_tapo, "build_law120", None)
        if builder is not None:
            for k in _LAW120_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law120()

_LAW121_KEYS = (
    121, "121", "LAW121", "PLAS_RATE", "MAT_LAW121", "MAT_PLAS_RATE",
    "LAW121_PLAS_RATE", "PLAS_TAB_RATE", "MAT_PLAS_TAB_RATE", "MAT_121",
)


def _register_law121():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law121_plas_rate, "build_law121", None)
        if builder is not None:
            for k in _LAW121_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law121()

_LAW90_KEYS = (
    90, "90", "LAW90", "HYST_FOAM", "TAB_FOAM", "TABULAR_FOAM",
    "MAT_LAW90", "MAT_HYST_FOAM", "MAT_TAB_FOAM", "LAW90_HYST_FOAM", "LAW90_TAB_FOAM",
)

_LAW190_KEYS = (
    190, "190", "LAW190", "FOAM_DUBOIS", "DUBOIS",
    "MAT_LAW190", "MAT_FOAM_DUBOIS", "LAW190_FOAM_DUBOIS", "MAT_190",
)


def _register_law190():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        builder = getattr(law190_dubois, "build_law190", None)
        if builder is not None:
            for k in _LAW190_KEYS:
                MAT_PHYSICS_REGISTRY[k] = builder
    except Exception:
        pass


_register_law190()

_NEW_PORTED_LAWS: dict[int, Any] = {
    0: law00_void,
    11: law11_fluid_visc,
    13: law13_fluid_ale,
    16: law16_gray_ewing,
    17: law17_orth_elastic,
    18: law18_plas_iso,
    20: law20_rigid,
    23: law23_user_mat,
    26: law26_honeycomb_sesame,
    41: law41_jwl_burn,
    45: law45_orth_fabric,
    46: law46_kin_hard,
    51: law51_granular_soil,
    53: law53_tab_foam,
    54: law54_hyd_visc,
    55: law55_shell_orth,
    56: law56_orth_fail,
    59: law59_spring_rate,
    63: law63_composite_shell,
    64: law64_rate_comp,
    65: law65_thermo_visc,
    68: law68_tab_plas,
    72: law72_hill_mmc,
    75: law75_thermo_trans,
    77: law77_polymer,
    78: law78_yoshida,
    80: law80_ramberg,
    84: law84_mooney_rivlin,
    85: law85_void_pinch,
    86: law86_honeycomb_shell,
    91: law91_pinch_shell,
    96: law96_thermovp,
    97: law97_orth_nonlinear,
    98: law98_loss_optim,
    108: law108_yield_fit,
    111: law111_barlat_yld2004,
    112: law112_xia_damage,
    113: law113_yield_curve_fit,
    115: law115_perzyna,
    116: law116_spotweld,
    118: law118_reserved,
    122: law122_chaboche,
    125: law125_tab_aniso_rate,
    127: law127_mohr_coulomb,
    128: law128_hill_rate_shell,
    129: law129_barlat_yld2000,
    130: law130_nonlocal_damage,
    131: law131_modular_elastoplas,
    132: law132_rate_modifier,
    133: law133_microplane_concrete,
    134: law134_fabric_wrinkle,
    135: law135_modular_user,
    151: law151_multimat_ale,
    158: law158_progressive_composite,
    187: law187_samp1_polymer,
}


def _resolve_law_num(mat: Any) -> Optional[int]:
    """Resolve integer law number from Material entity, card_dict, law attribute, or law_name."""
    for attr in ("law", "law_name", "type"):
        val = getattr(mat, attr, None)
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            val_clean = val.strip().upper()
            if val_clean.startswith("/MAT/LAW"):
                digits = val_clean[8:]
                if digits.isdigit():
                    return int(digits)
            elif val_clean.startswith("LAW"):
                digits = val_clean[3:]
                if digits.isdigit():
                    return int(digits)
            elif val_clean.startswith("MAT_LAW"):
                digits = val_clean[7:]
                if digits.isdigit():
                    return int(digits)
            elif val_clean.startswith("MAT_"):
                digits = val_clean[4:]
                if digits.isdigit():
                    return int(digits)
            elif val_clean.isdigit():
                return int(val_clean)
    if hasattr(mat, "card_dict") and isinstance(mat.card_dict, dict):
        for k in ("law", "LAW", "law_num", "mat_law"):
            if k in mat.card_dict:
                try:
                    return int(mat.card_dict[k])
                except (ValueError, TypeError):
                    pass
    return None


_STATE_VAR_COUNT: dict[str, tuple[int, ...]] = {
    "uv190": (16,),
    "uv90": (10,),
    "uv15": (8,),
    "uv22": (4,),
    "uv25": (12,),
    "uv32": (2,),
    "uv37": (5,),
    "uv38": (33,),
    "uv43": (4,),
    "uv50": (6,),
    "uvar66": (8,),
    "uv66": (8,),
    "uvar73": (7,),
    "uvar74": (10,),
    "uvar87": (1,),
    "uv88": (30,),
    "uvar88": (30,),
    "uvar93": (1,),
}

LAW_DISPATCH_METADATA: dict[Any, dict[str, Any]] = {
    48: {"plane_stress": True, "solid": True, "shell": True},
    "48": {"plane_stress": True, "solid": True, "shell": True},
    "LAW48": {"plane_stress": True, "solid": True, "shell": True},
    "ZHAO": {"plane_stress": True, "solid": True, "shell": True},
    "PLAS_ZHAO": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW48": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_ZHAO": {"plane_stress": True, "solid": True, "shell": True},
    "LAW48_ZHAO": {"plane_stress": True, "solid": True, "shell": True},
    52: {"plane_stress": True, "solid": True, "shell": True},
    "52": {"plane_stress": True, "solid": True, "shell": True},
    "LAW52": {"plane_stress": True, "solid": True, "shell": True},
    "GURSON": {"plane_stress": True, "solid": True, "shell": True},
    "PLAS_GURS": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW52": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_GURSON": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_PLAS_GURS": {"plane_stress": True, "solid": True, "shell": True},
    58: {"plane_stress": True, "solid": False, "shell": True},
    "58": {"plane_stress": True, "solid": False, "shell": True},
    "LAW58": {"plane_stress": True, "solid": False, "shell": True},
    "FABR_A": {"plane_stress": True, "solid": False, "shell": True},
    "FABRIC_A": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_LAW58": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_FABR_A": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_FABRIC_A": {"plane_stress": True, "solid": False, "shell": True},
    "LAW58_FABR_A": {"plane_stress": True, "solid": False, "shell": True},
    57: {"plane_stress": True, "solid": False, "shell": True},
    "57": {"plane_stress": True, "solid": False, "shell": True},
    "LAW57": {"plane_stress": True, "solid": False, "shell": True},
    "BARLAT": {"plane_stress": True, "solid": False, "shell": True},
    "BARLAT3": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_LAW57": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_BARLAT": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_BARLAT3": {"plane_stress": True, "solid": False, "shell": True},
    "LAW57_BARLAT": {"plane_stress": True, "solid": False, "shell": True},
    "LAW57_BARLAT3": {"plane_stress": True, "solid": False, "shell": True},
    60: {"plane_stress": True, "solid": True, "shell": True},
    "60": {"plane_stress": True, "solid": True, "shell": True},
    "LAW60": {"plane_stress": True, "solid": True, "shell": True},
    "PLAS_T3": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW60": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_PLAS_T3": {"plane_stress": True, "solid": True, "shell": True},
    "FABRIC": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_FABRIC": {"plane_stress": True, "solid": True, "shell": True},
    21: {"plane_stress": False, "solid": True, "shell": False},
    "21": {"plane_stress": False, "solid": True, "shell": False},
    "LAW21": {"plane_stress": False, "solid": True, "shell": False},
    "DPRAG": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW21": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_DPRAG": {"plane_stress": False, "solid": True, "shell": False},
    "LAW21_DPRAG": {"plane_stress": False, "solid": True, "shell": False},
    49: {"plane_stress": False, "solid": True, "shell": False},
    "49": {"plane_stress": False, "solid": True, "shell": False},
    "LAW49": {"plane_stress": False, "solid": True, "shell": False},
    "STEINB": {"plane_stress": False, "solid": True, "shell": False},
    "STEINBERG": {"plane_stress": False, "solid": True, "shell": False},
    "STEINBERG_GUINAN": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW49": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_STEINB": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_STEINBERG": {"plane_stress": False, "solid": True, "shell": False},
    "LAW49_STEINB": {"plane_stress": False, "solid": True, "shell": False},
    79: {"plane_stress": False, "solid": True, "shell": False},
    "79": {"plane_stress": False, "solid": True, "shell": False},
    "LAW79": {"plane_stress": False, "solid": True, "shell": False},
    "JOHN_HOLM": {"plane_stress": False, "solid": True, "shell": False},
    "JOHNSON_HOLMQUIST": {"plane_stress": False, "solid": True, "shell": False},
    "JH2": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW79": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_JOHN_HOLM": {"plane_stress": False, "solid": True, "shell": False},
    "LAW79_JOHN_HOLM": {"plane_stress": False, "solid": True, "shell": False},
    50: {"plane_stress": False, "solid": True, "shell": False},
    "50": {"plane_stress": False, "solid": True, "shell": False},
    "LAW50": {"plane_stress": False, "solid": True, "shell": False},
    "VISC_HONEY": {"plane_stress": False, "solid": True, "shell": False},
    "HYP_FOAM": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW50": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_VISC_HONEY": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_HYP_FOAM": {"plane_stress": False, "solid": True, "shell": False},
    73: {"plane_stress": True, "solid": False, "shell": True},
    "73": {"plane_stress": True, "solid": False, "shell": True},
    "LAW73": {"plane_stress": True, "solid": False, "shell": True},
    "BARLAT2000": {"plane_stress": True, "solid": False, "shell": True},
    "HILL_THERM": {"plane_stress": True, "solid": False, "shell": True},
    "THERM_HILL": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_LAW73": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_BARLAT2000": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_HILL_THERM": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_THERM_HILL": {"plane_stress": True, "solid": False, "shell": True},
    "LAW73_HILL_THERM": {"plane_stress": True, "solid": False, "shell": True},
    "LAW73_THERM_HILL": {"plane_stress": True, "solid": False, "shell": True},
    71: {"plane_stress": True, "solid": True, "shell": True},
    "71": {"plane_stress": True, "solid": True, "shell": True},
    "LAW71": {"plane_stress": True, "solid": True, "shell": True},
    "SUPER_ELAS": {"plane_stress": True, "solid": True, "shell": True},
    "NITINOL": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_71": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW71": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_SUPER_ELAS": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_NITINOL": {"plane_stress": True, "solid": True, "shell": True},
    "LAW71_SUPER_ELAS": {"plane_stress": True, "solid": True, "shell": True},
    "LAW71_NITINOL": {"plane_stress": True, "solid": True, "shell": True},
    76: {"plane_stress": True, "solid": True, "shell": True},
    "76": {"plane_stress": True, "solid": True, "shell": True},
    "LAW76": {"plane_stress": True, "solid": True, "shell": True},
    "SAMP": {"plane_stress": True, "solid": True, "shell": True},
    "SAMP-1": {"plane_stress": True, "solid": True, "shell": True},
    "SAMP_1": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_76": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW76": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_SAMP": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_SAMP-1": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_SAMP_1": {"plane_stress": True, "solid": True, "shell": True},
    "LAW76_SAMP": {"plane_stress": True, "solid": True, "shell": True},
    "LAW76_SAMP-1": {"plane_stress": True, "solid": True, "shell": True},
    "LAW76_SAMP_1": {"plane_stress": True, "solid": True, "shell": True},
    66: {"plane_stress": True, "solid": True, "shell": True},
    "66": {"plane_stress": True, "solid": True, "shell": True},
    "LAW66": {"plane_stress": True, "solid": True, "shell": True},
    "PLAS_TAB_COSSER": {"plane_stress": True, "solid": True, "shell": True},
    "PLAS_COSSER": {"plane_stress": True, "solid": True, "shell": True},
    "FOAM_TAB": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW66": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_PLAS_TAB_COSSER": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_PLAS_COSSER": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_FOAM_TAB": {"plane_stress": True, "solid": True, "shell": True},
    74: {"plane_stress": False, "solid": True, "shell": False},
    "74": {"plane_stress": False, "solid": True, "shell": False},
    "LAW74": {"plane_stress": False, "solid": True, "shell": False},
    "HILL_3D": {"plane_stress": False, "solid": True, "shell": False},
    "ORTH_PLAS": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW74": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_HILL_3D": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_ORTH_PLAS": {"plane_stress": False, "solid": True, "shell": False},
    "LAW74_HILL_3D": {"plane_stress": False, "solid": True, "shell": False},
    77: {"plane_stress": True, "solid": True, "shell": True},
    "77": {"plane_stress": True, "solid": True, "shell": True},
    "LAW77": {"plane_stress": True, "solid": True, "shell": True},
    "FOAM_AIR": {"plane_stress": True, "solid": True, "shell": True},
    "LAW77_POLYMER": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW77": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_FOAM_AIR": {"plane_stress": True, "solid": True, "shell": True},
    87: {"plane_stress": True, "solid": False, "shell": True},
    "87": {"plane_stress": True, "solid": False, "shell": True},
    "LAW87": {"plane_stress": True, "solid": False, "shell": True},
    "BARLAT_2000": {"plane_stress": True, "solid": False, "shell": True},
    "BARLAT2000_2D": {"plane_stress": True, "solid": False, "shell": True},
    "BARLAT_YLD2000": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_LAW87": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_BARLAT_2000": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_BARLAT2000_2D": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_BARLAT_YLD2000": {"plane_stress": True, "solid": False, "shell": True},
    "LAW87_BARLAT2000": {"plane_stress": True, "solid": False, "shell": True},
    88: {"plane_stress": True, "solid": True, "shell": True},
    "88": {"plane_stress": True, "solid": True, "shell": True},
    "LAW88": {"plane_stress": True, "solid": True, "shell": True},
    "HYP_TAB": {"plane_stress": True, "solid": True, "shell": True},
    "TAB_HYP": {"plane_stress": True, "solid": True, "shell": True},
    "HYPER_ELAS": {"plane_stress": True, "solid": True, "shell": True},
    "TABULATED_HYPERELASTIC": {"plane_stress": True, "solid": True, "shell": True},
    "TABULATED_HYP": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW88": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_HYP_TAB": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_TAB_HYP": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_HYPER_ELAS": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_TABULATED_HYPERELASTIC": {"plane_stress": True, "solid": True, "shell": True},
    "LAW88_TAB_HYP": {"plane_stress": True, "solid": True, "shell": True},
    92: {"plane_stress": True, "solid": True, "shell": True},
    "92": {"plane_stress": True, "solid": True, "shell": True},
    "LAW92": {"plane_stress": True, "solid": True, "shell": True},
    "ARRUDA_BOYCE": {"plane_stress": True, "solid": True, "shell": True},
    "ARRUDA-BOYCE": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW92": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_ARRUDA_BOYCE": {"plane_stress": True, "solid": True, "shell": True},
    "LAW92_ARRUDA_BOYCE": {"plane_stress": True, "solid": True, "shell": True},
    93: {"plane_stress": True, "solid": True, "shell": True},
    "93": {"plane_stress": True, "solid": True, "shell": True},
    "LAW93": {"plane_stress": True, "solid": True, "shell": True},
    "ORTH_HILL": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW93": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_ORTH_HILL": {"plane_stress": True, "solid": True, "shell": True},
    "LAW93_ORTH_HILL": {"plane_stress": True, "solid": True, "shell": True},
    94: {"plane_stress": True, "solid": True, "shell": True},
    "94": {"plane_stress": True, "solid": True, "shell": True},
    "LAW94": {"plane_stress": True, "solid": True, "shell": True},
    "YEOH": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW94": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_YEOH": {"plane_stress": True, "solid": True, "shell": True},
    "LAW94_YEOH": {"plane_stress": True, "solid": True, "shell": True},
    95: {"plane_stress": False, "solid": True, "shell": False},
    "95": {"plane_stress": False, "solid": True, "shell": False},
    "LAW95": {"plane_stress": False, "solid": True, "shell": False},
    "BERGSTROM_BOYCE": {"plane_stress": False, "solid": True, "shell": False},
    "BERGSTROM-BOYCE": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW95": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_BERGSTROM_BOYCE": {"plane_stress": False, "solid": True, "shell": False},
    "LAW95_BERGSTROM_BOYCE": {"plane_stress": False, "solid": True, "shell": False},
    100: {"plane_stress": False, "solid": True, "shell": False},
    "100": {"plane_stress": False, "solid": True, "shell": False},
    "LAW100": {"plane_stress": False, "solid": True, "shell": False},
    "VISC_HYP": {"plane_stress": False, "solid": True, "shell": False},
    "MNF": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW100": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_VISC_HYP": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_MNF": {"plane_stress": False, "solid": True, "shell": False},
    "LAW100_VISC_HYP": {"plane_stress": False, "solid": True, "shell": False},
    "LAW100_MNF": {"plane_stress": False, "solid": True, "shell": False},
    "SPOTWELD": {"plane_stress": False, "solid": True, "shell": False},
    "STRUCTURAL_ADHESIVE": {"plane_stress": False, "solid": True, "shell": False},
    101: {"plane_stress": False, "solid": True, "shell": False},
    "101": {"plane_stress": False, "solid": True, "shell": False},
    "LAW101": {"plane_stress": False, "solid": True, "shell": False},
    "PP": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_PP": {"plane_stress": False, "solid": True, "shell": False},
    "PLAS_POLY": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_PLAS_POLY": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW101": {"plane_stress": False, "solid": True, "shell": False},
    "LAW101_PP": {"plane_stress": False, "solid": True, "shell": False},
    "LAW101_PLAS_POLY": {"plane_stress": False, "solid": True, "shell": False},
    102: {"plane_stress": False, "solid": True, "shell": False},
    "102": {"plane_stress": False, "solid": True, "shell": False},
    "LAW102": {"plane_stress": False, "solid": True, "shell": False},
    "DPRAG2": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_DPRAG2": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW102": {"plane_stress": False, "solid": True, "shell": False},
    "LAW102_DPRAG2": {"plane_stress": False, "solid": True, "shell": False},
    "DRUCKER_PRAGER_2": {"plane_stress": False, "solid": True, "shell": False},
    103: {"plane_stress": False, "solid": True, "shell": False},
    "103": {"plane_stress": False, "solid": True, "shell": False},
    "LAW103": {"plane_stress": False, "solid": True, "shell": False},
    "HENSEL_SPITTEL": {"plane_stress": False, "solid": True, "shell": False},
    "HENSEL-SPITTEL": {"plane_stress": False, "solid": True, "shell": False},
    "PLAS_HENS": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_HENSEL_SPITTEL": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_PLAS_HENS": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_103": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW103": {"plane_stress": False, "solid": True, "shell": False},
    "HEN": {"plane_stress": False, "solid": True, "shell": False},
    104: {"plane_stress": True, "solid": True, "shell": True},
    "104": {"plane_stress": True, "solid": True, "shell": True},
    "LAW104": {"plane_stress": True, "solid": True, "shell": True},
    "DRUCKER": {"plane_stress": True, "solid": True, "shell": True},
    "JOHNS_VOCE_DRUCKER": {"plane_stress": True, "solid": True, "shell": True},
    "JOHNS-VOCE-DRUCKER": {"plane_stress": True, "solid": True, "shell": True},
    "PLAS_DRUCK": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_DRUCKER": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_JOHNS_VOCE_DRUCKER": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_PLAS_DRUCK": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_104": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW104": {"plane_stress": True, "solid": True, "shell": True},
    "LAW104_DRUCKER": {"plane_stress": True, "solid": True, "shell": True},
    105: {"plane_stress": False, "solid": True, "shell": False},
    "105": {"plane_stress": False, "solid": True, "shell": False},
    "LAW105": {"plane_stress": False, "solid": True, "shell": False},
    "POWDER_BURN": {"plane_stress": False, "solid": True, "shell": False},
    "POWDERBURN": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_POWDER_BURN": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_POWDERBURN": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_105": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW105": {"plane_stress": False, "solid": True, "shell": False},
    "LAW105_POWDER_BURN": {"plane_stress": False, "solid": True, "shell": False},
    106: {"plane_stress": True, "solid": True, "shell": True},
    "106": {"plane_stress": True, "solid": True, "shell": True},
    "LAW106": {"plane_stress": True, "solid": True, "shell": True},
    "JCOOK_ALM": {"plane_stress": True, "solid": True, "shell": True},
    "JOHNS_COOK_ALM": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_JCOOK_ALM": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_106": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW106": {"plane_stress": True, "solid": True, "shell": True},
    "LAW106_JCOOK_ALM": {"plane_stress": True, "solid": True, "shell": True},
    107: {"plane_stress": True, "solid": True, "shell": True},
    "107": {"plane_stress": True, "solid": True, "shell": True},
    "LAW107": {"plane_stress": True, "solid": True, "shell": True},
    "PAPER_LIGHT": {"plane_stress": True, "solid": True, "shell": True},
    "PLAS_PAPER_LIGHT": {"plane_stress": True, "solid": True, "shell": True},
    "LAW107_PAPER_LIGHT": {"plane_stress": True, "solid": True, "shell": True},
    "PFEIFFER": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_PFEIFFER": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_PAPER_LIGHT": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_PLAS_PAPER_LIGHT": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_107": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW107": {"plane_stress": True, "solid": True, "shell": True},
    109: {"plane_stress": True, "solid": True, "shell": True},
    "109": {"plane_stress": True, "solid": True, "shell": True},
    "LAW109": {"plane_stress": True, "solid": True, "shell": True},
    "TAB_PLAS": {"plane_stress": True, "solid": True, "shell": True},
    "ELASTO_PLAS_TAB": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW109": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_TAB_PLAS": {"plane_stress": True, "solid": True, "shell": True},
    "MLAW109": {"plane_stress": True, "solid": True, "shell": True},
    110: {"plane_stress": True, "solid": False, "shell": True},
    "110": {"plane_stress": True, "solid": False, "shell": True},
    "LAW110": {"plane_stress": True, "solid": False, "shell": True},
    "VEGTER": {"plane_stress": True, "solid": False, "shell": True},
    "PLAS_VEGTER": {"plane_stress": True, "solid": False, "shell": True},
    "LAW110_VEGTER": {"plane_stress": True, "solid": False, "shell": True},
    "MLAW110": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_LAW110": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_VEGTER": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_PLAS_VEGTER": {"plane_stress": True, "solid": False, "shell": True},
    "MAT_110": {"plane_stress": True, "solid": False, "shell": True},
    # M598: LAW124 (Concrete Damage Plasticity Model 2 / CDPM2)
    124: {"plane_stress": False, "solid": True, "shell": False},
    "124": {"plane_stress": False, "solid": True, "shell": False},
    "LAW124": {"plane_stress": False, "solid": True, "shell": False},
    "CDPM2": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW124": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_CDPM2": {"plane_stress": False, "solid": True, "shell": False},
    "LAW124_CDPM2": {"plane_stress": False, "solid": True, "shell": False},
    # M591: LAW126 (Johnson-Holmquist Concrete / HJC) and LAW169 (Arup Structural Adhesive)
    126: {"plane_stress": False, "solid": True, "shell": False},
    "126": {"plane_stress": False, "solid": True, "shell": False},
    "LAW126": {"plane_stress": False, "solid": True, "shell": False},
    "JOHNSON_HOLMQUIST_CONCRETE": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW126": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_JOHNSON_HOLMQUIST_CONCRETE": {"plane_stress": False, "solid": True, "shell": False},
    "HJC": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_HJC": {"plane_stress": False, "solid": True, "shell": False},
    169: {"plane_stress": True, "solid": True, "shell": True},
    "169": {"plane_stress": True, "solid": True, "shell": True},
    "LAW169": {"plane_stress": True, "solid": True, "shell": True},
    "ARUP_ADHESIVE": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW169": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_ARUP_ADHESIVE": {"plane_stress": True, "solid": True, "shell": True},
    "ARUP": {"plane_stress": True, "solid": True, "shell": True},
    "ADHESIVE": {"plane_stress": True, "solid": True, "shell": True},
    117: {"plane_stress": True, "solid": True, "shell": True},
    "117": {"plane_stress": True, "solid": True, "shell": True},
    "LAW117": {"plane_stress": True, "solid": True, "shell": True},
    "COH_MC": {"plane_stress": True, "solid": True, "shell": True},
    "COH_TAB": {"plane_stress": True, "solid": True, "shell": True},
    "COHESIVE": {"plane_stress": True, "solid": True, "shell": True},
    "COHESIVE_TABULATED": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW117": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_COH_MC": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_COH_TAB": {"plane_stress": True, "solid": True, "shell": True},
    "LAW117_COH_MC": {"plane_stress": True, "solid": True, "shell": True},
    "LAW117_COH_TAB": {"plane_stress": True, "solid": True, "shell": True},
    "MLAW117": {"plane_stress": True, "solid": True, "shell": True},
    90: {"plane_stress": False, "solid": True, "shell": False},
    "90": {"plane_stress": False, "solid": True, "shell": False},
    "LAW90": {"plane_stress": False, "solid": True, "shell": False},
    "HYST_FOAM": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW90": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_HYST_FOAM": {"plane_stress": False, "solid": True, "shell": False},
    # M600: LAW120 (TAPO) and LAW121 (PLAS_RATE)
    120: {"plane_stress": True, "solid": True, "shell": True},
    "120": {"plane_stress": True, "solid": True, "shell": True},
    "LAW120": {"plane_stress": True, "solid": True, "shell": True},
    "TAPO": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW120": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_TAPO": {"plane_stress": True, "solid": True, "shell": True},
    "LAW120_TAPO": {"plane_stress": True, "solid": True, "shell": True},
    "TAB_PONT_ORTH": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_TAB_PONT_ORTH": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_120": {"plane_stress": True, "solid": True, "shell": True},
    121: {"plane_stress": True, "solid": True, "shell": True},
    "121": {"plane_stress": True, "solid": True, "shell": True},
    "LAW121": {"plane_stress": True, "solid": True, "shell": True},
    "PLAS_RATE": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_LAW121": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_PLAS_RATE": {"plane_stress": True, "solid": True, "shell": True},
    "LAW121_PLAS_RATE": {"plane_stress": True, "solid": True, "shell": True},
    "PLAS_TAB_RATE": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_PLAS_TAB_RATE": {"plane_stress": True, "solid": True, "shell": True},
    "MAT_121": {"plane_stress": True, "solid": True, "shell": True},
    37: {"plane_stress": False, "solid": True, "shell": False},
    "37": {"plane_stress": False, "solid": True, "shell": False},
    "LAW37": {"plane_stress": False, "solid": True, "shell": False},
    "BIPHAS": {"plane_stress": False, "solid": True, "shell": False},
    "BIPHASIC": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW37": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_BIPHAS": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_BIPHASIC": {"plane_stress": False, "solid": True, "shell": False},
    24: {"plane_stress": False, "solid": True, "shell": False},
    "24": {"plane_stress": False, "solid": True, "shell": False},
    "LAW24": {"plane_stress": False, "solid": True, "shell": False},
    "CONC": {"plane_stress": False, "solid": True, "shell": False},
    "CONCRETE": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_LAW24": {"plane_stress": False, "solid": True, "shell": False},
    "MAT_CONC": {"plane_stress": False, "solid": True, "shell": False},
}

MATERIAL_SOLID_DISPATCH: dict[Any, Any] = {
    24: law24_concrete.solid_update, "24": law24_concrete.solid_update, "LAW24": law24_concrete.solid_update,
    "CONC": law24_concrete.solid_update, "CONCRETE": law24_concrete.solid_update,
    "MAT_LAW24": law24_concrete.solid_update, "MAT_CONC": law24_concrete.solid_update,
    37: law37_biphas.solid_update, "37": law37_biphas.solid_update, "LAW37": law37_biphas.solid_update,
    "BIPHAS": law37_biphas.solid_update, "BIPHASIC": law37_biphas.solid_update,
    "MAT_LAW37": law37_biphas.solid_update, "MAT_BIPHAS": law37_biphas.solid_update,
    "MAT_BIPHASIC": law37_biphas.solid_update,
    120: law120_solid_update, "120": law120_solid_update, "LAW120": law120_solid_update,
    "TAPO": law120_solid_update, "MAT_LAW120": law120_solid_update, "MAT_TAPO": law120_solid_update,
    "LAW120_TAPO": law120_solid_update, "TAB_PONT_ORTH": law120_solid_update,
    "MAT_TAB_PONT_ORTH": law120_solid_update, "MAT_120": law120_solid_update,
    121: law121_solid_update, "121": law121_solid_update, "LAW121": law121_solid_update,
    "PLAS_RATE": law121_solid_update, "MAT_LAW121": law121_solid_update, "MAT_PLAS_RATE": law121_solid_update,
    "LAW121_PLAS_RATE": law121_solid_update, "PLAS_TAB_RATE": law121_solid_update,
    "MAT_PLAS_TAB_RATE": law121_solid_update, "MAT_121": law121_solid_update,
    117: law117_solid_update, "117": law117_solid_update, "LAW117": law117_solid_update,
    "COH_MC": law117_solid_update, "COH_TAB": law117_solid_update, "COHESIVE": law117_solid_update,
    "COHESIVE_TABULATED": law117_solid_update, "MAT_LAW117": law117_solid_update,
    "MAT_COH_MC": law117_solid_update, "MAT_COH_TAB": law117_solid_update,
    "LAW117_COH_MC": law117_solid_update, "LAW117_COH_TAB": law117_solid_update,
    "MLAW117": law117_solid_update,
    90: law90_solid_update, "90": law90_solid_update, "LAW90": law90_solid_update,
    "HYST_FOAM": law90_solid_update, "MAT_LAW90": law90_solid_update,
    "MAT_HYST_FOAM": law90_solid_update, "TAB_FOAM": law90_solid_update,
    "MAT_TAB_FOAM": law90_solid_update, "LAW90_HYST_FOAM": law90_solid_update,
    124: law124_solid_update, "124": law124_solid_update, "LAW124": law124_solid_update,
    "CDPM2": law124_solid_update, "MAT_LAW124": law124_solid_update,
    "MAT_CDPM2": law124_solid_update, "LAW124_CDPM2": law124_solid_update,
    126: law126_solid_update, "126": law126_solid_update, "LAW126": law126_solid_update,
    "JOHNSON_HOLMQUIST_CONCRETE": law126_solid_update, "MAT_LAW126": law126_solid_update,
    "MAT_JOHNSON_HOLMQUIST_CONCRETE": law126_solid_update, "HJC": law126_solid_update, "MAT_HJC": law126_solid_update,
    169: law169_solid_update, "169": law169_solid_update, "LAW169": law169_solid_update,
    "ARUP_ADHESIVE": law169_solid_update, "MAT_LAW169": law169_solid_update,
    "MAT_ARUP_ADHESIVE": law169_solid_update, "ARUP": law169_solid_update, "ADHESIVE": law169_solid_update,
    48: solid_update_law48, "48": solid_update_law48, "LAW48": solid_update_law48,
    "ZHAO": solid_update_law48, "MAT_LAW48": solid_update_law48, "MAT_ZHAO": solid_update_law48,
    "PLAS_ZHAO": solid_update_law48, "LAW48_ZHAO": solid_update_law48,
    52: solid_update_law52, "52": solid_update_law52, "LAW52": solid_update_law52,
    "GURSON": solid_update_law52, "PLAS_GURS": solid_update_law52,
    "MAT_LAW52": solid_update_law52, "MAT_GURSON": solid_update_law52,
    "MAT_PLAS_GURS": solid_update_law52,
    58: solid_update_law58, "58": solid_update_law58, "LAW58": solid_update_law58,
    "FABR_A": solid_update_law58, "FABRIC_A": solid_update_law58,
    "MAT_LAW58": solid_update_law58, "MAT_FABR_A": solid_update_law58,
    "MAT_FABRIC_A": solid_update_law58, "LAW58_FABR_A": solid_update_law58,
    57: solid_update_law57, "57": solid_update_law57, "LAW57": solid_update_law57,
    "BARLAT": solid_update_law57, "BARLAT3": solid_update_law57,
    "MAT_LAW57": solid_update_law57, "MAT_BARLAT": solid_update_law57,
    "MAT_BARLAT3": solid_update_law57, "LAW57_BARLAT": solid_update_law57,
    "LAW57_BARLAT3": solid_update_law57,
    21: solid_update_law21, "21": solid_update_law21, "LAW21": solid_update_law21,
    "DPRAG": solid_update_law21, "MAT_LAW21": solid_update_law21, "MAT_DPRAG": solid_update_law21,
    "LAW21_DPRAG": solid_update_law21,
    49: solid_update_law49, "49": solid_update_law49, "LAW49": solid_update_law49,
    "STEINB": solid_update_law49, "STEINBERG": solid_update_law49,
    "STEINBERG_GUINAN": solid_update_law49, "MAT_LAW49": solid_update_law49,
    "MAT_STEINB": solid_update_law49, "MAT_STEINBERG": solid_update_law49,
    "LAW49_STEINB": solid_update_law49,
    79: solid_update_law79, "79": solid_update_law79, "LAW79": solid_update_law79,
    "JOHN_HOLM": solid_update_law79, "JOHNSON_HOLMQUIST": solid_update_law79,
    "JH2": solid_update_law79, "MAT_LAW79": solid_update_law79,
    "MAT_JOHN_HOLM": solid_update_law79, "LAW79_JOHN_HOLM": solid_update_law79,
    50: solid_update_law50, "50": solid_update_law50, "LAW50": solid_update_law50,
    "VISC_HONEY": solid_update_law50, "HYP_FOAM": solid_update_law50,
    "MAT_LAW50": solid_update_law50, "MAT_VISC_HONEY": solid_update_law50,
    "MAT_HYP_FOAM": solid_update_law50,
    163: solid_update_law163, "163": solid_update_law163, "LAW163": solid_update_law163,
    "CRUSHABLE_FOAM": solid_update_law163, "CRUSH_FOAM": solid_update_law163,
    "MAT_LAW163": solid_update_law163, "MAT_CRUSHABLE_FOAM": solid_update_law163,
    "MAT_CRUSH_FOAM": solid_update_law163, "LAW163_CRUSHABLE_FOAM": solid_update_law163,
    73: solid_update_law73, "73": solid_update_law73, "LAW73": solid_update_law73,
    "HILL_THERM": solid_update_law73, "THERM_HILL": solid_update_law73,
    "MAT_LAW73": solid_update_law73,
    "MAT_HILL_THERM": solid_update_law73, "MAT_THERM_HILL": solid_update_law73,
    "LAW73_HILL_THERM": solid_update_law73, "LAW73_THERM_HILL": solid_update_law73,
    71: law71_solid_update, "71": law71_solid_update, "LAW71": law71_solid_update,
    "SUPER_ELAS": law71_solid_update, "NITINOL": law71_solid_update,
    "MAT_71": law71_solid_update, "MAT_LAW71": law71_solid_update,
    "MAT_SUPER_ELAS": law71_solid_update, "MAT_NITINOL": law71_solid_update,
    "LAW71_SUPER_ELAS": law71_solid_update, "LAW71_NITINOL": law71_solid_update,
    76: law76_solid_update, "76": law76_solid_update, "LAW76": law76_solid_update,
    "SAMP": law76_solid_update, "SAMP-1": law76_solid_update, "SAMP_1": law76_solid_update,
    "MAT_76": law76_solid_update, "MAT_LAW76": law76_solid_update,
    "MAT_SAMP": law76_solid_update, "MAT_SAMP-1": law76_solid_update, "MAT_SAMP_1": law76_solid_update,
    "LAW76_SAMP": law76_solid_update, "LAW76_SAMP-1": law76_solid_update, "LAW76_SAMP_1": law76_solid_update,
    66: solid_update_law66, "66": solid_update_law66, "LAW66": solid_update_law66,
    "PLAS_TAB_COSSER": solid_update_law66, "PLAS_COSSER": solid_update_law66,
    "FOAM_TAB": solid_update_law66, "MAT_LAW66": solid_update_law66,
    "MAT_PLAS_TAB_COSSER": solid_update_law66, "MAT_PLAS_COSSER": solid_update_law66,
    "MAT_FOAM_TAB": solid_update_law66,
    74: solid_update_law74, "74": solid_update_law74, "LAW74": solid_update_law74,
    "HILL_3D": solid_update_law74, "ORTH_PLAS": solid_update_law74,
    "MAT_LAW74": solid_update_law74, "MAT_HILL_3D": solid_update_law74,
    "MAT_ORTH_PLAS": solid_update_law74, "LAW74_HILL_3D": solid_update_law74,
    87: solid_update_law87, "87": solid_update_law87, "LAW87": solid_update_law87,
    "BARLAT2000": solid_update_law87, "BARLAT_2000": solid_update_law87, "BARLAT2000_2D": solid_update_law87,
    "BARLAT_YLD2000": solid_update_law87, "MAT_LAW87": solid_update_law87,
    "MAT_BARLAT2000": solid_update_law87, "MAT_BARLAT_2000": solid_update_law87, "MAT_BARLAT2000_2D": solid_update_law87,
    "MAT_BARLAT_YLD2000": solid_update_law87, "LAW87_BARLAT2000": solid_update_law87,
    88: solid_update_law88, "88": solid_update_law88, "LAW88": solid_update_law88,
    "HYP_TAB": solid_update_law88, "TAB_HYP": solid_update_law88,
    "HYPER_ELAS": solid_update_law88, "TABULATED_HYPERELASTIC": solid_update_law88,
    "TABULATED_HYP": solid_update_law88,
    "MAT_LAW88": solid_update_law88, "MAT_HYP_TAB": solid_update_law88,
    "MAT_TAB_HYP": solid_update_law88, "MAT_HYPER_ELAS": solid_update_law88,
    "MAT_TABULATED_HYPERELASTIC": solid_update_law88,
    "LAW88_TAB_HYP": solid_update_law88,
    92: law92_solid_update, "92": law92_solid_update, "LAW92": law92_solid_update,
    "ARRUDA_BOYCE": law92_solid_update, "ARRUDA-BOYCE": law92_solid_update,
    "MAT_LAW92": law92_solid_update, "MAT_ARRUDA_BOYCE": law92_solid_update,
    "LAW92_ARRUDA_BOYCE": law92_solid_update,
    93: law93_solid_update, "93": law93_solid_update, "LAW93": law93_solid_update,
    "ORTH_HILL": law93_solid_update, "MAT_LAW93": law93_solid_update,
    "MAT_ORTH_HILL": law93_solid_update, "LAW93_ORTH_HILL": law93_solid_update,
    94: law94_solid_update, "94": law94_solid_update, "LAW94": law94_solid_update,
    "YEOH": law94_solid_update, "MAT_LAW94": law94_solid_update,
    "MAT_YEOH": law94_solid_update, "LAW94_YEOH": law94_solid_update,
    95: law95_solid_update, "95": law95_solid_update, "LAW95": law95_solid_update,
    "BERGSTROM_BOYCE": law95_solid_update, "BERGSTROM-BOYCE": law95_solid_update,
    "MAT_LAW95": law95_solid_update, "MAT_BERGSTROM_BOYCE": law95_solid_update,
    "LAW95_BERGSTROM_BOYCE": law95_solid_update,
    100: law100_solid_update, "100": law100_solid_update, "LAW100": law100_solid_update,
    "VISC_HYP": law100_solid_update, "MNF": law100_solid_update, "MAT_LAW100": law100_solid_update,
    "MAT_VISC_HYP": law100_solid_update, "MAT_MNF": law100_solid_update, "LAW100_VISC_HYP": law100_solid_update,
    "LAW100_MNF": law100_solid_update, "SPOTWELD": law100_solid_update, "STRUCTURAL_ADHESIVE": law100_solid_update,
    101: law101_solid_update, "101": law101_solid_update, "LAW101": law101_solid_update,
    "PP": law101_solid_update, "MAT_PP": law101_solid_update, "PLAS_POLY": law101_solid_update,
    "MAT_PLAS_POLY": law101_solid_update, "MAT_LAW101": law101_solid_update, "LAW101_PP": law101_solid_update,
    "LAW101_PLAS_POLY": law101_solid_update,
    102: law102_solid_update, "102": law102_solid_update, "LAW102": law102_solid_update,
    "DPRAG2": law102_solid_update, "MAT_DPRAG2": law102_solid_update, "MAT_LAW102": law102_solid_update,
    "LAW102_DPRAG2": law102_solid_update, "DRUCKER_PRAGER_2": law102_solid_update,
    103: law103_solid_update, "103": law103_solid_update, "LAW103": law103_solid_update,
    "HENSEL_SPITTEL": law103_solid_update, "HENSEL-SPITTEL": law103_solid_update, "PLAS_HENS": law103_solid_update,
    "MAT_HENSEL_SPITTEL": law103_solid_update, "MAT_PLAS_HENS": law103_solid_update, "MAT_103": law103_solid_update,
    "MAT_LAW103": law103_solid_update, "HEN": law103_solid_update,
    104: law104_solid_update, "104": law104_solid_update, "LAW104": law104_solid_update,
    "DRUCKER": law104_solid_update, "JOHNS_VOCE_DRUCKER": law104_solid_update, "JOHNS-VOCE-DRUCKER": law104_solid_update,
    "PLAS_DRUCK": law104_solid_update, "MAT_DRUCKER": law104_solid_update, "MAT_JOHNS_VOCE_DRUCKER": law104_solid_update,
    "MAT_PLAS_DRUCK": law104_solid_update, "MAT_104": law104_solid_update, "MAT_LAW104": law104_solid_update,
    "LAW104_DRUCKER": law104_solid_update,
    105: law105_solid_update, "105": law105_solid_update, "LAW105": law105_solid_update,
    "POWDER_BURN": law105_solid_update, "POWDERBURN": law105_solid_update, "MAT_POWDER_BURN": law105_solid_update,
    "MAT_POWDERBURN": law105_solid_update, "MAT_105": law105_solid_update, "MAT_LAW105": law105_solid_update,
    "LAW105_POWDER_BURN": law105_solid_update,
    106: law106_solid_update, "106": law106_solid_update, "LAW106": law106_solid_update,
    "JCOOK_ALM": law106_solid_update, "JOHNS_COOK_ALM": law106_solid_update, "MAT_JCOOK_ALM": law106_solid_update,
    "MAT_106": law106_solid_update, "MAT_LAW106": law106_solid_update, "LAW106_JCOOK_ALM": law106_solid_update,
    107: law107_solid_update, "107": law107_solid_update, "LAW107": law107_solid_update,
    "PAPER_LIGHT": law107_solid_update, "PLAS_PAPER_LIGHT": law107_solid_update, "LAW107_PAPER_LIGHT": law107_solid_update,
    "PFEIFFER": law107_solid_update, "MAT_PFEIFFER": law107_solid_update, "MAT_PAPER_LIGHT": law107_solid_update,
    "MAT_PLAS_PAPER_LIGHT": law107_solid_update, "MAT_107": law107_solid_update, "MAT_LAW107": law107_solid_update,
    109: law109_solid_update, "109": law109_solid_update, "LAW109": law109_solid_update,
    "TAB_PLAS": law109_solid_update, "MAT_LAW109": law109_solid_update, "MAT_TAB_PLAS": law109_solid_update,
}

MATERIAL_SHELL_DISPATCH: dict[Any, Any] = {
    120: law120_shell_update, "120": law120_shell_update, "LAW120": law120_shell_update,
    "TAPO": law120_shell_update, "MAT_LAW120": law120_shell_update, "MAT_TAPO": law120_shell_update,
    "LAW120_TAPO": law120_shell_update, "TAB_PONT_ORTH": law120_shell_update,
    "MAT_TAB_PONT_ORTH": law120_shell_update, "MAT_120": law120_shell_update,
    121: law121_shell_update, "121": law121_shell_update, "LAW121": law121_shell_update,
    "PLAS_RATE": law121_shell_update, "MAT_LAW121": law121_shell_update, "MAT_PLAS_RATE": law121_shell_update,
    "LAW121_PLAS_RATE": law121_shell_update, "PLAS_TAB_RATE": law121_shell_update,
    "MAT_PLAS_TAB_RATE": law121_shell_update, "MAT_121": law121_shell_update,
    117: law117_shell_update, "117": law117_shell_update, "LAW117": law117_shell_update,
    "COH_MC": law117_shell_update, "COH_TAB": law117_shell_update, "COHESIVE": law117_shell_update,
    "COHESIVE_TABULATED": law117_shell_update, "MAT_LAW117": law117_shell_update,
    "MAT_COH_MC": law117_shell_update, "MAT_COH_TAB": law117_shell_update,
    "LAW117_COH_MC": law117_shell_update, "LAW117_COH_TAB": law117_shell_update,
    "MLAW117": law117_shell_update,
    169: law169_shell_update, "169": law169_shell_update, "LAW169": law169_shell_update,
    "ARUP_ADHESIVE": law169_shell_update, "MAT_LAW169": law169_shell_update,
    "MAT_ARUP_ADHESIVE": law169_shell_update, "ARUP": law169_shell_update, "ADHESIVE": law169_shell_update,
    48: shell_update_law48, "48": shell_update_law48, "LAW48": shell_update_law48,
    "ZHAO": shell_update_law48, "MAT_LAW48": shell_update_law48, "MAT_ZHAO": shell_update_law48,
    "PLAS_ZHAO": shell_update_law48, "LAW48_ZHAO": shell_update_law48,
    52: shell_update_law52, "52": shell_update_law52, "LAW52": shell_update_law52,
    "GURSON": shell_update_law52, "PLAS_GURS": shell_update_law52,
    "MAT_LAW52": shell_update_law52, "MAT_GURSON": shell_update_law52,
    "MAT_PLAS_GURS": shell_update_law52,
    58: shell_update_law58, "58": shell_update_law58, "LAW58": shell_update_law58,
    "FABR_A": shell_update_law58, "FABRIC_A": shell_update_law58,
    "MAT_LAW58": shell_update_law58, "MAT_FABR_A": shell_update_law58,
    "MAT_FABRIC_A": shell_update_law58, "LAW58_FABR_A": shell_update_law58,
    57: shell_update_law57, "57": shell_update_law57, "LAW57": shell_update_law57,
    "BARLAT": shell_update_law57, "BARLAT3": shell_update_law57,
    "MAT_LAW57": shell_update_law57, "MAT_BARLAT": shell_update_law57,
    "MAT_BARLAT3": shell_update_law57, "LAW57_BARLAT": shell_update_law57,
    "LAW57_BARLAT3": shell_update_law57,
    21: shell_update_law21, "21": shell_update_law21, "LAW21": shell_update_law21,
    "DPRAG": shell_update_law21, "MAT_LAW21": shell_update_law21, "MAT_DPRAG": shell_update_law21,
    "LAW21_DPRAG": shell_update_law21,
    49: shell_update_law49, "49": shell_update_law49, "LAW49": shell_update_law49,
    "STEINB": shell_update_law49, "STEINBERG": shell_update_law49,
    "STEINBERG_GUINAN": shell_update_law49, "MAT_LAW49": shell_update_law49,
    "MAT_STEINB": shell_update_law49, "MAT_STEINBERG": shell_update_law49,
    "LAW49_STEINB": shell_update_law49,
    79: shell_update_law79, "79": shell_update_law79, "LAW79": shell_update_law79,
    "JOHN_HOLM": shell_update_law79, "JOHNSON_HOLMQUIST": shell_update_law79,
    "JH2": shell_update_law79, "MAT_LAW79": shell_update_law79,
    "MAT_JOHN_HOLM": shell_update_law79, "LAW79_JOHN_HOLM": shell_update_law79,
    50: shell_update_law50, "50": shell_update_law50, "LAW50": shell_update_law50,
    "VISC_HONEY": shell_update_law50, "HYP_FOAM": shell_update_law50,
    "MAT_LAW50": shell_update_law50, "MAT_VISC_HONEY": shell_update_law50,
    "MAT_HYP_FOAM": shell_update_law50,
    163: shell_update_law163, "163": shell_update_law163, "LAW163": shell_update_law163,
    "CRUSHABLE_FOAM": shell_update_law163, "CRUSH_FOAM": shell_update_law163,
    "MAT_LAW163": shell_update_law163, "MAT_CRUSHABLE_FOAM": shell_update_law163,
    "MAT_CRUSH_FOAM": shell_update_law163, "LAW163_CRUSHABLE_FOAM": shell_update_law163,
    73: shell_update_law73, "73": shell_update_law73, "LAW73": shell_update_law73,
    "HILL_THERM": shell_update_law73, "THERM_HILL": shell_update_law73,
    "MAT_LAW73": shell_update_law73,
    "MAT_HILL_THERM": shell_update_law73, "MAT_THERM_HILL": shell_update_law73,
    "LAW73_HILL_THERM": shell_update_law73, "LAW73_THERM_HILL": shell_update_law73,
    71: law71_shell_update, "71": law71_shell_update, "LAW71": law71_shell_update,
    "SUPER_ELAS": law71_shell_update, "NITINOL": law71_shell_update,
    "MAT_71": law71_shell_update, "MAT_LAW71": law71_shell_update,
    "MAT_SUPER_ELAS": law71_shell_update, "MAT_NITINOL": law71_shell_update,
    "LAW71_SUPER_ELAS": law71_shell_update, "LAW71_NITINOL": law71_shell_update,
    76: law76_shell_update, "76": law76_shell_update, "LAW76": law76_shell_update,
    "SAMP": law76_shell_update, "SAMP-1": law76_shell_update, "SAMP_1": law76_shell_update,
    "MAT_76": law76_shell_update, "MAT_LAW76": law76_shell_update,
    "MAT_SAMP": law76_shell_update, "MAT_SAMP-1": law76_shell_update, "MAT_SAMP_1": law76_shell_update,
    "LAW76_SAMP": law76_shell_update, "LAW76_SAMP-1": law76_shell_update, "LAW76_SAMP_1": law76_shell_update,
    66: shell_update_law66, "66": shell_update_law66, "LAW66": shell_update_law66,
    "PLAS_TAB_COSSER": shell_update_law66, "PLAS_COSSER": shell_update_law66,
    "FOAM_TAB": shell_update_law66, "MAT_LAW66": shell_update_law66,
    "MAT_PLAS_TAB_COSSER": shell_update_law66, "MAT_PLAS_COSSER": shell_update_law66,
    "MAT_FOAM_TAB": shell_update_law66,
    74: shell_update_law74, "74": shell_update_law74, "LAW74": shell_update_law74,
    "HILL_3D": shell_update_law74, "ORTH_PLAS": shell_update_law74,
    "MAT_LAW74": shell_update_law74, "MAT_HILL_3D": shell_update_law74,
    "MAT_ORTH_PLAS": shell_update_law74, "LAW74_HILL_3D": shell_update_law74,
    87: shell_update_law87, "87": shell_update_law87, "LAW87": shell_update_law87,
    "BARLAT2000": shell_update_law87, "BARLAT_2000": shell_update_law87, "BARLAT2000_2D": shell_update_law87,
    "BARLAT_YLD2000": shell_update_law87, "MAT_LAW87": shell_update_law87,
    "MAT_BARLAT2000": shell_update_law87, "MAT_BARLAT_2000": shell_update_law87, "MAT_BARLAT2000_2D": shell_update_law87,
    "MAT_BARLAT_YLD2000": shell_update_law87, "LAW87_BARLAT2000": shell_update_law87,
    88: shell_update_law88, "88": shell_update_law88, "LAW88": shell_update_law88,
    "HYP_TAB": shell_update_law88, "TAB_HYP": shell_update_law88,
    "HYPER_ELAS": shell_update_law88, "TABULATED_HYPERELASTIC": shell_update_law88,
    "TABULATED_HYP": shell_update_law88,
    "MAT_LAW88": shell_update_law88, "MAT_HYP_TAB": shell_update_law88,
    "MAT_TAB_HYP": shell_update_law88, "MAT_HYPER_ELAS": shell_update_law88,
    "MAT_TABULATED_HYPERELASTIC": shell_update_law88,
    "LAW88_TAB_HYP": shell_update_law88,
    92: law92_shell_update, "92": law92_shell_update, "LAW92": law92_shell_update,
    "ARRUDA_BOYCE": law92_shell_update, "ARRUDA-BOYCE": law92_shell_update,
    "MAT_LAW92": law92_shell_update, "MAT_ARRUDA_BOYCE": law92_shell_update,
    "LAW92_ARRUDA_BOYCE": law92_shell_update,
    93: law93_shell_update, "93": law93_shell_update, "LAW93": law93_shell_update,
    "ORTH_HILL": law93_shell_update, "MAT_LAW93": law93_shell_update,
    "MAT_ORTH_HILL": law93_shell_update, "LAW93_ORTH_HILL": law93_shell_update,
    94: law94_shell_update, "94": law94_shell_update, "LAW94": law94_shell_update,
    "YEOH": law94_shell_update, "MAT_LAW94": law94_shell_update,
    "MAT_YEOH": law94_shell_update, "LAW94_YEOH": law94_shell_update,
    104: law104_shell_update, "104": law104_shell_update, "LAW104": law104_shell_update,
    "DRUCKER": law104_shell_update, "JOHNS_VOCE_DRUCKER": law104_shell_update, "JOHNS-VOCE-DRUCKER": law104_shell_update,
    "PLAS_DRUCK": law104_shell_update, "MAT_DRUCKER": law104_shell_update, "MAT_JOHNS_VOCE_DRUCKER": law104_shell_update,
    "MAT_PLAS_DRUCK": law104_shell_update, "MAT_104": law104_shell_update, "MAT_LAW104": law104_shell_update,
    "LAW104_DRUCKER": law104_shell_update,
    106: law106_shell_update, "106": law106_shell_update, "LAW106": law106_shell_update,
    "JCOOK_ALM": law106_shell_update, "JOHNS_COOK_ALM": law106_shell_update, "MAT_JCOOK_ALM": law106_shell_update,
    "MAT_106": law106_shell_update, "MAT_LAW106": law106_shell_update, "LAW106_JCOOK_ALM": law106_shell_update,
    107: law107_shell_update, "107": law107_shell_update, "LAW107": law107_shell_update,
    "PAPER_LIGHT": law107_shell_update, "PLAS_PAPER_LIGHT": law107_shell_update, "LAW107_PAPER_LIGHT": law107_shell_update,
    "PFEIFFER": law107_shell_update, "MAT_PFEIFFER": law107_shell_update, "MAT_PAPER_LIGHT": law107_shell_update,
    "MAT_PLAS_PAPER_LIGHT": law107_shell_update, "MAT_107": law107_shell_update, "MAT_LAW107": law107_shell_update,
    109: law109_shell_update, "109": law109_shell_update, "LAW109": law109_shell_update,
    "TAB_PLAS": law109_shell_update, "MAT_LAW109": law109_shell_update, "MAT_TAB_PLAS": law109_shell_update,
    110: law110_shell_update, "110": law110_shell_update, "LAW110": law110_shell_update,
    "VEGTER": law110_shell_update, "PLAS_VEGTER": law110_shell_update,
    "LAW110_VEGTER": law110_shell_update, "MLAW110": law110_shell_update,
    "MAT_LAW110": law110_shell_update, "MAT_VEGTER": law110_shell_update,
    "MAT_PLAS_VEGTER": law110_shell_update, "MAT_110": law110_shell_update,
}



def register_materials():
    _get_law05()
    _get_law10()
    _get_law25()
    _get_law32()
    _get_law38()
    _get_law43()
    _get_law60()
    for mod in (eos, law01_elastic, law02_johnson_cook, law03_plas_bost,
                law04_hyd_jcook, law06_hyd_visc, law10_soil, law12_comp3d, law14_compso, law15_chang, law19_fabric, law22_dama, law24_concrete,
                law25_composite, law27_brittle, law28_honeycomb, law33_foamplas, law34_boltzmann, law35_kelvinmax,
                law36_tabulated, law37_biphas, law40_kelvinmax, law42_ogden, law44_cowper,
                law52_gurson,
                law60_plast3,
                law62_hypervisco, law70_tabfoam, law81_druckerprager,
                law83_spotweld, law114_seatbelt, law120_advanced,
                law58_fabr_a,
                law57_barlat,
                mat_gas, mat_void):
        fn = getattr(mod, "_register", None)
        if callable(fn):
            fn()
    _register_law12()
    _register_law14()
    _register_law15()
    _register_law22()
    if law05_jwl is not None:
        fn = getattr(law05_jwl, "_register", None)
        if callable(fn):
            fn()
    _register_law05()
    if law10_soil is not None:
        fn = getattr(law10_soil, "_register", None)
        if callable(fn):
            fn()
    _register_law10()
    if law28_honeycomb is not None:
        fn = getattr(law28_honeycomb, "_register", None)
        if callable(fn):
            fn()
    _register_law28()
    if law34_boltzmann is not None:
        fn = getattr(law34_boltzmann, "_register", None)
        if callable(fn):
            fn()
    _register_law34()
    if law37_biphas is not None:
        fn = getattr(law37_biphas, "_register", None)
        if callable(fn):
            fn()
    _register_law37()
    if law38_visc_tab is not None:
        fn = getattr(law38_visc_tab, "_register", None)
        if callable(fn):
            fn()
    _register_law38()
    if law32_hill is not None:
        fn = getattr(law32_hill, "_register", None)
        if callable(fn):
            fn()
    _register_law32()
    if law25_composite is not None:
        fn = getattr(law25_composite, "_register", None)
        if callable(fn):
            fn()
    _register_law25()
    if law43_hill_tab is not None:
        fn = getattr(law43_hill_tab, "_register", None)
        if callable(fn):
            fn()
    if law69_hyperelastic is not None:
        fn = getattr(law69_hyperelastic, "_register", None)
        if callable(fn):
            fn()
    _register_law69()
    if law82_ogden is not None:
        fn = getattr(law82_ogden, "_register", None)
        if callable(fn):
            fn()
    _register_law82()
    if law60_plast3 is not None:
        fn = getattr(law60_plast3, "_register", None)
        if callable(fn):
            fn()
    _register_law60()
    _register_law48()
    _register_law58()
    _register_law52()
    _register_law57()
    _register_law21()
    _register_law49()
    _register_law79()
    _register_law50()
    _register_law163()
    _register_law71()
    _register_law73()
    _register_law66()
    _register_law74()
    _register_law87()
    _register_law88()
    _register_law92()
    _register_law93()
    _register_law94()
    _register_law95()
    _register_law100()
    _register_law101()
    _register_law102()
    _register_law103()
    _register_law104()
    _register_law105()
    _register_law106()
    _register_law107()
    _register_law109()
    _register_law110()
    if hasattr(law126_hjc, "_register"):
        law126_hjc._register()
    if hasattr(law169_arup, "_register"):
        law169_arup._register()
    if hasattr(law117_cohesive, "_register"):
        law117_cohesive._register()
    _register_law117()
    if hasattr(law123_daimler, "_register"):
        law123_daimler._register()
    _register_law123()
    _register_law120()
    _register_law121()


def extra_shapes(mat, nip=None):
    """Per-element persistent state a law needs beyond (sig, epsp).

    Returns {name: trailing_shape}; the kernels allocate arrays of shape
    (n, *trailing_shape) for solids and (n, nip, *trailing_shape[1:])...
    — in practice the shapes below already include the layer dimension
    for shell laws (nip is the layer count of the property; ``nip=None``
    for solids, whose per-point state is per-element)."""
    shapes = {}
    if mat.law == 22 or getattr(mat, "law_name", None) in ("22", "LAW22", "DAMA", "PLAS_DAMA"):
        shapes.update(law22_dama.extra_shapes(mat, nip=nip))
    if mat.law == 27:
        shapes.update(eps27=(nip, 3), crk27=(nip,), ang27=(nip,),
                      dmg27=(nip, 2))
    if mat.law == 36:
        if mat.params.get("c_hard", 0.0) > 0.0:
            shapes.update(sigb36=(nip, 3) if nip else (6,))
        if mat.params.get("f_cut", 0.0) > 0.0:
            shapes.update(epsd36=(nip,) if nip else ())
    if mat.law == 19:
        # M37 pack 2: total strain, zerostress reference stress SIGI and
        # the law's own time accumulator (see law19_fabric docstring)
        shapes.update(eps19=(nip, 3), sigi19=(nip, 3), t19=(nip,))
    if getattr(mat, "law", None) in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A"):
        shapes.update(eps58=(nip, 3) if nip else (3,), yc=(nip,) if nip else (), yt=(nip,) if nip else (), fn=(nip,) if nip else (),
                      sigv_xy=(nip,) if nip else (), tan_phi=(nip,) if nip else (), sigi58=(nip, 3) if nip else (3,), t58=(nip,) if nip else ())
    if mat.law == 24:
        # M37 pack 2: the LBUF state of conc24.F (law24_concrete docstring)
        # Includes steel reinforcement state (siga24, epsa24)
        shapes.update(strain24=(6,), sigc24=(6,), crak24=(3,),
                      dam24=(3,), ang24=(6,), epsf24=(3,),
                      siga24=(3,), epsa24=(3,),
                      vk024=(), vk24=(), rob24=(), off24=(), ini24=())
    if getattr(mat, "law", None) in (81, "81", "LAW81", "DPRAG_CAP", "MAT_LAW81", "MAT_DPRAG_CAP") or getattr(mat, "law_name", None) in (81, "81", "LAW81", "DPRAG_CAP", "MAT_LAW81", "MAT_DPRAG_CAP"):
        # M37 pack 2: the defp(nel, 2) plastic strains of sigeps81.F90
        shapes.update(epspd81=(), epspv81=())
    if mat.law == 62 and mat.params.get("NPRONY", 0) > 0:
        # M37 pack 2: Prony history of sigeps62.F (UVAR 1:6 = previous
        # global deviatoric PK2, then 6 per Prony term)
        shapes.update(sdg62=(6,), h62=(mat.params["NPRONY"], 6))
    if mat.law == 70:
        # M37 pack 1: total strain, the 10 UVARs of sigeps70.F and the
        # filtered strain rate (law70_tabfoam docstring; solids only)
        shapes.update(eps70=(6,), uv70=(10,), epsd70=())
    if mat.law == 35:
        # M37 pack 1: total strain, closed-cell air pressure (UVAR1) and
        # the filtered strain rate (UVAR4) of sigeps35.F (solids only)
        shapes.update(eps35=(6,), sigair35=(), edot35=())
    if mat.law == 40:
        # M37 pack 1: total strain + the 40 UVARs of sigeps40.F (Stassi
        # criteria, EDRV rate memory, 5x6 Prony branch stresses)
        shapes.update(eps40=(6,), uv40=(40,))
    if mat.law == 44:
        # M37 pack 1 (law44_cowper): the filtered strain-rate state is
        # only needed when filtering / VP=1 is active; the total strain
        # only when the tension softening (eps_t1) is finite
        if mat.params.get("ismooth", 0) or mat.params.get("vflag") == 1:
            shapes["epsd44"] = (nip,) if nip is not None else ()
        if mat.params.get("epsr1", 1e30) < 1e30:
            shapes["eps44"] = (nip, 3) if nip is not None else (6,)
    if mat.law == 2 and "mT" in mat.params:
        # adiabatic temperature RISE above T_i (M6 thermal terms)
        shapes["temp"] = (nip,) if nip is not None else ()
    if getattr(mat, "law", None) == 83 or type(mat).__name__ == "Law83":
        shapes.update(epsp=(), asrate=())
    if mat.law == 33 and abs(mat.params.get("KEN", 0)) == 1:
        # M533: ICASE=2 Kelvin model needs total strain (solids only)
        shapes.update(eps33=(6,))
    if mat.law == 28 or getattr(mat, "law_name", None) in ("LAW28", "HONEYCOMB", "HONEYCOMB_SOL"):
        # M538: LAW28 (honeycomb) needs total strain and element deletion flag (solids only)
        shapes.update(eps28=(6,), off28=())
    if mat.law == 34 or getattr(mat, "law_name", None) in ("LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"):
        # M539: LAW34 (Boltzmann viscoelastic) needs total strain and history variables (7 for shells, sigeps34c.F)
        shapes.update(eps34=(nip, 3) if nip else (6,), uv34=(nip, 7) if nip else (6,))
        if nip:
            shapes.update(ezz34=(nip,))
    if mat.law in (37, "37", "LAW37", "BIPHAS", "BIPHASIC") or getattr(mat, "law_name", None) in ("LAW37", "BIPHAS", "BIPHASIC"):
        # M540: LAW37 (Biphasic fluid-gas) needs uv37 (5 variables per element: liquid mass density, rho2, rho1, alpha_v1, alpha_v2)
        shapes.update(uv37=(5,))
    if mat.law in (38, "38", "LAW38", "VISC_TAB") or getattr(mat, "law_name", None) in ("LAW38", "VISC_TAB"):
        # M541: LAW38 (VISC_TAB tabulated viscoelastic foam) needs 33 state variables (uv38)
        shapes.update(eps38=(6,), uv38=_STATE_VAR_COUNT.get("uv38", (33,)), off38=())
    if getattr(mat, "law", None) in (32, "32", "LAW32", "HILL") or getattr(mat, "law_name", None) in ("32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL"):
        # M542: LAW32 (Hill orthotropic plasticity) needs 2 state variables per point (uv32) + off32 flag
        uv_shape = _STATE_VAR_COUNT.get("uv32", (2,))
        shapes.update(uv32=(nip, *uv_shape) if nip else uv_shape,
                      off32=(nip,) if nip else ())
    if mat.law == 4:
        shapes["temp"] = ()
    if getattr(mat, "law", None) in (5, "5", "LAW5", "JWL") or getattr(mat, "law_name", None) in ("LAW5", "JWL"):
        shapes.update(bfrac=(), aburn=(), eint=(), tb=())
    if getattr(mat, "law", None) in (10, "10", "LAW10", "SOIL", "DPRAG1") or (getattr(mat, "law_name", None) in ("LAW10", "SOIL", "DPRAG1") and getattr(mat, "law", None) not in (21, "21", "LAW21", "DPRAG")):
        shapes.update(mu_bak=(), epxe=(), p_old=())
    if getattr(mat, "law", None) in (21, "21", "LAW21", "DPRAG", "MAT_LAW21", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB", "MAT_DUCKHUB") or (getattr(mat, "law_name", None) in ("21", "LAW21", "DPRAG", "MAT_LAW21", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB", "MAT_DUCKHUB") and getattr(mat, "law", None) != 10):
        if nip is not None:
            shapes.update(mu_bak=(nip,), epxe=(nip,), p=(nip,), defp=(nip,), p_old=(nip,), mu=(nip,), c_solid=(nip,), g0=(nip,))
        else:
            shapes.update(mu_bak=(), epxe=(), p=(), defp=(), p_old=(), mu=(), c_solid=(), g0=())
    if getattr(mat, "law", None) in (12, "12", "LAW12", "3D_COMP", "COMP_3D") or getattr(mat, "law_name", None) in ("12", "LAW12", "3D_COMP", "COMP_3D"):
        shapes.update(law12_comp3d.extra_shapes(mat, nip))
    if getattr(mat, "law", None) in (14, "14", "LAW14", "COMPSO", "COMP_SOL") or getattr(mat, "law_name", None) in ("14", "LAW14", "COMPSO", "COMP_SOL"):
        shapes.update(law14_compso.extra_shapes(mat, nip))
    if getattr(mat, "law", None) in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG") or getattr(mat, "law_name", None) in ("15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
        shapes.update(law15_chang.extra_shapes(mat, nip))
    if getattr(mat, "law", None) in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS") or getattr(mat, "law_name", None) in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
        shapes.update(law25_composite.extra_shapes(mat, nip))
    if getattr(mat, "law", None) in (43, "43", "LAW43", "HILL_TAB", "LAW43_HILL_TAB") or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB"):
        # M548: LAW43 (Hill tabulated orthotropic plasticity)
        _get_law43()
        if law43_hill_tab is not None and hasattr(law43_hill_tab, "extra_shapes"):
            shapes.update(law43_hill_tab.extra_shapes(mat, nip))
        else:
            uv_shape = _STATE_VAR_COUNT.get("uv43", (4,))
            shapes.update(uv43=(nip, *uv_shape) if nip else uv_shape,
                          off43=(nip,) if nip else ())
    if getattr(mat, "law", None) in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC") or getattr(mat, "law_name", None) in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC"):
        shapes.update(law69_hyperelastic.extra_shapes(mat, nip))
    if getattr(mat, "law", None) in (82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN") or getattr(mat, "law_name", None) in ("82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN"):
        shapes.update(law82_ogden.extra_shapes(mat, nip))
    if getattr(mat, "law", None) in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC") or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
        _get_law60()
        if law60_plast3 is not None and hasattr(law60_plast3, "extra_shapes"):
            shapes.update(law60_plast3.extra_shapes(mat, nip))
        else:
            nfunc = 5
            if hasattr(mat, "params") and isinstance(mat.params, dict):
                nfunc = int(mat.params.get("nfunc", mat.params.get("NFUNC", 5)))
            elif hasattr(mat, "nfunc"):
                nfunc = int(mat.nfunc)
            uvar_dim = (5 + nfunc,)
            shapes.update(uvar=(nip, *uvar_dim) if nip else uvar_dim,
                          off60=(nip,) if nip else ())
    if getattr(mat, "law", None) in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_LAW48", "MAT_ZHAO", "MAT_PLAS_ZHAO", "LAW48_ZHAO") or getattr(mat, "law_name", None) in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_LAW48", "MAT_ZHAO", "MAT_PLAS_ZHAO", "LAW48_ZHAO"):
        shapes.update(law48_zhao.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS"):
        shapes.update(law52_gurson.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
        if law57_barlat is not None and hasattr(law57_barlat, "extra_shapes"):
            shapes.update(law57_barlat.extra_shapes(mat, nip=nip))
        else:
            shapes.update(
                sigb57=(nip, 3) if nip else (3,),
                eps57=(nip, 3) if nip else (3,),
                thk57=(nip,) if nip else (),
                off57=(nip,) if nip else (),
                pla57=(nip,) if nip else (),
            )
    if getattr(mat, "law", None) in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB") or getattr(mat, "law_name", None) in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
        shapes.update(law49_steinb.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or getattr(mat, "law_name", None) in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
        shapes.update(law79_john_holm.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM") or getattr(mat, "law_name", None) in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
        shapes.update(law50_visc_honey.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM") or getattr(mat, "law_name", None) in ("163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
        shapes.update(law163_crush_foam.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW71_KEYS or getattr(mat, "law_name", None) in _LAW71_KEYS:
        shapes.update(law71_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW76_KEYS or getattr(mat, "law_name", None) in _LAW76_KEYS:
        shapes.update(law76_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in (73, "73", "LAW73", "BARLAT2000", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_BARLAT2000", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "BARLAT2000", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_BARLAT2000", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
        shapes.update(law73_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
        shapes.update(law66_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D") or getattr(mat, "law_name", None) in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D"):
        shapes.update(law74_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000"):
        shapes.update(law87_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW88_KEYS or getattr(mat, "law_name", None) in _LAW88_KEYS:
        shapes.update(law88_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW92_KEYS or getattr(mat, "law_name", None) in _LAW92_KEYS:
        shapes.update(law92_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW93_KEYS or getattr(mat, "law_name", None) in _LAW93_KEYS:
        shapes.update(law93_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW94_KEYS or getattr(mat, "law_name", None) in _LAW94_KEYS:
        shapes.update(law94_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW95_KEYS or getattr(mat, "law_name", None) in _LAW95_KEYS:
        shapes.update(law95_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW100_KEYS or getattr(mat, "law_name", None) in _LAW100_KEYS:
        shapes.update(law100_multi_network.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW101_KEYS or getattr(mat, "law_name", None) in _LAW101_KEYS:
        shapes.update(law101_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW102_KEYS or getattr(mat, "law_name", None) in _LAW102_KEYS:
        shapes.update(law102_dprag2.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW103_KEYS or getattr(mat, "law_name", None) in _LAW103_KEYS:
        shapes.update(law103_hensel_spittel.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW104_KEYS or getattr(mat, "law_name", None) in _LAW104_KEYS:
        shapes.update(law104_drucker.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW105_KEYS or getattr(mat, "law_name", None) in _LAW105_KEYS:
        shapes.update(law105_powder_burn.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW106_KEYS or getattr(mat, "law_name", None) in _LAW106_KEYS:
        shapes.update(law106_jcook_alm.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW107_KEYS or getattr(mat, "law_name", None) in _LAW107_KEYS:
        shapes.update(law107_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW109_KEYS or getattr(mat, "law_name", None) in _LAW109_KEYS:
        shapes.update(law109_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW110_KEYS or getattr(mat, "law_name", None) in _LAW110_KEYS:
        shapes.update(law110_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW124_KEYS or getattr(mat, "law_name", None) in _LAW124_KEYS:
        shapes.update(law124_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW126_KEYS or getattr(mat, "law_name", None) in _LAW126_KEYS:
        shapes.update(law126_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW169_KEYS or getattr(mat, "law_name", None) in _LAW169_KEYS:
        shapes.update(law169_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW117_KEYS or getattr(mat, "law_name", None) in _LAW117_KEYS:
        shapes.update(law117_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW119_KEYS or getattr(mat, "law_name", None) in _LAW119_KEYS:
        shapes.update(law119_seatbelt.extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW90_KEYS or getattr(mat, "law_name", None) in _LAW90_KEYS:
        shapes.update(law90_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW190_KEYS or getattr(mat, "law_name", None) in _LAW190_KEYS:
        shapes.update(law190_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW123_KEYS or getattr(mat, "law_name", None) in _LAW123_KEYS or getattr(mat, "law", None) in _LAW132_KEYS or getattr(mat, "law_name", None) in _LAW132_KEYS:
        shapes.update(law123_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW120_KEYS or getattr(mat, "law_name", None) in _LAW120_KEYS:
        shapes.update(law120_extra_shapes(mat, nip=nip))
    if getattr(mat, "law", None) in _LAW121_KEYS or getattr(mat, "law_name", None) in _LAW121_KEYS:
        shapes.update(law121_extra_shapes(mat, nip=nip))
    if getattr(mat, "fail", None) is not None and mat.fail.type == "FLD":
        shapes["eps_fld"] = (nip, 3) if nip is not None else (3,)
    _lnum = _resolve_law_num(mat)
    if _lnum in _NEW_PORTED_LAWS:
        _fn = getattr(_NEW_PORTED_LAWS[_lnum], "extra_shapes", None)
        if callable(_fn):
            try:
                shapes.update(_fn(mat, nip=nip))
            except TypeError:
                shapes.update(_fn(mat))
    return shapes


def needs_defgrad(mat) -> bool:
    """True if the law is total-strain and needs F from the kernel
    (LAW42 Ogden; LAW62 hyper-visco foam since M37 pack 2; LAW69 hyperelastic M550; LAW82 Ogden M549; LAW95 Bergstrom-Boyce M569; LAW100 Multi-Network M570; LAW101 Bouvard PP M571; LAW190 Du Bois Foam)."""
    return (mat.law in (42, 62, 69, 82, 95, 100, 101)
            or getattr(mat, "law_name", None) in ("LAW42", "LAW62", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN", "LAW95", "95", "BERGSTROM_BOYCE", "BERGSTROM-BOYCE", "MAT_LAW95", "MAT_BERGSTROM_BOYCE", "LAW95_BERGSTROM_BOYCE", "LAW100", "100", "VISC_HYP", "MNF", "MAT_LAW100", "MAT_VISC_HYP", "MAT_MNF", "LAW101", "101", "PP", "MAT_PP", "PLAS_POLY", "MAT_PLAS_POLY", "MAT_LAW101", "LAW101_PP", "LAW101_PLAS_POLY")
            or getattr(mat, "law", None) in _LAW95_KEYS
            or getattr(mat, "law_name", None) in _LAW95_KEYS
            or getattr(mat, "law", None) in _LAW100_KEYS
            or getattr(mat, "law_name", None) in _LAW100_KEYS
            or getattr(mat, "law", None) in _LAW101_KEYS
            or getattr(mat, "law", None) in _LAW190_KEYS
            or getattr(mat, "law_name", None) in _LAW190_KEYS
            or bool((_resolve_law_num(mat) in _NEW_PORTED_LAWS) and callable(getattr(_NEW_PORTED_LAWS[_resolve_law_num(mat)], "needs_defgrad", None)) and _NEW_PORTED_LAWS[_resolve_law_num(mat)].needs_defgrad(mat)))


def needs_env(mat) -> bool:
    """True if the law wants the kernel's per-cycle environment views in
    ``extra`` — current density ``rho`` and internal energy ``eint``"""
    if getattr(mat, "law", None) in _LAW88_KEYS or getattr(mat, "law_name", None) in _LAW88_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW92_KEYS or getattr(mat, "law_name", None) in _LAW92_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW93_KEYS or getattr(mat, "law_name", None) in _LAW93_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW94_KEYS or getattr(mat, "law_name", None) in _LAW94_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW95_KEYS or getattr(mat, "law_name", None) in _LAW95_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW100_KEYS or getattr(mat, "law_name", None) in _LAW100_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW101_KEYS or getattr(mat, "law_name", None) in _LAW101_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW105_KEYS or getattr(mat, "law_name", None) in _LAW105_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW106_KEYS or getattr(mat, "law_name", None) in _LAW106_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW107_KEYS or getattr(mat, "law_name", None) in _LAW107_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW109_KEYS or getattr(mat, "law_name", None) in _LAW109_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW71_KEYS or getattr(mat, "law_name", None) in _LAW71_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW76_KEYS or getattr(mat, "law_name", None) in _LAW76_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW124_KEYS or getattr(mat, "law_name", None) in _LAW124_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW126_KEYS or getattr(mat, "law_name", None) in _LAW126_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW169_KEYS or getattr(mat, "law_name", None) in _LAW169_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW90_KEYS or getattr(mat, "law_name", None) in _LAW90_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW190_KEYS or getattr(mat, "law_name", None) in _LAW190_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW123_KEYS or getattr(mat, "law_name", None) in _LAW123_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW132_KEYS or getattr(mat, "law_name", None) in _LAW132_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW120_KEYS or getattr(mat, "law_name", None) in _LAW120_KEYS:
        return True
    if getattr(mat, "law", None) in _LAW121_KEYS or getattr(mat, "law_name", None) in _LAW121_KEYS:
        return True
    return (getattr(mat, "law", None) in (2, 4, 5, "5", "LAW5", "JWL", 6, 10, "10", "LAW10", "SOIL", "DPRAG", "DPRAG1", 15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", 21, "21", "LAW21", "MAT_LAW21", "MAT_DPRAG", "LAW21_DPRAG", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 24, 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 28, 33, 34, "34", "LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN", 35, 36, 37, "37", "LAW37", "BIPHAS", "BIPHASIC", 38, "38", "LAW38", "VISC_TAB", 40, 43, "43", "LAW43", "HILL_TAB", "LAW43_HILL_TAB", 44, 48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_LAW48", "MAT_ZHAO", "MAT_PLAS_ZHAO", "LAW48_ZHAO", 49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB", 50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", 52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", 57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3", 58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A", 60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", 62, 66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB", 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", 70, 73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL", 74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D", 79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM", 81, 82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", 87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000", 163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM", 190, "190", "LAW190", "FOAM_DUBOIS", "DUBOIS", "MAT_LAW190", "MAT_FOAM_DUBOIS", "LAW190_FOAM_DUBOIS")
            or getattr(mat, "law_name", None) in ("LAW5", "JWL", "LAW10", "SOIL", "DPRAG", "DPRAG1", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", "21", "LAW21", "MAT_LAW21", "MAT_DPRAG", "LAW21_DPRAG", "LAW22", "DAMA", "PLAS_DAMA", "MAT_LAW22", "MAT_DAMA", "MAT_PLAS_DAMA", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", "LAW28", "HONEYCOMB", "HONEYCOMB_SOL", "LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN", "LAW37", "BIPHAS", "BIPHASIC", "LAW38", "VISC_TAB", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB", "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_LAW48", "MAT_ZHAO", "MAT_PLAS_ZHAO", "LAW48_ZHAO", "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB", "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3", "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A", "LAW60", "PLAS_T3", "FABRIC", "MAT_LAW60", "MAT_PLAS_T3", "MAT_FABRIC", "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC", "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL", "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D", "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM", "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN", "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM", "190", "LAW190", "FOAM_DUBOIS", "DUBOIS", "MAT_LAW190", "MAT_FOAM_DUBOIS", "LAW190_FOAM_DUBOIS"))


def solid_update(mat, sig, deps, epsp=None, dt=0.0, extra=None, **kwargs):
    """Dispatch a solid stress update to the material's law.

    Returns (sig, epsp, c): c is the law's current sound speed array or
    None (constant elastic estimate is a bound). ``epsp`` may be None for
    laws without plasticity."""
    if getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000"):
        raise NotImplementedError("LAW87 (/MAT/BARLAT2000) is implemented for shell elements only.")
    if getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
        raise NotImplementedError("LAW73 (/MAT/LAW73 /MAT/HILL_THERM) is implemented for shell elements only.")
    if getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
        if not getattr(mat, "is_3d", False) and not (isinstance(mat, dict) and mat.get("is_3d")):
            raise NotImplementedError("LAW57 (/MAT/BARLAT3) is implemented for shell elements only.")
        sign, epsp_out, c = law57_barlat.solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_sound_speed=True)
        sig[:] = sign
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A"):
        raise NotImplementedError("LAW58 (/MAT/FABR_A) is implemented for shell elements only.")
    if getattr(mat, "law", None) in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG") or getattr(mat, "law_name", None) in ("15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
        raise NotImplementedError("LAW15 is for shell elements only")
    if getattr(mat, "law", None) in _LAW114_KEYS or getattr(mat, "law_name", None) in _LAW114_KEYS:
        raise NotImplementedError("LAW114 (/MAT/SPR_SEATBELT) is for spring elements only.")
    if getattr(mat, "law", None) in _LAW119_KEYS or getattr(mat, "law_name", None) in _LAW119_KEYS:
        raise NotImplementedError("LAW119 (/MAT/SH_SEATBELT) is implemented for shell elements only.")
    if getattr(mat, "law", None) in (32, "32", "LAW32", "HILL") or getattr(mat, "law_name", None) in ("32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL"):
        raise NotImplementedError("LAW32 (HILL anisotropic plasticity) is implemented for shell elements only.")
    if mat.law == 1:
        return law01_elastic.solid_update(mat, sig, deps), epsp, None
    if mat.law == 2:
        sig, epsp = law02_johnson_cook.solid_update(mat, sig, deps, epsp,
                                                    dt, extra)
        return sig, epsp, None
    if mat.law == 3:
        sig, epsp = law03_plas_bost.solid_update(mat, sig, deps, epsp,
                                                  dt, extra)
        return sig, epsp, None
    if mat.law == 36:
        sig, epsp = law36_tabulated.solid_update(mat, sig, deps, epsp, dt,
                                                 extra)
        return sig, epsp, None
    if mat.law == 42:
        return law42_ogden.solid_update(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (24, "24", "LAW24", "CONC", "CONCRETE", "PERIC_CONC") or getattr(mat, "law_name", None) in (24, "24", "LAW24", "CONC", "CONCRETE", "PERIC_CONC"):
        return law24_concrete.solid_update(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (81, "81", "LAW81", "DPRAG_CAP", "MAT_LAW81", "MAT_DPRAG_CAP") or getattr(mat, "law_name", None) in (81, "81", "LAW81", "DPRAG_CAP", "MAT_LAW81", "MAT_DPRAG_CAP"):
        return law81_druckerprager.solid_update(mat, sig, deps, epsp, dt,
                                                extra)
    if mat.law == 83:
        return law83_spotweld.solid_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 62:
        return law62_hypervisco.solid_update(mat, sig, deps, epsp, dt,
                                             extra)
    if mat.law == 0:
        return mat_void.solid_update(mat, sig, deps), epsp, None
    if mat.law == 4:
        return law04_hyd_jcook.solid_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 6:
        sig, epsp, c = law06_hyd_visc.solid_update(mat, sig, deps, epsp, dt, extra)
        return sig, epsp, c
    if mat.law == 999:
        # /MAT/GAS: zero deviator; the pressure and the sound speed come
        # from the attached IDEAL-GAS /EOS through the kernels' EOS block
        return mat_gas.solid_update(mat, sig), epsp, None
    if mat.law == 70:
        sig, c = law70_tabfoam.solid_update(mat, sig, deps, dt, extra)
        return sig, epsp, c
    if mat.law == 35:
        sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt, extra)
        return sig, epsp, c
    if mat.law == 40:
        sig, c = law40_kelvinmax.solid_update(mat, sig, deps, dt, extra)
        return sig, epsp, c
    if mat.law == 44:
        return law44_cowper.solid_update(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_LAW48", "LAW48_ZHAO") or getattr(mat, "law_name", None) in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_LAW48", "LAW48_ZHAO"):
        res = law48_zhao.solid_update_law48(mat, sig, deps, epsp, dt, extra)
        if isinstance(res, tuple):
            sign = res[0]
            epsp_out = res[1] if len(res) > 1 else epsp
            c = res[2] if len(res) > 2 else None
        else:
            sign = res
            epsp_out = epsp
            c = None
        sig[:] = sign
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON"):
        sign, epsp_out, c = law52_gurson.solid_update_law52(mat, sig, deps, epsp, dt, extra, return_sound_speed=True)
        sig[:] = sign
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if mat.law == 33:
        return law33_foamplas.solid_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 28 or getattr(mat, "law_name", None) in ("LAW28", "HONEYCOMB", "HONEYCOMB_SOL"):
        sign, epsp_out, c = law28_honeycomb.solid_update(mat, sig, deps, epsp, dt, extra)
        sig[:] = sign
        return sig, epsp_out, c
    if mat.law == 34 or getattr(mat, "law_name", None) in ("LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"):
        sign, epsp_out, c = law34_boltzmann.solid_update(mat, sig, deps, epsp, dt, extra)
        sig[:] = sign
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (5, "5", "LAW5", "JWL") or getattr(mat, "law_name", None) in ("LAW5", "JWL"):
        _get_law05()
        if law05_solid_update is not None:
            try:
                return law05_solid_update(mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
            except TypeError:
                try:
                    return law05_solid_update(mat, sig, deps, epsp, dt, extra=extra)
                except TypeError:
                    return law05_solid_update(sig, epsp, deps, mat, dt, extra=extra)
        raise NotImplementedError("LAW5 solid_update not available")
    if getattr(mat, "law", None) in (21, "21", "LAW21", "MAT_LAW21", "MAT_DPRAG") or (getattr(mat, "law_name", None) in ("21", "LAW21", "DPRAG", "MAT_DPRAG") and getattr(mat, "law", None) != 10):
        sign, epsp_out, c = law21_dprag.solid_update(mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB") or getattr(mat, "law_name", None) in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
        sign, epsp_out, c = law49_steinb.solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or getattr(mat, "law_name", None) in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
        sign, epsp_out, c = law79_john_holm.solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM") or getattr(mat, "law_name", None) in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
        sign, epsp_out, c = law50_visc_honey.solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (10, "10", "LAW10", "SOIL", "DPRAG", "DPRAG1") or getattr(mat, "law_name", None) in ("LAW10", "SOIL", "DPRAG", "DPRAG1"):
        _get_law10()
        if law10_solid_update is not None:
            return law10_solid_update(mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        raise NotImplementedError("LAW10 solid_update not available")
    if getattr(mat, "law", None) in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS") or getattr(mat, "law_name", None) in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
        sign, epsp_out, c = law25_composite.solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        sig[:] = sign
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (37, "37", "LAW37", "BIPHAS", "BIPHASIC") or getattr(mat, "law_name", None) in ("LAW37", "BIPHAS", "BIPHASIC"):
        return law37_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    if getattr(mat, "law", None) in (38, "38", "LAW38", "VISC_TAB") or getattr(mat, "law_name", None) in ("LAW38", "VISC_TAB"):
        _get_law38()
        if law38_solid_update is not None:
            sign, epsp_out, c = law38_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
            sig[:] = sign
            return sig, epsp_out, c
        raise NotImplementedError("LAW38 solid_update not available")
    if getattr(mat, "law", None) in (22, "22", "LAW22", "DAMA", "PLAS_DAMA") or getattr(mat, "law_name", None) in ("22", "LAW22", "DAMA", "PLAS_DAMA"):
        return law22_dama.solid_update(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (12, "12", "LAW12", "3D_COMP", "COMP_3D") or getattr(mat, "law_name", None) in ("12", "LAW12", "3D_COMP", "COMP_3D"):
        sign, epsp_out, c = law12_comp3d.solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        sig[:] = sign
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (14, "14", "LAW14", "COMPSO", "COMP_SOL") or getattr(mat, "law_name", None) in ("14", "LAW14", "COMPSO", "COMP_SOL"):
        sign, epsp_out, c = law14_compso.solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        sig[:] = sign
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (43, "43", "LAW43", "HILL_TAB", "LAW43_HILL_TAB") or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB"):
        _get_law43()
        if law43_solid_update is not None:
            res = law43_solid_update(mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra)
            if isinstance(res, tuple):
                if len(res) == 3:
                    sign, epsp_out, c = res
                elif len(res) == 2:
                    sign, epsp_out = res
                    c = None
                else:
                    sign, epsp_out, c = res[0], epsp, None
            else:
                sign, epsp_out, c = res, epsp, None
            sig[:] = sign
            if epsp is not None and hasattr(epsp, "__setitem__"):
                try:
                    epsp[:] = epsp_out
                except Exception:
                    pass
            return sig, epsp_out, c
        raise NotImplementedError("LAW43 solid_update not available")
    if getattr(mat, "law", None) in (82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN") or getattr(mat, "law_name", None) in ("82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN"):
        res = law82_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                try:
                    c = law82_sound_speed(mat, rho=extra.get("rho") if extra else None, extra=extra)
                except Exception:
                    c = None
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        sig[:] = sign
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC") or getattr(mat, "law_name", None) in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC"):
        res = law69_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                try:
                    c = law69_sound_speed(mat, rho=extra.get("rho") if extra else None, extra=extra)
                except Exception:
                    c = None
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        sig[:] = sign
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC") or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
        _get_law60()
        res = law60_solid_update(mat, None, deps, sig, epsp_old=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            sign = res[0]
            epsp_out = res[1] if len(res) > 1 else epsp
            c = res[2] if len(res) > 2 else None
        else:
            sign = res
            epsp_out = epsp
            c = None
        sig[:] = sign
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM") or getattr(mat, "law_name", None) in ("163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
        sign, epsp_out, c = law163_crush_foam.solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW71_KEYS or getattr(mat, "law_name", None) in _LAW71_KEYS:
        res = law71_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law71_sound_speed(mat, rho=extra.get("rho") if extra else None, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW76_KEYS or getattr(mat, "law_name", None) in _LAW76_KEYS:
        res = law76_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law76_sound_speed(mat, rho=extra.get("rho") if extra else None, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
        res = law66_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law66_sound_speed(mat, rho=extra.get("rho") if extra else None, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D") or getattr(mat, "law_name", None) in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D"):
        res = law74_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law74_sound_speed(mat, rho=extra.get("rho") if extra else None, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW88_KEYS or getattr(mat, "law_name", None) in _LAW88_KEYS:
        res = law88_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law88_solid_sound_speed(mat, rho=extra.get("rho") if extra else None, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW92_KEYS or getattr(mat, "law_name", None) in _LAW92_KEYS:
        res = law92_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law92_sound_speed(mat, rho=extra.get("rho") if extra else None, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW94_KEYS or getattr(mat, "law_name", None) in _LAW94_KEYS:
        res = law94_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law94_sound_speed(mat, rho=extra.get("rho") if extra else None, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW93_KEYS or getattr(mat, "law_name", None) in _LAW93_KEYS:
        res = law93_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law93_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW95_KEYS or getattr(mat, "law_name", None) in _LAW95_KEYS:
        res = law95_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law95_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW100_KEYS or getattr(mat, "law_name", None) in _LAW100_KEYS:
        res = law100_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law100_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW101_KEYS or getattr(mat, "law_name", None) in _LAW101_KEYS:
        res = law101_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law101_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW102_KEYS or getattr(mat, "law_name", None) in _LAW102_KEYS:
        res = law102_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law102_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW103_KEYS or getattr(mat, "law_name", None) in _LAW103_KEYS:
        res = law103_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law103_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW104_KEYS or getattr(mat, "law_name", None) in _LAW104_KEYS:
        res = law104_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law104_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW105_KEYS or getattr(mat, "law_name", None) in _LAW105_KEYS:
        res = law105_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law105_sound_speed(mat, rho=extra.get("rho") if extra else None, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW106_KEYS or getattr(mat, "law_name", None) in _LAW106_KEYS:
        res = law106_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law106_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW107_KEYS or getattr(mat, "law_name", None) in _LAW107_KEYS:
        res = law107_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law107_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW109_KEYS or getattr(mat, "law_name", None) in _LAW109_KEYS:
        extra_dict = extra if extra is not None else {}
        if epsp is not None and "pla" not in extra_dict:
            extra_dict = dict(extra_dict)
            extra_dict["pla"] = epsp
        res = law109_solid_update(mat, deps=deps, sigo=sig, extra=extra_dict, dt=dt)
        if isinstance(res, tuple):
            if len(res) == 2:
                sign, extra_out = res
                epsp_out = extra_out.get("pla", epsp) if isinstance(extra_out, dict) else epsp
                c = extra_out.get("soundsp") if isinstance(extra_out, dict) else None
                if isinstance(extra, dict) and isinstance(extra_out, dict):
                    extra.update(extra_out)
            elif len(res) == 3:
                sign, epsp_out, c = res
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW110_KEYS or getattr(mat, "law_name", None) in _LAW110_KEYS:
        raise NotImplementedError("material LAW110 (VEGTER) is only supported for shells")
    if getattr(mat, "law", None) in _LAW124_KEYS or getattr(mat, "law_name", None) in _LAW124_KEYS:
        res = law124_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law124_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW126_KEYS or getattr(mat, "law_name", None) in _LAW126_KEYS:
        res = law126_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law126_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW169_KEYS or getattr(mat, "law_name", None) in _LAW169_KEYS:
        res = law169_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law169_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW117_KEYS or getattr(mat, "law_name", None) in _LAW117_KEYS:
        res = law117_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law117_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW123_KEYS or getattr(mat, "law_name", None) in _LAW123_KEYS or getattr(mat, "law", None) in _LAW132_KEYS or getattr(mat, "law_name", None) in _LAW132_KEYS:
        res = law123_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law123_sound_speed(mat, rho=extra.get("rho") if extra else None)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW90_KEYS or getattr(mat, "law_name", None) in _LAW90_KEYS:
        res = law90_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law90_sound_speed(mat, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW190_KEYS or getattr(mat, "law_name", None) in _LAW190_KEYS:
        res = law190_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law190_sound_speed(mat, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if c is None:
            c = law190_sound_speed(mat, extra=extra)
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW120_KEYS or getattr(mat, "law_name", None) in _LAW120_KEYS:
        res = law120_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law120_sound_speed(mat, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    if getattr(mat, "law", None) in _LAW121_KEYS or getattr(mat, "law_name", None) in _LAW121_KEYS:
        res = law121_solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            if len(res) == 3:
                sign, epsp_out, c = res
            elif len(res) == 2:
                sign, epsp_out = res
                c = law121_sound_speed(mat, extra=extra)
            else:
                sign, epsp_out, c = res[0], epsp, None
        else:
            sign, epsp_out, c = res, epsp, None
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = sign
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = epsp_out
            except Exception:
                pass
        return sig, epsp_out, c
    _lnum = _resolve_law_num(mat)
    if _lnum in _NEW_PORTED_LAWS:
        _mod = _NEW_PORTED_LAWS[_lnum]
        _fn = getattr(_mod, "solid_update", None)
        if callable(_fn):
            res = _fn(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_sound_speed=True)
            if isinstance(res, tuple):
                if len(res) == 3:
                    sign, epsp_out, c = res
                elif len(res) == 2:
                    sign, epsp_out = res
                    c = None
                else:
                    sign, epsp_out, c = res[0], epsp, None
            else:
                sign, epsp_out, c = res, epsp, None
            if hasattr(sig, "__setitem__"):
                try:
                    sig[:] = sign
                except Exception:
                    pass
            if epsp is not None and hasattr(epsp, "__setitem__"):
                try:
                    epsp[:] = epsp_out
                except Exception:
                    pass
            return sig, epsp_out, c
    raise NotImplementedError(f"material LAW{mat.law} not ported for solids")


def sound_speed(mat, rho=None, extra=None, is_shell: bool = False):
    """Dispatch sound speed calculation to material law."""
    law = getattr(mat, "law", None)
    law_name = getattr(mat, "law_name", None)
    if law in _LAW120_KEYS or law_name in _LAW120_KEYS:
        return law120_sound_speed(mat, rho=rho, extra=extra, is_shell=is_shell)
    if law in _LAW121_KEYS or law_name in _LAW121_KEYS:
        return law121_sound_speed(mat, rho=rho, extra=extra, is_shell=is_shell)
    if law in _LAW190_KEYS or law_name in _LAW190_KEYS:
        return law190_sound_speed(mat, rho=rho, extra=extra)
    if law in _LAW90_KEYS or law_name in _LAW90_KEYS:
        return law90_sound_speed(mat, rho=rho, extra=extra)
    if law in _LAW124_KEYS or law_name in _LAW124_KEYS:
        return law124_sound_speed(mat, rho=rho)
    if law in _LAW126_KEYS or law_name in _LAW126_KEYS:
        return law126_sound_speed(mat, rho=rho)
    if law in _LAW169_KEYS or law_name in _LAW169_KEYS:
        return law169_sound_speed(mat, rho=rho)
    if law in _LAW117_KEYS or law_name in _LAW117_KEYS:
        return law117_sound_speed(mat, rho=rho, extra=extra, is_shell=is_shell)
    if law in _LAW119_KEYS or law_name in _LAW119_KEYS:
        return law119_seatbelt.sound_speed(mat, rho=rho, extra=extra, is_shell=is_shell)
    if law in _LAW123_KEYS or law_name in _LAW123_KEYS or law in _LAW132_KEYS or law_name in _LAW132_KEYS:
        return law123_sound_speed(mat, rho=rho, extra=extra, is_shell=is_shell)
    if law in _LAW110_KEYS or law_name in _LAW110_KEYS:
        return law110_sound_speed(mat, rho=rho, is_shell=is_shell)
    if law in _LAW109_KEYS or law_name in _LAW109_KEYS:
        return law109_sound_speed(mat, rho=rho, is_shell=is_shell)
    if law in _LAW105_KEYS or law_name in _LAW105_KEYS:
        return law105_sound_speed(mat, rho=rho, extra=extra)
    if law in _LAW107_KEYS or law_name in _LAW107_KEYS:
        if is_shell:
            return law107_sound_speed_shell(mat, rho=rho)
        return law107_sound_speed(mat, rho=rho)
    if law in _LAW106_KEYS or law_name in _LAW106_KEYS:
        if is_shell:
            return law106_sound_speed_shell(mat, rho=rho)
        return law106_sound_speed(mat, rho=rho)
    if law in _LAW104_KEYS or law_name in _LAW104_KEYS:
        if is_shell:
            return law104_sound_speed_shell(mat, rho=rho)
        return law104_sound_speed(mat, rho=rho)
    if law in _LAW103_KEYS or law_name in _LAW103_KEYS:
        return law103_sound_speed(mat, rho=rho)
    if law in _LAW102_KEYS or law_name in _LAW102_KEYS:
        return law102_sound_speed(mat, rho=rho)
    if law in _LAW101_KEYS or law_name in _LAW101_KEYS:
        return law101_sound_speed(mat, rho=rho, extra=extra)
    if law in _LAW100_KEYS or law_name in _LAW100_KEYS:
        return law100_sound_speed(mat, rho=rho)
    if law in _LAW95_KEYS or law_name in _LAW95_KEYS:
        return law95_sound_speed(mat, rho=rho)
    if law in _LAW93_KEYS or law_name in _LAW93_KEYS:
        if is_shell:
            return law93_sound_speed_shell(mat, rho=rho)
        return law93_sound_speed(mat, rho=rho)
    if law in _LAW94_KEYS or law_name in _LAW94_KEYS:
        return law94_sound_speed(mat, rho=rho, extra=extra, is_shell=is_shell)
    if law in _LAW92_KEYS or law_name in _LAW92_KEYS:
        return law92_sound_speed(mat, rho=rho, extra=extra, is_shell=is_shell)
    if law in _LAW88_KEYS or law_name in _LAW88_KEYS:
        if is_shell:
            return law88_tab_hyp.sound_speed_shell(mat, rho=rho, extra=extra)
        return law88_sound_speed(mat, rho=rho, extra=extra)
    if law in (22, "22", "LAW22", "DAMA", "PLAS_DAMA") or law_name in ("22", "LAW22", "DAMA", "PLAS_DAMA"):
        return law22_dama.sound_speed(mat, rho=rho, extra=extra)
    if law in (12, "12", "LAW12", "3D_COMP", "COMP_3D") or law_name in ("12", "LAW12", "3D_COMP", "COMP_3D"):
        return law12_comp3d.sound_speed(mat, rho=rho, extra=extra)
    if law in (14, "14", "LAW14", "COMPSO", "COMP_SOL") or law_name in ("14", "LAW14", "COMPSO", "COMP_SOL"):
        return law14_compso.sound_speed(mat, rho=rho, extra=extra)
    if law in (5, "5", "LAW5", "JWL") or law_name in ("LAW5", "JWL"):
        _get_law05()
        if law05_sound_speed is not None:
            return law05_sound_speed(mat, rho=rho, extra=extra)
        raise NotImplementedError("LAW5 sound_speed not available")
    if law in (21, "21", "LAW21", "MAT_LAW21", "MAT_DPRAG") or (law_name in ("21", "LAW21", "DPRAG", "MAT_DPRAG") and law != 10):
        return law21_dprag.sound_speed(mat, rho=rho, extra=extra)
    if law in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB") or law_name in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
        return law49_steinb.sound_speed_solid(mat, rho=rho, extra=extra)
    if law in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or law_name in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
        return law79_john_holm.sound_speed_solid(mat, rho=rho, extra=extra)
    if law in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM") or law_name in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
        return law50_visc_honey.sound_speed_solid(mat, rho=rho, extra=extra)
    if law in (10, "10", "LAW10", "SOIL", "DPRAG", "DPRAG1") or law_name in ("LAW10", "SOIL", "DPRAG", "DPRAG1"):
        _get_law10()
        if law10_sound_speed is not None:
            return law10_sound_speed(mat, rho=rho, extra=extra)
        raise NotImplementedError("LAW10 sound_speed not available")
    if law == 28 or law_name in ("LAW28", "HONEYCOMB", "HONEYCOMB_SOL"):
        return law28_honeycomb.sound_speed(mat, rho=rho, extra=extra)
    if law in (34, "34", "LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN") or law_name in ("LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"):
        return law34_boltzmann.sound_speed(mat, rho=rho, extra=extra)
    if law in (37, "37", "LAW37", "BIPHAS", "BIPHASIC") or law_name in ("LAW37", "BIPHAS", "BIPHASIC"):
        return law37_sound_speed(mat, rho=rho, extra=extra)
    if law in (24, "24", "LAW24", "CONC", "CONCRETE", "PERIC_CONC") or law_name in (24, "24", "LAW24", "CONC", "CONCRETE", "PERIC_CONC"):
        return law24_concrete.sound_speed(mat, rho=rho, extra=extra)
    if law in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG") or law_name in ("15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
        return law15_chang.sound_speed(mat, rho=rho, extra=extra)
    if law in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS") or law_name in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
        return law25_composite.sound_speed(mat, rho=rho, extra=extra)
    if law in (38, "38", "LAW38", "VISC_TAB") or law_name in ("LAW38", "VISC_TAB"):
        _get_law38()
        if law38_sound_speed is not None:
            return law38_sound_speed(mat, rho=rho, extra=extra)
        raise NotImplementedError("LAW38 sound_speed not available")
    if law in (32, "32", "LAW32", "HILL") or law_name in ("32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL"):
        _get_law32()
        if law32_sound_speed is not None:
            return law32_sound_speed(mat, rho=rho, extra=extra)
        if hasattr(mat, "sound_speed_solid"):
            return mat.sound_speed_solid()
        import numpy as np
        rho0 = float(getattr(mat, "rho0", 0.0) or (mat.params.get("rho", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("MAT_RHO", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        e = float(getattr(mat, "E", 0.0) or getattr(mat, "e", 0.0) or (mat.params.get("e", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("MAT_E", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        r = rho if rho is not None else rho0
        return np.sqrt(e / np.maximum(r, 1e-20))
    if law in (43, "43", "LAW43", "HILL_TAB", "LAW43_HILL_TAB") or law_name in ("43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB"):
        _get_law43()
        if law43_sound_speed is not None:
            return law43_sound_speed(mat, rho=rho, extra=extra)
        if hasattr(mat, "sound_speed_solid"):
            return mat.sound_speed_solid()
        import numpy as np
        rho0 = float(getattr(mat, "rho0", 0.0) or (mat.params.get("rho", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("MAT_RHO", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        e = float(getattr(mat, "E", 0.0) or getattr(mat, "e", 0.0) or (mat.params.get("e", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("MAT_E", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        r = rho if rho is not None else rho0
        return np.sqrt(e / np.maximum(r, 1e-20))
    if law in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC") or law_name in ("60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
        _get_law60()
        return law60_sound_speed(mat, rho=rho, extra=extra)
    if law in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC") or law_name in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC"):
        return law69_sound_speed(mat, rho=rho, extra=extra)
    if law in (81, "81", "LAW81", "DPRAG_CAP", "MAT_LAW81", "MAT_DPRAG_CAP") or law_name in ("81", "LAW81", "DPRAG_CAP", "MAT_LAW81", "MAT_DPRAG_CAP"):
        return law81_druckerprager.sound_speed(mat, rho=rho, extra=extra)
    if law in (82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN") or law_name in ("82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN"):
        return law82_sound_speed(mat, rho=rho, extra=extra)
    if law in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO") or law_name in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO"):
        return law48_zhao.sound_speed_solid_law48(mat, rho0=rho)
    if law in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB") or law_name in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
        return law49_steinb.sound_speed_solid(mat, rho=rho, extra=extra)
    if law in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or law_name in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
        return law79_john_holm.sound_speed_solid_law79(mat, rho=rho, extra=extra)
    if law in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON") or law_name in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON"):
        return law52_gurson.sound_speed_solid_law52(mat, rho=rho)
    if law in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A") or law_name in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A"):
        return sound_speed_shell_law58(mat, rho0=rho)
    if law in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or law_name in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
        return sound_speed_shell_law57(mat, rho0=rho)
    if law in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM") or law_name in ("163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
        return law163_crush_foam.sound_speed_solid(mat, rho=rho, extra=extra)
    if law in _LAW71_KEYS or law_name in _LAW71_KEYS:
        return law71_sound_speed(mat, rho=rho, extra=extra)
    if law in _LAW76_KEYS or law_name in _LAW76_KEYS:
        return law76_sound_speed(mat, rho=rho, extra=extra)
    if law in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or law_name in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
        return law73_sound_speed(mat, rho=rho, extra=extra)
    if law in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or law_name in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
        return law66_sound_speed(mat, rho=rho, extra=extra)
    if law in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D") or law_name in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D"):
        return law74_sound_speed(mat, rho=rho, extra=extra)
    if law in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000") or law_name in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000"):
        return law87_sound_speed(mat, rho0=rho)
    if is_shell and hasattr(mat, "sound_speed_shell"):
        return mat.sound_speed_shell()
    if hasattr(mat, "sound_speed_solid"):
        return mat.sound_speed_solid()
    if hasattr(mat, "sound_speed_shell"):
        return mat.sound_speed_shell()
    _lnum = _resolve_law_num(mat)
    if _lnum in _NEW_PORTED_LAWS:
        _mod = _NEW_PORTED_LAWS[_lnum]
        _fn = getattr(_mod, "sound_speed", None)
        if callable(_fn):
            try:
                return _fn(mat, eps=None, extra=extra, is_shell=is_shell)
            except TypeError:
                try:
                    return _fn(mat, extra=extra, is_shell=is_shell)
                except TypeError:
                    try:
                        return _fn(mat, extra=extra)
                    except TypeError:
                        return _fn(mat)
    raise NotImplementedError(f"material LAW{law} does not implement sound_speed")


def shell_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """Dispatch a plane-stress (shell) update to the material's law."""
    if getattr(mat, "law", None) in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D") or getattr(mat, "law_name", None) in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D"):
        raise NotImplementedError("LAW74 (/MAT/LAW74 /MAT/HILL_3D /MAT/ORTH_PLAS) is implemented for solid elements only.")
    if getattr(mat, "law", None) in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM") or getattr(mat, "law_name", None) in ("163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
        raise NotImplementedError("LAW163 (/MAT/CRUSHABLE_FOAM) is implemented for solid elements only.")
    if getattr(mat, "law", None) in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC") or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
        _get_law60()
        res = law60_shell_update(mat, None, deps, sig, epsp_old=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            s_out = res[0]
            ep_out = res[1] if len(res) > 1 else epsp
        else:
            s_out = res
            ep_out = epsp
        sig[:] = s_out
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = ep_out
            except Exception:
                pass
        return s_out, ep_out
    if getattr(mat, "law", None) in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC") or getattr(mat, "law_name", None) in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC"):
        res = law69_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in (82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN") or getattr(mat, "law_name", None) in ("82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN"):
        res = law82_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in (43, "43", "LAW43", "HILL_TAB", "LAW43_HILL_TAB") or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB"):
        _get_law43()
        if law43_shell_update is not None:
            res = law43_shell_update(mat, sig, deps, epsp, dt, extra)
            if isinstance(res, tuple):
                return res[0], res[1]
            return res, epsp
        raise NotImplementedError("LAW43 shell_update not available in law43_hill_tab")
    if getattr(mat, "law", None) in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG") or getattr(mat, "law_name", None) in ("15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
        res = law15_chang.shell_update(mat, sig, deps, epsp, dt, extra)
        return res[0], res[1]
    if getattr(mat, "law", None) in (32, "32", "LAW32", "HILL") or getattr(mat, "law_name", None) in ("32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL"):
        _get_law32()
        if law32_shell_update is not None:
            return law32_shell_update(mat, sig, deps, epsp, dt, extra)
        raise NotImplementedError("LAW32 shell_update not available in law32_hill")
    if getattr(mat, "law", None) in (38, "38", "LAW38", "VISC_TAB") or getattr(mat, "law_name", None) in ("LAW38", "VISC_TAB"):
        raise NotImplementedError("LAW38 (VISC_TAB tabulated viscoelastic) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in (12, "12", "LAW12", "3D_COMP", "COMP_3D") or getattr(mat, "law_name", None) in ("12", "LAW12", "3D_COMP", "COMP_3D"):
        return law12_comp3d.shell_update(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (14, "14", "LAW14", "COMPSO", "COMP_SOL") or getattr(mat, "law_name", None) in ("14", "LAW14", "COMPSO", "COMP_SOL"):
        return law14_compso.shell_update(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (24, "24", "LAW24", "CONC", "CONCRETE", "PERIC_CONC") or getattr(mat, "law_name", None) in (24, "24", "LAW24", "CONC", "CONCRETE", "PERIC_CONC"):
        raise NotImplementedError("LAW24 (concrete) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in (37, "37", "LAW37", "BIPHAS", "BIPHASIC") or getattr(mat, "law_name", None) in ("LAW37", "BIPHAS", "BIPHASIC"):
        raise NotImplementedError("LAW37 (biphasic fluid/gas) is implemented for 3D solid and SPH elements only.")
    if getattr(mat, "law", None) == 28 or getattr(mat, "law_name", None) in ("LAW28", "HONEYCOMB", "HONEYCOMB_SOL"):
        raise NotImplementedError("LAW28 (HONEYCOMB crushable) is implemented for 3D solid and SPH elements only.")
    if getattr(mat, "law", None) in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM") or getattr(mat, "law_name", None) in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
        raise NotImplementedError("LAW50 (/MAT/VISC_HONEY) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in (4, "4", "LAW4", "HYD_JCOOK") or getattr(mat, "law_name", None) in ("4", "LAW4", "HYD_JCOOK"):
        return law04_hyd_jcook.shell_update(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (5, "5", "LAW5", "JWL") or getattr(mat, "law_name", None) in ("LAW5", "JWL"):
        raise NotImplementedError("LAW5 (JWL explosive) is implemented for 3D solid and SPH elements only.")
    if getattr(mat, "law", None) in (21, "21", "LAW21", "MAT_LAW21", "MAT_DPRAG") or (getattr(mat, "law_name", None) in ("21", "LAW21", "DPRAG", "MAT_DPRAG") and getattr(mat, "law", None) != 10):
        return law21_dprag.shell_update(mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra)
    if getattr(mat, "law", None) in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB") or getattr(mat, "law_name", None) in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
        return law49_steinb.shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    if getattr(mat, "law", None) in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or getattr(mat, "law_name", None) in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
        return law79_john_holm.shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    if getattr(mat, "law", None) in (10, "10", "LAW10", "SOIL", "DPRAG", "DPRAG1") or getattr(mat, "law_name", None) in ("LAW10", "SOIL", "DPRAG", "DPRAG1"):
        raise NotImplementedError("LAW10 (soil/Drucker-Prager) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS") or getattr(mat, "law_name", None) in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
        res = law25_composite.shell_update(mat, sig, deps, epsp, dt, extra)
        return res[0], res[1]
    if mat.law == 34 or getattr(mat, "law_name", None) in ("LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"):
        return law34_boltzmann.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 1:
        return law01_elastic.shell_update(mat, sig, deps), epsp
    if mat.law == 2:
        return law02_johnson_cook.shell_update(mat, sig, deps, epsp, dt,
                                               extra)
    if mat.law == 3:
        return law03_plas_bost.shell_update(mat, sig, deps, epsp, dt,
                                             extra)
    if mat.law == 36:
        return law36_tabulated.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 27:
        return law27_brittle.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 19:
        return law19_fabric.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 0:
        return mat_void.shell_update(mat, sig, deps), epsp
    if mat.law == 44:
        return law44_cowper.shell_update(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO") or getattr(mat, "law_name", None) in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO"):
        return law48_zhao.shell_update_law48(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON"):
        return law52_gurson.shell_update_law52(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (58, "58", "LAW58", "FABR_A", "MAT_FABR_A", "FABRIC_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "MAT_FABR_A", "FABRIC_A", "LAW58_FABR_A"):
        return law58_fabr_a.shell_update_law58(mat, sig, deps, epsp, dt, extra)
    if getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
        res = shell_update_law57(mat, sig, deps, epsp, dt, extra, return_sound_speed=False)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if mat.law == 35:
        return law35_kelvinmax.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 40:
        return law40_kelvinmax.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 62:
        return law62_hypervisco.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 83:
        return law83_spotweld.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 6:
        return law06_hyd_visc.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 22 or getattr(mat, "law_name", None) in ("22", "LAW22", "DAMA", "PLAS_DAMA"):
        res = law22_dama.shell_update(mat, sig, deps, epsp, dt, extra)
        return res[0], res[1]
    if getattr(mat, "law", None) in _LAW71_KEYS or getattr(mat, "law_name", None) in _LAW71_KEYS:
        res = law71_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in _LAW76_KEYS or getattr(mat, "law_name", None) in _LAW76_KEYS:
        res = law76_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
        res = law73_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
        res = law66_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000"):
        res = law87_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in _LAW88_KEYS or getattr(mat, "law_name", None) in _LAW88_KEYS:
        res = law88_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in _LAW92_KEYS or getattr(mat, "law_name", None) in _LAW92_KEYS:
        res = law92_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in _LAW93_KEYS or getattr(mat, "law_name", None) in _LAW93_KEYS:
        res = law93_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in _LAW94_KEYS or getattr(mat, "law_name", None) in _LAW94_KEYS:
        res = law94_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in _LAW95_KEYS or getattr(mat, "law_name", None) in _LAW95_KEYS:
        raise NotImplementedError("LAW95 (/MAT/BERGSTROM_BOYCE) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW100_KEYS or getattr(mat, "law_name", None) in _LAW100_KEYS:
        raise NotImplementedError("LAW100 (/MAT/VISC_HYP) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW101_KEYS or getattr(mat, "law_name", None) in _LAW101_KEYS:
        raise NotImplementedError("LAW101 (/MAT/PP) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW102_KEYS or getattr(mat, "law_name", None) in _LAW102_KEYS:
        raise NotImplementedError("LAW102 (/MAT/DPRAG2) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW103_KEYS or getattr(mat, "law_name", None) in _LAW103_KEYS:
        raise NotImplementedError("LAW103 (/MAT/HENSEL_SPITTEL) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW104_KEYS or getattr(mat, "law_name", None) in _LAW104_KEYS:
        res = law104_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple) and len(res) == 3:
            return res[0], res[1]
        return res
    if getattr(mat, "law", None) in _LAW106_KEYS or getattr(mat, "law_name", None) in _LAW106_KEYS:
        res = law106_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple) and len(res) == 3:
            return res[0], res[1]
        return res
    if getattr(mat, "law", None) in _LAW107_KEYS or getattr(mat, "law_name", None) in _LAW107_KEYS:
        res = law107_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, epsp
    if getattr(mat, "law", None) in _LAW109_KEYS or getattr(mat, "law_name", None) in _LAW109_KEYS:
        extra_dict = extra if extra is not None else {}
        if epsp is not None and "pla" not in extra_dict:
            extra_dict = dict(extra_dict)
            extra_dict["pla"] = epsp
        res = law109_shell_update(mat, deps=deps, sigo=sig, extra=extra_dict, dt=dt)
        if isinstance(res, tuple):
            s_out = res[0]
            extra_out = res[1] if len(res) > 1 else {}
            ep_out = extra_out.get("pla", epsp) if isinstance(extra_out, dict) else epsp
            if isinstance(extra, dict) and isinstance(extra_out, dict):
                extra.update(extra_out)
        else:
            s_out = res
            ep_out = epsp
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = s_out
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = ep_out
            except Exception:
                pass
        return s_out, ep_out
    if getattr(mat, "law", None) in _LAW110_KEYS or getattr(mat, "law_name", None) in _LAW110_KEYS:
        extra_dict = extra if extra is not None else {}
        if epsp is not None and "pla" not in extra_dict:
            extra_dict = dict(extra_dict)
            extra_dict["pla"] = epsp
        res = law110_shell_update(mat, deps=deps, sigo=sig, extra=extra_dict, dt=dt)
        if isinstance(res, tuple):
            s_out = res[0]
            extra_out = res[1] if len(res) > 1 else {}
            ep_out = extra_out.get("pla", epsp) if isinstance(extra_out, dict) else epsp
            if isinstance(extra, dict) and isinstance(extra_out, dict):
                extra.update(extra_out)
        else:
            s_out = res
            ep_out = epsp
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = s_out
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = ep_out
            except Exception:
                pass
        return s_out, ep_out
    if getattr(mat, "law", None) in _LAW124_KEYS or getattr(mat, "law_name", None) in _LAW124_KEYS:
        raise NotImplementedError("material LAW124 (CDPM2) is only supported for solids")
    if getattr(mat, "law", None) in _LAW126_KEYS or getattr(mat, "law_name", None) in _LAW126_KEYS:
        raise NotImplementedError("material LAW126 (HJC) is only supported for solids")
    if getattr(mat, "law", None) in _LAW169_KEYS or getattr(mat, "law_name", None) in _LAW169_KEYS:
        res = law169_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            s_out = res[0]
            ep_out = res[1] if len(res) > 1 else epsp
        else:
            s_out = res
            ep_out = epsp
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = s_out
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = ep_out
            except Exception:
                pass
        return s_out, ep_out
    if getattr(mat, "law", None) in _LAW117_KEYS or getattr(mat, "law_name", None) in _LAW117_KEYS:
        res = law117_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            s_out = res[0]
            ep_out = res[1] if len(res) > 1 else epsp
        else:
            s_out = res
            ep_out = epsp
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = s_out
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = ep_out
            except Exception:
                pass
        return s_out, ep_out
    if getattr(mat, "law", None) in _LAW123_KEYS or getattr(mat, "law_name", None) in _LAW123_KEYS or getattr(mat, "law", None) in _LAW132_KEYS or getattr(mat, "law_name", None) in _LAW132_KEYS:
        res = law123_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
        if isinstance(res, tuple):
            s_out = res[0]
            ep_out = res[1] if len(res) > 1 else epsp
        else:
            s_out = res
            ep_out = epsp
        if hasattr(sig, "__setitem__"):
            try:
                sig[:] = s_out
            except Exception:
                pass
        if epsp is not None and hasattr(epsp, "__setitem__"):
            try:
                epsp[:] = ep_out
            except Exception:
                pass
        return s_out, ep_out
    if getattr(mat, "law", None) in _LAW90_KEYS or getattr(mat, "law_name", None) in _LAW90_KEYS:
        raise NotImplementedError("material LAW90 (tabulated hysteretic foam) is formulated for 3D solid elements only")
    if getattr(mat, "law", None) in _LAW190_KEYS or getattr(mat, "law_name", None) in _LAW190_KEYS:
        raise NotImplementedError("material LAW190 (FOAM_DUBOIS) is formulated for 3D solid elements only")
    if getattr(mat, "law", None) in _LAW119_KEYS or getattr(mat, "law_name", None) in _LAW119_KEYS:
        return law119_seatbelt.shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    if getattr(mat, "law", None) in _LAW120_KEYS or getattr(mat, "law_name", None) in _LAW120_KEYS:
        return law120_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    if getattr(mat, "law", None) in _LAW121_KEYS or getattr(mat, "law_name", None) in _LAW121_KEYS:
        return law121_shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    if getattr(mat, "law", None) in _LAW114_KEYS or getattr(mat, "law_name", None) in _LAW114_KEYS:
        raise NotImplementedError("LAW114 (/MAT/SPR_SEATBELT) is for spring elements only.")
    _lnum = _resolve_law_num(mat)
    if _lnum in _NEW_PORTED_LAWS:
        _mod = _NEW_PORTED_LAWS[_lnum]
        _fn = getattr(_mod, "shell_update", None)
        if callable(_fn):
            res = _fn(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_sound_speed=False)
            if isinstance(res, tuple):
                s_out = res[0]
                ep_out = res[1] if len(res) > 1 else epsp
            else:
                s_out = res
                ep_out = epsp
            if hasattr(sig, "__setitem__"):
                try:
                    sig[:] = s_out
                except Exception:
                    pass
            if epsp is not None and hasattr(epsp, "__setitem__"):
                try:
                    epsp[:] = ep_out
                except Exception:
                    pass
            return s_out, ep_out
    raise NotImplementedError(f"material LAW{mat.law} not ported for shells")


# ----------------------------------------------------------------------------
# Consistent tangents for the implicit solver (M8)
# ----------------------------------------------------------------------------

def solid_tangent(mat, sig=None, epsp=None, epsp_incr=None, extra=None):
    """Dispatch the (n, 6, 6) consistent solid tangent for the implicit
    solve. LAW1 returns the constant elastic C broadcast over the group;
    LAW2 returns the CONSISTENT (algorithmic) elastoplastic tangent of the
    radial return (see law02.consistent_solid_tangent for the derivation);
    LAW36 (M13) the same algebra with the hardening slope from the table's
    local segment (law36.consistent_solid_tangent); LAW42 (M14) the exact
    spectral SPATIAL tangent of the total-form Ogden stress, built from
    the trial deformation gradient the element passes in ``extra["F"]``
    (law42.consistent_solid_tangent — pairs with the assembler's K_geo)."""
    import numpy as np
    if getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
        raise NotImplementedError("LAW57 (/MAT/BARLAT3) has no solid tangent (shells only)")
    if getattr(mat, "law", None) in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "MAT_FABRIC_A", "LAW58_FABR_A"):
        raise NotImplementedError("LAW58 (/MAT/FABR_A) has no solid tangent (shells only)")
    if getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
        raise NotImplementedError("LAW73 (/MAT/LAW73 /MAT/HILL_THERM) has no solid tangent (shells only)")
    if getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000"):
        raise NotImplementedError("LAW87 (/MAT/BARLAT2000) has no solid tangent (shells only)")
    if sig is None:
        sig = np.zeros(6, dtype=np.float64)
    n = sig.shape[0] if sig.ndim > 1 else 1
    if mat.law == 1:
        return np.broadcast_to(law01_elastic.solid_tangent(mat),
                               (n, 6, 6)).copy()
    if mat.law == 2:
        return law02_johnson_cook.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr)
    if mat.law == 3:
        return law03_plas_bost.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr)
    if mat.law == 36:
        return law36_tabulated.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr)
    if mat.law == 42:
        if extra is None or "F" not in extra:
            raise NotImplementedError(
                "LAW42 implicit tangent needs the deformation gradient — "
                "supported for the solid kernels (hexa8/tetra4) under "
                "/IMPL/NONLIN only (PORTING_GUIDE M14)")
        return law42_ogden.consistent_solid_tangent(mat, extra["F"])
    if getattr(mat, "law", None) in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC") or getattr(mat, "law_name", None) in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC"):
        if extra is not None and "F" in extra:
            return law69_solid_tangent(mat, extra["F"])
        eps = extra.get("eps") if extra else None
        if eps is None:
            eps = np.zeros((n, 6), dtype=np.float64)
        return law69_solid_tangent(mat, eps)
    if getattr(mat, "law", None) in (82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN") or getattr(mat, "law_name", None) in ("82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN"):
        if extra is None or "F" not in extra:
            raise NotImplementedError(
                "LAW82 implicit tangent needs the deformation gradient — "
                "supported for the solid kernels (hexa8/tetra4) under "
                "/IMPL/NONLIN only")
        return law82_solid_tangent(mat, extra["F"])
    if getattr(mat, "law", None) in (24, "24", "LAW24", "CONC", "CONCRETE", "PERIC_CONC") or getattr(mat, "law_name", None) in (24, "24", "LAW24", "CONC", "CONCRETE", "PERIC_CONC"):
        return law24_concrete.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 35:
        return law35_kelvinmax.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 40:
        return law40_kelvinmax.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 44:
        return law44_cowper.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if getattr(mat, "law", None) in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO") or getattr(mat, "law_name", None) in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO"):
        return law48_zhao.tangent_law48_solid(mat, sig, epsp, epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS"):
        return law52_gurson.tangent_law52_solid(mat, sig, epsp, epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC") or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
        _get_law60()
        return law60_solid_tangent(
            mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (81, "81", "LAW81", "DPRAG_CAP", "MAT_LAW81", "MAT_DPRAG_CAP") or getattr(mat, "law_name", None) in (81, "81", "LAW81", "DPRAG_CAP", "MAT_LAW81", "MAT_DPRAG_CAP"):
        return law81_druckerprager.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 62:
        return law62_hypervisco.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 83:
        return law83_spotweld.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 6:
        return law06_hyd_visc.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra=extra)
    if mat.law == 33:
        return law33_foamplas.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if getattr(mat, "law", None) == 28 or getattr(mat, "law_name", None) in ("LAW28", "HONEYCOMB", "HONEYCOMB_SOL"):
        return law28_honeycomb.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 4:
        return law04_hyd_jcook.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 34 or getattr(mat, "law_name", None) in ("LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"):
        return law34_boltzmann.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if getattr(mat, "law", None) in (5, "5", "LAW5", "JWL") or getattr(mat, "law_name", None) in ("LAW5", "JWL"):
        _get_law05()
        if law05_solid_tangent is not None:
            try:
                return law05_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
            except TypeError:
                return law05_solid_tangent(sig, epsp, mat, extra=extra)
        raise NotImplementedError("LAW5 solid_tangent not available")
    if getattr(mat, "law", None) in (21, "21", "LAW21", "MAT_LAW21", "MAT_DPRAG") or (getattr(mat, "law_name", None) in ("21", "LAW21", "DPRAG", "MAT_DPRAG") and getattr(mat, "law", None) != 10):
        return law21_dprag.consistent_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB") or getattr(mat, "law_name", None) in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
        return law49_steinb.consistent_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or getattr(mat, "law_name", None) in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
        return law79_john_holm.consistent_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (10, "10", "LAW10", "SOIL", "DPRAG", "DPRAG1") or getattr(mat, "law_name", None) in ("LAW10", "SOIL", "DPRAG", "DPRAG1"):
        _get_law10()
        if law10_solid_tangent is not None:
            return law10_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
        raise NotImplementedError("LAW10 solid_tangent not available")
    if getattr(mat, "law", None) in (37, "37", "LAW37", "BIPHAS", "BIPHASIC") or getattr(mat, "law_name", None) in ("LAW37", "BIPHAS", "BIPHASIC"):
        return law37_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS") or getattr(mat, "law_name", None) in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
        return law25_composite.consistent_solid_tangent(mat, sig, epsp=epsp, dt=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (38, "38", "LAW38", "VISC_TAB") or getattr(mat, "law_name", None) in ("LAW38", "VISC_TAB"):
        _get_law38()
        if law38_solid_tangent is not None:
            return law38_solid_tangent(mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
        raise NotImplementedError("LAW38 solid_tangent not available")
    if getattr(mat, "law", None) in (22, "22", "LAW22", "DAMA", "PLAS_DAMA") or getattr(mat, "law_name", None) in ("22", "LAW22", "DAMA", "PLAS_DAMA"):
        return law22_dama.consistent_solid_tangent(mat, sig, epsp=epsp, dt=0.0, extra=extra, epsp_incr=epsp_incr)
    if getattr(mat, "law", None) in (12, "12", "LAW12", "3D_COMP", "COMP_3D") or getattr(mat, "law_name", None) in ("12", "LAW12", "3D_COMP", "COMP_3D"):
        return law12_comp3d.consistent_solid_tangent(mat, sig, epsp=epsp, dt=0.0, extra=extra, epsp_incr=epsp_incr)
    if getattr(mat, "law", None) in (14, "14", "LAW14", "COMPSO", "COMP_SOL") or getattr(mat, "law_name", None) in ("14", "LAW14", "COMPSO", "COMP_SOL"):
        return law14_compso.consistent_solid_tangent(mat, sig, epsp=epsp, dt=0.0, extra=extra, epsp_incr=epsp_incr)
    if getattr(mat, "law", None) in (43, "43", "LAW43", "HILL_TAB", "LAW43_HILL_TAB") or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB"):
        _get_law43()
        if law43_solid_tangent is not None:
            return law43_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
        raise NotImplementedError("LAW43 solid_tangent not available")
    if getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS"):
        return law52_gurson.tangent_law52_solid(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM") or getattr(mat, "law_name", None) in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
        return law50_visc_honey.consistent_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM") or getattr(mat, "law_name", None) in ("163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
        return law163_crush_foam.consistent_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in _LAW71_KEYS or getattr(mat, "law_name", None) in _LAW71_KEYS:
        return law71_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW76_KEYS or getattr(mat, "law_name", None) in _LAW76_KEYS:
        return law76_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
        return law66_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D") or getattr(mat, "law_name", None) in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D"):
        return law74_solid_tangent(mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in _LAW88_KEYS or getattr(mat, "law_name", None) in _LAW88_KEYS:
        eps_trial = extra.get("eps") if extra else None
        if eps_trial is None:
            if sig is not None:
                eps_trial = np.zeros_like(sig)
            else:
                eps_trial = np.zeros(6, dtype=np.float64)
        return law88_solid_tangent(eps_trial, mat)
    if getattr(mat, "law", None) in _LAW92_KEYS or getattr(mat, "law_name", None) in _LAW92_KEYS:
        eps_trial = extra.get("eps") if extra else None
        if eps_trial is None:
            if sig is not None:
                eps_trial = np.zeros_like(sig)
            else:
                eps_trial = np.zeros(6, dtype=np.float64)
        return law92_consistent_tangent(mat, eps_trial, is_shell=False)
    if getattr(mat, "law", None) in _LAW93_KEYS or getattr(mat, "law_name", None) in _LAW93_KEYS:
        pla_val = epsp if epsp is not None else 0.0
        return law93_consistent_tangent(mat, sig if sig is not None else np.zeros(6), pla=pla_val, plane_stress=False)
    if getattr(mat, "law", None) in _LAW94_KEYS or getattr(mat, "law_name", None) in _LAW94_KEYS:
        eps_trial = extra.get("eps") if extra else None
        if eps_trial is None:
            if sig is not None:
                eps_trial = np.zeros_like(sig)
            else:
                eps_trial = np.zeros(6, dtype=np.float64)
        return law94_consistent_tangent(mat, eps_trial, is_shell=False)
    if getattr(mat, "law", None) in _LAW95_KEYS or getattr(mat, "law_name", None) in _LAW95_KEYS:
        eps_trial = extra.get("eps") if extra else None
        if eps_trial is None:
            if sig is not None:
                eps_trial = np.zeros_like(sig)
            else:
                eps_trial = np.zeros(6, dtype=np.float64)
        return law95_consistent_tangent(mat, eps_trial, extra=extra)
    if getattr(mat, "law", None) in _LAW100_KEYS or getattr(mat, "law_name", None) in _LAW100_KEYS:
        return law100_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW101_KEYS or getattr(mat, "law_name", None) in _LAW101_KEYS:
        return law101_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW102_KEYS or getattr(mat, "law_name", None) in _LAW102_KEYS:
        return law102_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW103_KEYS or getattr(mat, "law_name", None) in _LAW103_KEYS:
        return law103_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW104_KEYS or getattr(mat, "law_name", None) in _LAW104_KEYS:
        return law104_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW105_KEYS or getattr(mat, "law_name", None) in _LAW105_KEYS:
        return law105_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW106_KEYS or getattr(mat, "law_name", None) in _LAW106_KEYS:
        return law106_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW107_KEYS or getattr(mat, "law_name", None) in _LAW107_KEYS:
        return law107_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW109_KEYS or getattr(mat, "law_name", None) in _LAW109_KEYS:
        return law109_solid_tangent(mat, sig=sig, extra=extra)
    if getattr(mat, "law", None) in _LAW110_KEYS or getattr(mat, "law_name", None) in _LAW110_KEYS:
        raise NotImplementedError("material LAW110 (VEGTER) is only supported for shells")
    if getattr(mat, "law", None) in _LAW124_KEYS or getattr(mat, "law_name", None) in _LAW124_KEYS:
        return law124_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW126_KEYS or getattr(mat, "law_name", None) in _LAW126_KEYS:
        return law126_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW169_KEYS or getattr(mat, "law_name", None) in _LAW169_KEYS:
        return law169_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW117_KEYS or getattr(mat, "law_name", None) in _LAW117_KEYS:
        return law117_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW123_KEYS or getattr(mat, "law_name", None) in _LAW123_KEYS or getattr(mat, "law", None) in _LAW132_KEYS or getattr(mat, "law_name", None) in _LAW132_KEYS:
        return law123_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW90_KEYS or getattr(mat, "law_name", None) in _LAW90_KEYS:
        return law90_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW190_KEYS or getattr(mat, "law_name", None) in _LAW190_KEYS:
        return law190_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW120_KEYS or getattr(mat, "law_name", None) in _LAW120_KEYS:
        c = law120_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
        if c.ndim == 2 and n > 1:
            return np.broadcast_to(c, (n, 6, 6)).copy()
        return c
    if getattr(mat, "law", None) in _LAW121_KEYS or getattr(mat, "law_name", None) in _LAW121_KEYS:
        c = law121_solid_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
        if c.ndim == 2 and n > 1:
            return np.broadcast_to(c, (n, 6, 6)).copy()
        return c
    _lnum = _resolve_law_num(mat)
    if _lnum in _NEW_PORTED_LAWS:
        _mod = _NEW_PORTED_LAWS[_lnum]
        _fn = getattr(_mod, "solid_tangent", None) or getattr(_mod, "consistent_solid_tangent", None)
        if callable(_fn):
            try:
                res = _fn(mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
            except TypeError:
                try:
                    res = _fn(mat, sig=sig, epsp=epsp, extra=extra)
                except TypeError:
                    res = _fn(mat)
            if hasattr(res, "ndim") and res.ndim == 2 and n > 1:
                return np.broadcast_to(res, (n, 6, 6)).copy()
            return res
    raise NotImplementedError(
        f"material LAW{mat.law} has no implicit solid tangent (LAW1 "
        f"elastic, LAW2, LAW4, LAW5, LAW6, LAW10, LAW24, LAW28, LAW33, LAW34, LAW35, LAW36, LAW38, LAW40, LAW44, LAW62, LAW81 and LAW83, LAW42 hyperelastic "
        f"are ported; LAW27 is deferred — see PORTING_GUIDE M14)")


consistent_solid_tangent = solid_tangent


def resolve_curves(mat, model, log=None):
    """Wire curve resolution hook for /FUNCT references so model.curves can be accessed by the kernel."""
    if getattr(mat, "law", None) in _LAW120_KEYS or getattr(mat, "law_name", None) in _LAW120_KEYS:
        return law120_resolve(mat, model, log)

    if getattr(mat, "law", None) in _LAW121_KEYS or getattr(mat, "law_name", None) in _LAW121_KEYS:
        return law121_resolve(mat, model, log)

    if getattr(mat, "law", None) in _LAW190_KEYS or getattr(mat, "law_name", None) in _LAW190_KEYS:
        if hasattr(law190_dubois, "resolve"):
            return law190_dubois.resolve(mat, model, log)

    if getattr(mat, "law", None) in _LAW90_KEYS or getattr(mat, "law_name", None) in _LAW90_KEYS:
        if hasattr(law90_foam, "resolve"):
            return law90_foam.resolve(mat, model, log)

    if getattr(mat, "law", None) in _LAW110_KEYS or getattr(mat, "law_name", None) in _LAW110_KEYS:
        if hasattr(law110_vegter, "resolve"):
            return law110_vegter.resolve(mat, model, log)

    if getattr(mat, "law", None) in _LAW109_KEYS or getattr(mat, "law_name", None) in _LAW109_KEYS:
        if hasattr(law109_tab_plas, "resolve"):
            return law109_tab_plas.resolve(mat, model, log)

    if getattr(mat, "law", None) in _LAW105_KEYS or getattr(mat, "law_name", None) in _LAW105_KEYS:
        if hasattr(law105_powder_burn, "resolve"):
            return law105_powder_burn.resolve(mat, model, log)

    if getattr(mat, "law", None) in _LAW106_KEYS or getattr(mat, "law_name", None) in _LAW106_KEYS:
        if hasattr(law106_jcook_alm, "resolve"):
            return law106_jcook_alm.resolve(mat, model, log)

    if getattr(mat, "law", None) in _LAW107_KEYS or getattr(mat, "law_name", None) in _LAW107_KEYS:
        if hasattr(law107_paper_light, "resolve"):
            return law107_paper_light.resolve(mat, model, log)

    if getattr(mat, "law", None) in _LAW93_KEYS or getattr(mat, "law_name", None) in _LAW93_KEYS:
        if hasattr(law93_orth_hill, "resolve"):
            return law93_orth_hill.resolve(mat)
    if getattr(mat, "law", None) in _LAW94_KEYS or getattr(mat, "law_name", None) in _LAW94_KEYS:
        if hasattr(law94_yeoh, "resolve"):
            return law94_yeoh.resolve(mat, model, log)
    if getattr(mat, "law", None) in _LAW92_KEYS or getattr(mat, "law_name", None) in _LAW92_KEYS:
        if hasattr(law92_arruda_boyce, "resolve"):
            return law92_arruda_boyce.resolve(mat, model, log)
    if getattr(mat, "law", None) in _LAW88_KEYS or getattr(mat, "law_name", None) in _LAW88_KEYS:
        if hasattr(law88_tab_hyp, "resolve"):
            return law88_tab_hyp.resolve(mat, model, log)
    if getattr(mat, "law", None) in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM") or getattr(mat, "law_name", None) in ("163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
        return law163_crush_foam.resolve(mat, model, log)
    if getattr(mat, "law", None) in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM") or getattr(mat, "law_name", None) in ("50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
        if hasattr(law50_visc_honey, "resolve"):
            return law50_visc_honey.resolve(mat, model, log)
    if getattr(mat, "law", None) in (38, "38", "LAW38", "VISC_TAB") or getattr(mat, "law_name", None) in ("LAW38", "VISC_TAB"):
        _get_law38()
        if law38_visc_tab is not None and hasattr(law38_visc_tab, "resolve"):
            return law38_visc_tab.resolve(mat, model, log)
    if getattr(mat, "law", None) in (43, "43", "LAW43", "HILL_TAB", "LAW43_HILL_TAB") or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB"):
        _get_law43()
        if law43_hill_tab is not None and hasattr(law43_hill_tab, "resolve"):
            return law43_hill_tab.resolve(mat, model, log)
    if getattr(mat, "law", None) in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC") or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
        _get_law60()
        if law60_plast3 is not None and hasattr(law60_plast3, "resolve"):
            return law60_plast3.resolve(mat, model, log)
    if getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
        if law57_barlat is not None and hasattr(law57_barlat, "resolve"):
            return law57_barlat.resolve(mat, model, log)
    if getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
        if hasattr(law73_hill_therm, "resolve"):
            return law73_hill_therm.resolve(mat, model, log)
    if getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
        if hasattr(law66_plas_tab, "resolve"):
            return law66_plas_tab.resolve(mat, model, log)
    if getattr(mat, "law", None) in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D") or getattr(mat, "law_name", None) in ("74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL", "LAW74_HILL_3D"):
        if hasattr(law74_hill_3d, "resolve"):
            return law74_hill_3d.resolve(mat, model, log)
    if getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000"):
        if hasattr(law87_barlat2000, "resolve"):
            return law87_barlat2000.resolve(mat, model, log)


def shell_membrane_tangent(mat):
    """(3, 3) plane-stress membrane/bending tangent for the shell implicit
    tangent — the constant elastic matrix (LAW1 shells use it for every
    layer; the shell kernels take this fast path so the M8 results stay
    byte-identical). Elastoplastic shells go through the per-layer
    ``shell_layer_tangent`` instead (M11)."""
    import numpy as np
    if getattr(mat, "law", None) == 1:
        return law01_elastic.shell_membrane_tangent(mat)
    if getattr(mat, "law", None) == 19:
        return law19_fabric.shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in (58, "58", "LAW58", "FABR_A", "MAT_FABR_A", "FABRIC_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "MAT_FABR_A", "FABRIC_A", "LAW58_FABR_A"):
        return law58_fabr_a.shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
        import numpy as np
        e = float(getattr(mat, "E", 0.0) or getattr(mat, "e", 0.0) or (mat.params.get("e", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("MAT_E", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        nu = float(getattr(mat, "nu", 0.0) or (mat.params.get("nu", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("MAT_NU", 0.0) if hasattr(mat, "params") else 0.0) or 0.3)
        c = e / max(1.0 - nu * nu, 1e-15)
        g = e / max(2.0 * (1.0 + nu), 1e-15)
        return np.array([
            [c, nu * c, 0.0],
            [nu * c, c, 0.0],
            [0.0, 0.0, g],
        ])
    if mat.law == 44:
        return law44_cowper.shell_membrane_tangent(mat)
    if mat.law == 3:
        return law03_plas_bost.shell_membrane_tangent(mat)
    if mat.law == 34 or getattr(mat, "law_name", None) in ("LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"):
        return law34_boltzmann.shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG") or getattr(mat, "law_name", None) in ("15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
        return law15_chang.shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS") or getattr(mat, "law_name", None) in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
        return law25_composite.shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in (32, "32", "LAW32", "HILL") or getattr(mat, "law_name", None) in ("32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL"):
        _get_law32()
        if law32_hill is not None and hasattr(law32_hill, "shell_membrane_tangent"):
            return law32_hill.shell_membrane_tangent(mat)
        import numpy as np
        e = float(getattr(mat, "E", 0.0) or getattr(mat, "e", 0.0) or (mat.params.get("e", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("MAT_E", 0.0) if hasattr(mat, "params") else 0.0))
        nu = float(getattr(mat, "nu", 0.0) or (mat.params.get("nu", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("MAT_NU", 0.0) if hasattr(mat, "params") else 0.0))
        c = e / max(1.0 - nu * nu, 1e-15)
        g = e / max(2.0 * (1.0 + nu), 1e-15)
        return np.array([
            [c, nu * c, 0.0],
            [nu * c, c, 0.0],
            [0.0, 0.0, g],
        ])
    if getattr(mat, "law", None) in (43, "43", "LAW43", "HILL_TAB", "LAW43_HILL_TAB") or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB"):
        _get_law43()
        if law43_hill_tab is not None and hasattr(law43_hill_tab, "shell_membrane_tangent"):
            return law43_hill_tab.shell_membrane_tangent(mat)
        import numpy as np
        e = float(getattr(mat, "E", 0.0) or getattr(mat, "e", 0.0) or (mat.params.get("e", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("MAT_E", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        nu = float(getattr(mat, "nu", 0.0) or (mat.params.get("nu", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("MAT_NU", 0.0) if hasattr(mat, "params") else 0.0) or 0.3)
        c = e / max(1.0 - nu * nu, 1e-15)
        g = e / max(2.0 * (1.0 + nu), 1e-15)
        return np.array([
            [c, nu * c, 0.0],
            [nu * c, c, 0.0],
            [0.0, 0.0, g],
        ])
    if getattr(mat, "law", None) in (22, "22", "LAW22", "DAMA", "PLAS_DAMA") or getattr(mat, "law_name", None) in ("22", "LAW22", "DAMA", "PLAS_DAMA"):
        return law22_dama.shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC") or getattr(mat, "law_name", None) in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC"):
        import numpy as np
        nu = float(getattr(mat, "nu", 0.495))
        e = float(getattr(mat, "E", 0.0) or getattr(mat, "gmax", 10.0) * (1.0 + nu))
        c = e / max(1.0 - nu * nu, 1e-15)
        g = float(getattr(mat, "G", getattr(mat, "g0", 0.0)) or e / (2.0 * (1.0 + nu)))
        return np.array([
            [c, nu * c, 0.0],
            [nu * c, c, 0.0],
            [0.0, 0.0, g],
        ])
    if getattr(mat, "law", None) in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC") or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
        import numpy as np
        e = float(getattr(mat, "E", 0.0) or getattr(mat, "e0", 0.0) or (mat.params.get("E", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("e0", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        nu = float(getattr(mat, "nu", 0.0) or (mat.params.get("nu", 0.0) if hasattr(mat, "params") else 0.0) or 0.3)
        c = e / max(1.0 - nu * nu, 1e-15)
        g = e / max(2.0 * (1.0 + nu), 1e-15)
        return np.array([
            [c, nu * c, 0.0],
            [nu * c, c, 0.0],
            [0.0, 0.0, g],
        ])
    if getattr(mat, "law", None) in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO") or getattr(mat, "law_name", None) in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO"):
        return law48_zhao.shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS"):
        return law52_shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
        return law73_shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in _LAW71_KEYS or getattr(mat, "law_name", None) in _LAW71_KEYS:
        return law71_shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in _LAW76_KEYS or getattr(mat, "law_name", None) in _LAW76_KEYS:
        return law76_shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
        return law66_shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000"):
        return law87_shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in _LAW88_KEYS or getattr(mat, "law_name", None) in _LAW88_KEYS:
        eps_trial = np.zeros(3, dtype=np.float64)
        return law88_shell_membrane_tangent(eps_trial, mat)
    if getattr(mat, "law", None) in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or getattr(mat, "law_name", None) in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
        raise NotImplementedError("LAW79 (Johnson-Holmquist) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW104_KEYS or getattr(mat, "law_name", None) in _LAW104_KEYS:
        e = float(getattr(mat, "young", 0.0) or getattr(mat, "E", 0.0) or getattr(mat, "e", 0.0) or (mat.params.get("young", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        nu = float(getattr(mat, "nu", 0.0) or (mat.params.get("nu", 0.0) if hasattr(mat, "params") else 0.0) or 0.3)
        c = e / max(1.0 - nu * nu, 1e-15)
        g = e / max(2.0 * (1.0 + nu), 1e-15)
        return np.array([
            [c, nu * c, 0.0],
            [nu * c, c, 0.0],
            [0.0, 0.0, g],
        ])
    if getattr(mat, "law", None) in _LAW106_KEYS or getattr(mat, "law_name", None) in _LAW106_KEYS:
        e = float(getattr(mat, "young", 0.0) or getattr(mat, "E", 0.0) or getattr(mat, "e", 0.0) or (mat.params.get("young", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        nu = float(getattr(mat, "nu", 0.0) or (mat.params.get("nu", 0.0) if hasattr(mat, "params") else 0.0) or 0.3)
        c = e / max(1.0 - nu * nu, 1e-15)
        g = e / max(2.0 * (1.0 + nu), 1e-15)
        return np.array([
            [c, nu * c, 0.0],
            [nu * c, c, 0.0],
            [0.0, 0.0, g],
        ])
    if getattr(mat, "law", None) in _LAW124_KEYS or getattr(mat, "law_name", None) in _LAW124_KEYS:
        raise NotImplementedError("LAW124 (CDPM2) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW126_KEYS or getattr(mat, "law_name", None) in _LAW126_KEYS:
        raise NotImplementedError("LAW126 (Johnson-Holmquist Concrete / HJC) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW169_KEYS or getattr(mat, "law_name", None) in _LAW169_KEYS:
        e = float(getattr(mat, "young", 0.0) or getattr(mat, "E", 0.0) or getattr(mat, "e", 0.0) or (mat.params.get("young", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("e", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        nu = float(getattr(mat, "nu", 0.0) or (mat.params.get("nu", 0.0) if hasattr(mat, "params") else 0.0) or 0.3)
        c = e / max(1.0 - nu * nu, 1e-15)
        g = e / max(2.0 * (1.0 + nu), 1e-15)
        return np.array([
            [c, nu * c, 0.0],
            [nu * c, c, 0.0],
            [0.0, 0.0, g],
        ])
    if getattr(mat, "law", None) in _LAW117_KEYS or getattr(mat, "law_name", None) in _LAW117_KEYS:
        return law117_shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in _LAW123_KEYS or getattr(mat, "law_name", None) in _LAW123_KEYS or getattr(mat, "law", None) in _LAW132_KEYS or getattr(mat, "law_name", None) in _LAW132_KEYS:
        return law123_shell_tangent(mat)
    if getattr(mat, "law", None) in _LAW120_KEYS or getattr(mat, "law_name", None) in _LAW120_KEYS:
        return law120_shell_membrane_tangent(mat)
    if getattr(mat, "law", None) in _LAW121_KEYS or getattr(mat, "law_name", None) in _LAW121_KEYS:
        e = float(getattr(mat, "young", 0.0) or getattr(mat, "E", 0.0) or getattr(mat, "e", 0.0) or (mat.params.get("young", 0.0) if hasattr(mat, "params") else 0.0) or (mat.params.get("E", 0.0) if hasattr(mat, "params") else 0.0) or 1.0)
        nu = float(getattr(mat, "nu", 0.0) or (mat.params.get("nu", 0.0) if hasattr(mat, "params") else 0.0) or 0.3)
        c = e / max(1.0 - nu * nu, 1e-15)
        g = e / max(2.0 * (1.0 + nu), 1e-15)
        return np.array([
            [c, nu * c, 0.0],
            [nu * c, c, 0.0],
            [0.0, 0.0, g],
        ])
    _lnum = _resolve_law_num(mat)
    if _lnum in _NEW_PORTED_LAWS:
        _mod = _NEW_PORTED_LAWS[_lnum]
        _fn = getattr(_mod, "shell_tangent", None) or getattr(_mod, "consistent_shell_tangent", None) or getattr(_mod, "shell_membrane_tangent", None)
        if callable(_fn):
            try:
                res = _fn(mat)
            except TypeError:
                res = _fn(mat, sig=None, extra=None)
            if hasattr(res, "ndim") and res.ndim == 3:
                return res[0]
            return res
    raise NotImplementedError(
        f"material LAW{mat.law} has no implicit shell tangent (LAW1 elastic, "
        f"LAW3 plas_bost, LAW19 fabric, LAW34 Boltzmann, LAW32 Hill and LAW2/44 elastoplastic are ported; see PORTING_GUIDE)")


def shell_layer_tangent(mat, sig=None, epsp=None, epsp_incr=None, extra=None):
    """Dispatch the (n, 3, 3) consistent PLANE-STRESS tangent of one
    through-thickness layer for the implicit shell tangents (M11). LAW1
    broadcasts the elastic matrix; LAW2 returns the CONSISTENT (algorithmic)
    tangent of the Iplas=2 radial projection (see
    law02.consistent_shell_tangent for the derivation); LAW36 (M13) the
    same projection tangent with the table's local hardening slope;
    LAW27 (M15) the damaged fixed-crack unilateral tangent built from the
    layer's trial crack state passed in ``extra`` (the eps27/crk27/ang27/
    dmg27/layfail views — law27.consistent_shell_tangent for the
    per-branch derivation: uncracked / open-frozen / open-growing /
    closed / broken); LAW19 (M524) the orthotropic fabric tangent with
    RCOMP and beta compression scaling; LAW32 (M542) consistent Hill tangent;
    LAW60 consistent plastic fabric tangent."""
    if sig is None:
        sig = np.zeros((1, 3))
    n = sig.shape[0]
    if getattr(mat, "law", None) in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC") or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
        _get_law60()
        return law60_shell_tangent(mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC") or getattr(mat, "law_name", None) in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "LAW69_HYPERELASTIC"):
        eps = extra.get("eps") if extra else None
        if eps is None:
            eps = np.zeros((n, 3), dtype=np.float64)
        return law69_shell_tangent(mat, eps)
    if getattr(mat, "law", None) in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG") or getattr(mat, "law_name", None) in ("15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
        return law15_chang.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if getattr(mat, "law", None) in (32, "32", "LAW32", "HILL") or getattr(mat, "law_name", None) in ("32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL"):
        _get_law32()
        if law32_shell_tangent is not None:
            return law32_shell_tangent(mat, sig, epsp, epsp_incr, extra)
        raise NotImplementedError("LAW32 consistent_shell_tangent not available in law32_hill")
    if getattr(mat, "law", None) in (43, "43", "LAW43", "HILL_TAB", "LAW43_HILL_TAB") or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "LAW43_HILL_TAB"):
        _get_law43()
        if law43_shell_tangent is not None:
            return law43_shell_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
        raise NotImplementedError("LAW43 consistent_shell_tangent not available in law43_hill_tab")
    if mat.law == 1:
        return np.broadcast_to(law01_elastic.shell_membrane_tangent(mat),
                               (n, 3, 3)).copy()
    if mat.law == 2:
        return law02_johnson_cook.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr)
    if mat.law == 3:
        return law03_plas_bost.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr)
    if mat.law == 36:
        return law36_tabulated.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr)
    if mat.law == 27:
        if extra is None:
            raise NotImplementedError(
                "LAW27 implicit tangent needs the layer crack state "
                "(the shell kernels pass it since M15)")
        return law27_brittle.consistent_shell_tangent(mat, extra)
    if mat.law == 19:
        return law19_fabric.consistent_shell_tangent(mat, extra)
    if mat.law == 44:
        return law44_cowper.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if getattr(mat, "law", None) in (48, "48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO") or getattr(mat, "law_name", None) in ("48", "LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "LAW48_ZHAO"):
        return law48_zhao.tangent_law48_shell(
            mat, sig, epsp, epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (52, "52", "LAW52", "GURSON", "PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS"):
        return law52_gurson.tangent_law52_shell(
            mat, sig, epsp, epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
        return law57_barlat.tangent_law57_shell(
            mat, sig=sig, deps=None, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (58, "58", "LAW58", "FABR_A", "MAT_FABR_A", "FABRIC_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "MAT_FABR_A", "FABRIC_A", "LAW58_FABR_A"):
        return law58_fabr_a.tangent_law58_shell(
            mat, sig=sig, deps=None, epsp=epsp, extra=extra)
    if mat.law == 34 or getattr(mat, "law_name", None) in ("LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"):
        return law34_boltzmann.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if getattr(mat, "law", None) in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS") or getattr(mat, "law_name", None) in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS"):
        return law25_composite.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if getattr(mat, "law", None) in (22, "22", "LAW22", "DAMA", "PLAS_DAMA") or getattr(mat, "law_name", None) in ("22", "LAW22", "DAMA", "PLAS_DAMA"):
        return law22_dama.consistent_shell_tangent(
            mat, sig, epsp=epsp, dt=0.0, extra=extra, epsp_incr=epsp_incr)
    if getattr(mat, "law", None) in _LAW71_KEYS or getattr(mat, "law_name", None) in _LAW71_KEYS:
        return law71_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in _LAW76_KEYS or getattr(mat, "law_name", None) in _LAW76_KEYS:
        return law76_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
        return law73_shell_tangent(
            mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
        return law66_shell_tangent(
            mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000"):
        return law87_shell_tangent(
            mat, sig=sig, deps=None, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
    if getattr(mat, "law", None) in _LAW88_KEYS or getattr(mat, "law_name", None) in _LAW88_KEYS:
        eps_trial = extra.get("eps") if extra else None
        if eps_trial is None:
            if sig is not None:
                eps_trial = np.zeros_like(sig)[:3] if np.ndim(sig) == 1 else np.zeros((np.shape(sig)[0], 3), dtype=np.float64)
            else:
                eps_trial = np.zeros(3, dtype=np.float64)
        return law88_shell_tangent(eps_trial, mat)
    if getattr(mat, "law", None) in _LAW92_KEYS or getattr(mat, "law_name", None) in _LAW92_KEYS:
        eps_trial = extra.get("eps") if extra else None
        if eps_trial is None:
            if sig is not None:
                eps_trial = np.zeros_like(sig)[:3] if np.ndim(sig) == 1 else np.zeros((np.shape(sig)[0], 3), dtype=np.float64)
            else:
                eps_trial = np.zeros(3, dtype=np.float64)
        return law92_consistent_tangent(mat, eps_trial, is_shell=True)
    if getattr(mat, "law", None) in _LAW93_KEYS or getattr(mat, "law_name", None) in _LAW93_KEYS:
        pla_val = epsp if epsp is not None else 0.0
        return law93_consistent_tangent(mat, sig if sig is not None else np.zeros(3), pla=pla_val, plane_stress=True)
    if getattr(mat, "law", None) in _LAW94_KEYS or getattr(mat, "law_name", None) in _LAW94_KEYS:
        eps_trial = extra.get("eps") if extra else None
        if eps_trial is None:
            if sig is not None:
                eps_trial = np.zeros_like(sig)[:3] if np.ndim(sig) == 1 else np.zeros((np.shape(sig)[0], 3), dtype=np.float64)
            else:
                eps_trial = np.zeros(3, dtype=np.float64)
        return law94_consistent_tangent(mat, eps_trial, is_shell=True)
    if getattr(mat, "law", None) in (21, "21", "LAW21", "MAT_LAW21", "MAT_DPRAG") or (getattr(mat, "law_name", None) in ("21", "LAW21", "DPRAG", "MAT_DPRAG") and getattr(mat, "law", None) != 10):
        raise NotImplementedError("LAW21 (Drucker-Prager) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB") or getattr(mat, "law_name", None) in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
        raise NotImplementedError("LAW49 (Steinberg-Guinan) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM") or getattr(mat, "law_name", None) in ("79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
        raise NotImplementedError("LAW79 (Johnson-Holmquist) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW95_KEYS or getattr(mat, "law_name", None) in _LAW95_KEYS:
        raise NotImplementedError("LAW95 (Bergstrom-Boyce) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW100_KEYS or getattr(mat, "law_name", None) in _LAW100_KEYS:
        raise NotImplementedError("LAW100 (Multi-Network Visco-Hyperelastic) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW101_KEYS or getattr(mat, "law_name", None) in _LAW101_KEYS:
        raise NotImplementedError("LAW101 (Bouvard Polymer Viscoplasticity) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW102_KEYS or getattr(mat, "law_name", None) in _LAW102_KEYS:
        raise NotImplementedError("LAW102 (Extended Drucker-Prager) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW103_KEYS or getattr(mat, "law_name", None) in _LAW103_KEYS:
        raise NotImplementedError("LAW103 (Hensel-Spittel Hot Metal Forming) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW104_KEYS or getattr(mat, "law_name", None) in _LAW104_KEYS:
        return law104_consistent_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW106_KEYS or getattr(mat, "law_name", None) in _LAW106_KEYS:
        return law106_consistent_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW107_KEYS or getattr(mat, "law_name", None) in _LAW107_KEYS:
        return law107_consistent_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW109_KEYS or getattr(mat, "law_name", None) in _LAW109_KEYS:
        return law109_consistent_shell_tangent(mat, stress=sig, extra=extra)
    if getattr(mat, "law", None) in _LAW110_KEYS or getattr(mat, "law_name", None) in _LAW110_KEYS:
        return law110_consistent_shell_tangent(mat, stress=sig, extra=extra)
    if getattr(mat, "law", None) in _LAW124_KEYS or getattr(mat, "law_name", None) in _LAW124_KEYS:
        raise NotImplementedError("LAW124 (CDPM2) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW126_KEYS or getattr(mat, "law_name", None) in _LAW126_KEYS:
        raise NotImplementedError("LAW126 (Johnson-Holmquist Concrete / HJC) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW169_KEYS or getattr(mat, "law_name", None) in _LAW169_KEYS:
        return law169_consistent_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW117_KEYS or getattr(mat, "law_name", None) in _LAW117_KEYS:
        return law117_consistent_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in _LAW119_KEYS or getattr(mat, "law_name", None) in _LAW119_KEYS:
        return law119_seatbelt.consistent_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, extra=extra)
    if getattr(mat, "law", None) in _LAW123_KEYS or getattr(mat, "law_name", None) in _LAW123_KEYS or getattr(mat, "law", None) in _LAW132_KEYS or getattr(mat, "law_name", None) in _LAW132_KEYS:
        return law123_consistent_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
    if getattr(mat, "law", None) in (24, "24", "LAW24", "CONC", "CONCRETE", "PERIC_CONC") or getattr(mat, "law_name", None) in (24, "24", "LAW24", "CONC", "CONCRETE", "PERIC_CONC"):
        raise NotImplementedError("LAW24 (concrete) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in (37, "37", "LAW37", "BIPHAS", "BIPHASIC") or getattr(mat, "law_name", None) in ("LAW37", "BIPHAS", "BIPHASIC"):
        raise NotImplementedError("LAW37 (biphasic fluid/gas) is implemented for 3D solid and SPH elements only.")
    if getattr(mat, "law", None) in _LAW90_KEYS or getattr(mat, "law_name", None) in _LAW90_KEYS:
        raise NotImplementedError("LAW90 (tabulated hysteretic foam) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW190_KEYS or getattr(mat, "law_name", None) in _LAW190_KEYS:
        raise NotImplementedError("LAW190 (FOAM_DUBOIS) is implemented for 3D solid elements only.")
    if getattr(mat, "law", None) in _LAW120_KEYS or getattr(mat, "law_name", None) in _LAW120_KEYS:
        c = law120_consistent_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, epsp=epsp, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
        if c.ndim == 2 and n > 1:
            return np.broadcast_to(c, (n, 3, 3)).copy()
        return c
    if getattr(mat, "law", None) in _LAW121_KEYS or getattr(mat, "law_name", None) in _LAW121_KEYS:
        c = law121_consistent_shell_tangent(mat, sig=sig, deps=extra.get("deps") if extra else None, epsp=epsp, dt=extra.get("dt", 0.0) if extra else 0.0, extra=extra)
        if c.ndim == 2 and n > 1:
            return np.broadcast_to(c, (n, 3, 3)).copy()
        return c
    _lnum = _resolve_law_num(mat)
    if _lnum in _NEW_PORTED_LAWS:
        _mod = _NEW_PORTED_LAWS[_lnum]
        _fn = getattr(_mod, "shell_tangent", None) or getattr(_mod, "consistent_shell_tangent", None) or getattr(_mod, "shell_membrane_tangent", None)
        if callable(_fn):
            try:
                res = _fn(mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
            except TypeError:
                try:
                    res = _fn(mat, sig=sig, extra=extra)
                except TypeError:
                    res = _fn(mat)
            if hasattr(res, "ndim") and res.ndim == 2 and n > 1:
                return np.broadcast_to(res, (n, 3, 3)).copy()
            return res
    raise NotImplementedError(
        f"material LAW{mat.law} has no implicit shell tangent (LAW1 "
        f"elastic, LAW2, LAW3, LAW36 and LAW44 elastoplastic, LAW27 brittle cracking, "
        f"LAW19 fabric are ported — see PORTING_GUIDE M15)")


consistent_shell_tangent = shell_layer_tangent
