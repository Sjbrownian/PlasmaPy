import numpy as np
import astropy.units as u
import warnings

from astropy.constants.si import c
from plasmapy.formulary import parameters as pfp
from plasmapy.particles import Particle
from plasmapy.particles.exceptions import ChargeError
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import PhysicsWarning

from typing import Union


def alfven_dispersion_solution(
    *,
    B: u.T,
    ion: Union[str, Particle],
    k: u.rad / u.m,
    n_i: u.m ** -3,
    T_e: u.K,
    T_i: u.K,
    theta: u.deg,
    gamma_e: Union[float, int] = 1,
    gamma_i: Union[float, int] = 3,
    z_mean: Union[float, int] = None,
 ):
    r'''
    Notes
    -----
    
    Solves equation 5 in Bellan2012JGR (2x2 matrix method
    argued in Hasegawa and Uberoi 1982, Morales and Maggs 1997,
    and Lysak and Lotko 1996)
    
    ..math::
        \omega^2 = k_{\rm z}^2 v_{\rm A}^2 \left(1 + \frac{k_{\rm x}^2 &
        c_{\rm s}^2}{\omega_{\rm ci}^2} \right)
    
    Examples
    --------
    
    >>> from astropy import units as u
    >>> from plasmapy.dispersion import two_fluid_dispersion
    >>> inputs = {
    ...    "k": np.logspace(-7,-2,2) * u.rad / u.m,
    ...    "theta": 30 * u.deg,
    ...    "B": 8.3e-9 * u.T,
    ...    "n_i": 5 * u.m ** -3,
    ...    "T_e": 1.6e6 * u.K,
    ...    "T_i": 4.0e5 * u.K,
    ...    "ion": Particle("p+"),
    ...}
    >>> omegas = alfven_dispersion_solution(**inputs)
    >>> omegas
    [7.01005647e+00 6.70197761e+08] rad / s
    
    '''
    
    # validate argument ion
    if not isinstance(ion, Particle):
        try:
            ion = Particle(ion)
        except TypeError:
            raise TypeError(
                f"For argument 'ion' expected type {Particle} but got {type(ion)}."
            )
    if not (ion.is_ion or ion.is_category("element")):
        raise ValueError("The particle passed for 'ion' must be an ion or element.")

    # validate z_mean
    if z_mean is None:
        try:
            z_mean = abs(ion.integer_charge)
        except ChargeError:
            z_mean = 1
    else:
        if not isinstance(z_mean, (int, np.integer, float, np.floating)):
            raise TypeError(
                f"Expected int or float for argument 'z_mean', but got {type(z_mean)}."
            )
        z_mean = abs(z_mean)

    # validate arguments
    for arg_name in ("B", "n_i", "T_e", "T_i"):
        val = locals()[arg_name].squeeze()
        if val.shape != ():
            raise ValueError(
                f"Argument '{arg_name}' must a single value and not an array of "
                f"shape {val.shape}."
            )
        locals()[arg_name] = val

    # validate arguments
    for arg_name in ("gamma_e", "gamma_i"):
        if not isinstance(locals()[arg_name], (int, np.integer, float, np.floating)):
            raise TypeError(
                f"Expected int or float for argument '{arg_name}', but got "
                f"{type(locals()[arg_name])}."
            )

    # validate argument k
    k = k.squeeze()
    if not (k.ndim == 0 or k.ndim == 1):
        raise ValueError(
            f"Argument 'k' needs to be a single valued or 1D array astropy Quantity,"
            f" got array of shape {k.shape}."
        )
    if np.any(k <= 0):
        raise ValueError("Argument 'k' can not be a or have negative values.")

    # validate argument theta
    theta = theta.squeeze()
    theta = theta.to(u.radian)
    if not (theta.ndim == 0 or theta.ndim == 1):
        raise ValueError(
            f"Argument 'theta' needs to be a single valued or 1D array astropy "
            f"Quantity, got array of shape {k.shape}."
        ) 
        
    n_e = z_mean * n_i
    c_s = pfp.ion_sound_speed(
        T_e=T_e,
        T_i=T_i,
        ion=ion,
        n_e=n_e,
        gamma_e=gamma_e,
        gamma_i=gamma_i,
        z_mean=z_mean,
        )   
    v_A = pfp.Alfven_speed(B, n_i, ion=ion, z_mean=z_mean)
    omega_ci = pfp.gyrofrequency(B=B, particle=ion, signed=False, Z=z_mean)
    
    
    #Parameters kz
    
    kz = np.cos(theta.value) * k
    kx = np.sqrt(k ** 2 - kz ** 2)
    
    
    #Parameters sigma, D, and F to simplify equation 3
    A = (kz * v_A) ** 2
    F = ((kx * c_s) / omega_ci ) ** 2
    
    omega = np.sqrt(A * (1 + F))
#    print(omega_ci)

    v_Te = pfp.thermal_speed(T=T_e, particle='e-')
    v_Ti = pfp.thermal_speed(T=T_i, particle=ion)
    
    w_max = np.max(omega)
    
    if w_max / omega_ci > .01:
        warnings.warn(
            f"The calculation produced a high-frequency wave, "
            f"which violates the low frequency assumption w << w_ci",
            PhysicsWarning,
            )
    
    
    return omega

inputs = {
"k": np.logspace(-2, -7, 2) * u.rad / u.m,
"theta": 30 * u.deg,
"B": 8.3e-9 * u.T,
"n_i": 5 * u.m ** -3,
"T_e": 1.6e6 * u.K,
"T_i": 4.0e5 * u.K,
"ion": Particle("p+"),
}



print(alfven_dispersion_solution(**inputs))