import numpy as np
import astropy.units as u

from astropy.constants.si import c
from plasmapy.formulary import parameters as pfp
from plasmapy.particles import Particle
from plasmapy.particles.exceptions import ChargeError
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import PhysicsWarning

from typing import Union



def hirose_dispersion_solution(
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
    
    Solves equation 7 in Bellan2012JGR (originally from Hirose2004)
    
    ..math::
        \left(\omega^2 - k_{\rm z}^2 v_{\rm A}^2 \right) &
        \left(\omega^4 - \omega^2 k^2 \left(c_{\rm s}^2 + v_{\rm A}^2 \right) &
        + k^2 v_{\rm A}^2 k_{\rm z}^2 c_{\rm s}^2 \right) & 
        \frac{k^2 c^2}{\omega_{\rm pi}^2} \omega^2 v_{\rm A}^2 k_{\rm z}^2 &
        \left(\omega^2 - k^2 c_{\rm s}^2 \right)
    
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
    ...    "T_i": 1e-4 * u.K,
    ...    "ion": Particle("p+"),
    ...}
    >>> omegas = hirose_dispersion_solution(**inputs)
    >>> omegas
    {'fast_mode': <Quantity [7.12646288e+01, 7.13838935e+11] rad / s>,
     'alfven_mode': <Quantity [7.96179537e-01, 1.14921892e+03] rad / s>,
     'acoustic_mode': <Quantity [0.00995224, 0.68834011] rad / s>}
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
    omega_pi = pfp.plasma_frequency(n=n_i, particle=ion)
    
    
    #Parameters kz
    
    kz = np.cos(theta.value) * k
    
    
    #Parameters sigma, D, and F to simplify equation 3
    A = (kz * v_A) ** 2
    B = (k * c_s) ** 2
    C = (k * v_A) ** 2
    D = ((k * c) / omega_pi ) ** 2
    
    #Polynomial coefficients where x in 'cx' represents the order of the term
    
    #c3 must be an astropy.units.quantiy.Quantity type.
    #Typing "1" doesn't work since it's an int, "A ** 0" gives astropy units.
    #May be simpler way to change to proper type.
    
    c3 = A ** 0
    c2 = -A * (1 + D) + B  + C
    c1 = A * (2 * B + C + B * D)
    c0 = -B * A ** 2
    
    omega = {}
    fast_mode = []
    alfven_mode = []
    acoustic_mode = []
    
    # If a single k value is given
    if np.isscalar(k.value) == True:
        
        w = np.emath.sqrt(np.roots([c3.value, c2.value, c1.value, c0.value]))
        fast_mode = np.max(w)
        alfven_mode = np.median(w)
        acoustic_mode = np.min(w)
        
    # If mutliple k values are given
    else:
        # a0*x^3 + a1*x^2 + a2*x^3 + a3 = 0
        for (a0,a1,a2,a3) in zip(c3, c2, c1, c0):
    
            w = np.emath.sqrt(np.roots([a0.value, a1.value, a2.value, a3.value]))
            fast_mode.append(np.max(w))
            alfven_mode.append(np.median(w))
            acoustic_mode.append(np.min(w)) 

    omega['fast_mode'] = fast_mode * u.rad / u.s
    omega['alfven_mode'] = alfven_mode * u.rad / u.s
    omega['acoustic_mode'] = acoustic_mode * u.rad / u.s
    
    return omega

inputs = {
"k": np.logspace(-7,-2,2) * u.rad / u.m,
"theta": 30 * u.deg,
"B": 8.3e-9 * u.T,
"n_i": 5 * u.m ** -3,
"T_e": 1.6e6 * u.K,
"T_i": 1e-4 * u.K,
"ion": Particle("p+"),
}



omegas = hirose_dispersion_solution(**inputs)
print(type(omegas['fast_mode']))
print(omegas)

