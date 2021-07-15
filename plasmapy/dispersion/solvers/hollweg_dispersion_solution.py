# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:35:46 2021

@author: sshan
"""
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

# Solves the equation 3 in Bellan2012JGR (equation 38 in hollweg1999)

def hollweg_dispersion_solution(
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
    # Calc needed plasma parameters   
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
    omega_pe = pfp.plasma_frequency(n=n_e, particle="e-")
    
    # Parameters kx and kz
    
    kz = np.cos(theta.value) * k
    kx = np.sqrt(k ** 2 - kz ** 2)
    
    # Bellan2012JGR beta param equation 3
    beta = (c_s / v_A) ** 2
    
    # Parameters D, F, sigma, and alpha to simplify equation 3
    D = (c_s / omega_ci) ** 2
    F = (c / omega_pe) ** 2
    sigma = (kz * v_A) ** 2
    alpha = (k * v_A) ** 2
    
    # Polynomial coefficients: c3*x^3 + c2*x^2 + c1*x + c0 = 0
    c3 = (F * kx ** 2 + 1) / sigma
    c2 = -((alpha / sigma) * (1 + beta + F * kx ** 2) + D * kx ** 2 + 1)
    c1 =  alpha * (1 + 2 * beta + D * kx ** 2)
    c0 = -beta * alpha * sigma
    
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
    "k": .01 * u.rad / u.m,
    "theta": 88 * u.deg,
    "n_i": 5 * u.cm ** -3,
    "B": 2.2e-8 * u.T,
    "T_e": 1.6e6 * u.K,
    "T_i": 4.0e5 * u.K,
    "ion": Particle("p+"),
}



print(hollweg_dispersion_solution(**inputs))

    

