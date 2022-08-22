"""
This module contains functionality for calculating various numerical
solutions to Hirose's two fluid dispersion relation, see
:cite:t:`hirose:2004` and :cite:t:`bellan:2012`.
"""

import astropy.units as u
import numpy as np
import warnings

from astropy.constants.si import c
from typing import Optional, Union

from plasmapy.formulary.frequencies import gyrofrequency, plasma_frequency
from plasmapy.formulary.speeds import Alfven_speed, ion_sound_speed
from plasmapy.particles import Particle
from plasmapy.particles.exceptions import ChargeError
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import PhysicsWarning

c_si_unitless = c.value


@validate_quantities(
    B={"can_be_negative": False},
    n_i={"can_be_negative": False},
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
def hirose(
    *,
    B: u.T,
    ion: Union[str, Particle],
    k: u.rad / u.m,
    n_i: u.m**-3,
    T_e: u.K,
    theta: u.rad,
    gamma_e: Optional[Union[float, int]] = 1,
    gamma_i: Optional[Union[float, int]] = 3,
    z_mean: Optional[Union[float, int]] = None,
    **kwargs,
):

    r"""
    Calculate the two fluid dispersion relation presented by
    :cite:t:`hirose:2004`, and discussed by :cite:t:`bellan:2012`.
    This is a numerical solver of equation 7 in :cite:t:`bellan:2012`.
    See the **Notes** section below for additional details.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to T.
    ion : `str` or `~plasmapy.particles.particle_class.Particle`
        Representation of the ion species (e.g., ``"p"`` for protons,
        ``"D+"`` for deuterium, ``"He-4 +1"`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized.
    k : `~astropy.units.Quantity`, single valued or 1-D array
        Wavenumber in units convertible to rad/m.  Either single
        valued or 1-D array of length :math:`N`.
    n_i : `~astropy.units.Quantity`
        Ion number density in units convertible to m\ :sup:`-3`.
    T_e : `~astropy.units.Quantity`
        The electron temperature in units of K or eV.
    theta : `~astropy.units.Quantity`, single valued or 1-D array
        The angle of propagation of the wave with respect to the
        magnetic field, :math:`\cos^{-1}(k_z / k)`, in units convertible
        to radians.  Either single valued or 1-D array of size
        :math:`M`.
    gamma_e : `float` or `int`, optional
        The adiabatic index for electrons, which defaults to 1.  This
        value assumes that the electrons are able to equalize their
        temperature rapidly enough that the electrons are effectively
        isothermal.  (DEFAULT ``1``)
    gamma_i : `float` or `int`, optional
        The adiabatic index for ions, which defaults to 3. This value
        assumes that ion motion has only one degree of freedom, namely
        along magnetic field lines.  (DEFAULT ``3``)
    z_mean : `float` or int, optional
        The average ionization state (arithmetic mean) of the ``ion``
        composing the plasma.  Will override any charge state defined
        by argument ``ion``.  (DEFAULT `None`)

    Returns
    -------
    omega : Dict[`str`, `~astropy.units.Quantity`]
        A dictionary of computed wave frequencies in units rad/s.  The
        dictionary contains three keys: ``'fast_mode'`` for the fast
        mode, ``'alfven_mode'`` for the Alfvén mode, and
        ``'acoustic_mode'`` for the ion-acoustic mode.  The value for
        each key will be a :math:`N x M` array.

    Raises
    ------
    TypeError
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    TypeError
        If ``ion`` is not of type or convertible to
        `~plasmapy.particles.particle_class.Particle`.

    TypeError
        If ``gamma_e``, ``gamma_i``, or ``z_mean`` are not of type `int`
        or `float`.

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    ValueError
        If any of ``B``, ``k``, ``n_i``, or ``T_e`` is negative.

    ValueError
        If ``k`` is negative or zero.

    ValueError
        If ``ion`` is not of category ion or element.

    ValueError
        If ``B``, ``n_i``, or ``T_e`` are not single valued
        `astropy.units.Quantity`.

    ValueError
        If ``k`` or ``theta`` are not single valued or a 1-D array.

    Warns
    -----
    : `~plasmapy.utils.exceptions.PhysicsWarning`
        When :math:`\omega / \omega_{\rm ci} > 0.1`, violation of the
        low-frequency assumption.

    Notes
    -----
    The dispersion relation presented in :cite:t:`hirose:2004`
    (equation 7 in :cite:t:`bellan:2012`) is:

    .. math::
        \left(\omega^2 - k_{\rm z}^2 v_{\rm A}^2 \right)
        \left(\omega^4 - \omega^2 k^2 \left(c_{\rm s}^2 + v_{\rm A}^2 \right)
        + k^2 v_{\rm A}^2 k_{\rm z}^2 c_{\rm s}^2 \right)
        = \frac{k^2 c^2}{\omega_{\rm pi}^2} \omega^2 v_{\rm A}^2 k_{\rm z}^2
        \left(\omega^2 - k^2 c_{\rm s}^2 \right)

    where

    .. math::
        \mathbf{B_o} &= B_{o} \mathbf{\hat{z}} \\
        \cos \theta &= \frac{k_z}{k} \\
        \mathbf{k} &= k_{\rm x} \hat{x} + k_{\rm z} \hat{z}

    :math:`\omega` is the wave frequency, :math:`k` is the wavenumber,
    :math:`v_{\rm A}` is the Alfvén velocity, :math:`c_{\rm s}` is the
    sound speed, :math:`\omega_{\rm ci}` is the ion gyrofrequency, and
    :math:`\omega_{\rm pi}` is the ion plasma frequency. In the
    derivation of this relation Hirose assumed low-frequency waves
    :math:`\omega / \omega_{\rm ci} \ll 1`, no D.C. electric field
    :math:`\mathbf{E_o}=0`, and cold ions :math:`T_{i}=0`.

    This routine solves for ω for given :math:`k` values by numerically
    solving for the roots of the above expression.

    Examples
    --------
    >>> from astropy import units as u
    >>> from plasmapy.dispersion.numerical import hirose
    >>> inputs = {
    ...    "k": np.logspace(-7,-2,2) * u.rad / u.m,
    ...    "theta": 30 * u.deg,
    ...    "B": 8.3e-9 * u.T,
    ...    "n_i": 5 * u.m ** -3,
    ...    "T_e": 1.6e6 * u.K,
    ...    "ion": Particle("p+"),
    ... }
    >>> omegas = hirose(**inputs)
    >>> omegas
    {'fast_mode': <Quantity [7.21782095e+01+0.j, 7.13838935e+11+0.j] rad / s>,
    'alfven_mode': <Quantity [7.86100475e-01+0.j, 1.14921892e+03+0.j] rad / s>,
    'acoustic_mode': <Quantity [0.00995226+0.j, 0.68834011+0.j] rad / s>}
    """

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
            z_mean = abs(ion.charge_number)
        except ChargeError:
            ion.ionize(n=1, inplace=True)
            z_mean = abs(ion.charge_number)
    else:
        if not isinstance(z_mean, (int, np.integer, float, np.floating)):
            raise TypeError(
                f"Expected int or float for argument 'z_mean', but got {type(z_mean)}."
            )
        z_mean = abs(z_mean)

    # invalidate T_i argument
    if "T_i" in kwargs["kwargs"]:
        raise TypeError(
            "Got unexpected keyword 'T_i', dispersion relation assumes T_i = 0."
        )

    # validate arguments
    for arg_name in ("B", "n_i", "T_e"):
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
    if k.ndim not in (0, 1):
        raise ValueError(
            f"Argument 'k' needs to be a single valued or 1D array astropy Quantity,"
            f" got array of shape {k.shape}."
        )
    if np.any(k <= 0):
        raise ValueError("Argument 'k' can not be 0 or have negative values.")

    # validate argument theta
    theta = theta.squeeze()
    if theta.ndim not in (0, 1):
        raise ValueError(
            f"Argument 'theta' needs to be a single valued or 1D array astropy "
            f"Quantity, got array of shape {theta.shape}."
        )

    # Single k value case
    if np.isscalar(k.value):
        k = np.array([k.value]) * u.rad / u.m

    # Calc needed plasma parameters
    n_e = z_mean * n_i
    v_A = Alfven_speed(B, n_i, ion=ion, z_mean=z_mean).value
    omega_ci = gyrofrequency(B=B, particle=ion, signed=False, Z=z_mean).value
    omega_pi = plasma_frequency(n=n_i, particle=ion, z_mean=z_mean).value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PhysicsWarning)
        c_s = ion_sound_speed(
            T_e=T_e,
            T_i=0 * u.K,
            ion=ion,
            n_e=n_e,
            gamma_e=gamma_e,
            gamma_i=gamma_i,
            z_mean=z_mean,
        ).value

    thetav, kv = np.meshgrid(theta.value, k.value)

    # Parameter kz
    kz = np.cos(thetav) * kv

    # Define helpful parameters
    A = (kz * v_A) ** 2
    B = (kv * c_s) ** 2
    C = (kv * v_A) ** 2
    D = ((kv * c_si_unitless) / omega_pi) ** 2

    # Polynomial coefficients: c3*x^6 + c2*x^4 + c1*x^2 + c0
    c3 = np.ones_like(A)
    c2 = -A * (1 + D) - B - C
    c1 = A * (2 * B + C + B * D)
    c0 = -B * A**2

    # Find roots of polynomial
    coefficients = np.array([c3, c2, c1, c0], ndmin=3)
    nroots = coefficients.shape[0] - 1  # 3
    nks = coefficients.shape[1]
    nthetas = coefficients.shape[2]
    roots = np.empty((nroots, nks, nthetas), dtype=np.complex128)
    for ii in range(nks):
        for jj in range(nthetas):
            roots[:, ii, jj] = np.roots(coefficients[:, ii, jj])

    roots = np.sqrt(roots)
    roots = np.sort(roots, axis=0)

    # dispersion relation is only valid in the regime w << w_ci
    w_max = np.max(roots)
    w_wci_max = w_max / omega_ci
    if w_wci_max > 0.1:
        warnings.warn(
            f"This solver is valid in the regime w/w_ci << 1.  A w "
            f"value of {w_max:.2f} and a w/w_ci value of "
            f"{w_wci_max:.2f} were calculated which may affect the "
            f"validity of the solution.",
            PhysicsWarning,
        )

    return {
        "fast_mode": roots[2, :].squeeze() * u.rad / u.s,
        "alfven_mode": roots[1, :].squeeze() * u.rad / u.s,
        "acoustic_mode": roots[0, :].squeeze() * u.rad / u.s,
    }