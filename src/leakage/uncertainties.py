import random
import types
import numpy as np
from typing import Union


def decorrelation(tau, T):
    return np.exp(-((tau / T) ** 2))  #


def pulse_pair_coherence(T_pp, T_corr_surface, T_corr_Doppler, SNR):
    """
    Calculates the Pulse pair velocity standard deviation within a resolution cell due to coherence loss

    NOTE assumes broadside geometry (non-squinted)

    Parameters
    ----------
    T_pp : scaler
        Intra pulse pair time separation
    T_corr_surface : scaler
        Decorrelation time of ocean surface at scales of radio wavelength of interest
    T_corr_Doppler : scaler
        Decorrelation time of velocities within resolution cell as a result of satellite motion during pulse-pair transmit
    SNR : scaler
        Signal to Noise ratio (for pulse-pair system we assume signal to clutter ratio of 1 dominates)

    Returns
    -------
    Scaler of coherence

    """

    gamma_velocity = decorrelation(
        T_pp, T_corr_Doppler
    )  # eq 6 & 7 Rodriguez (2018), NOTE not valid for squint NOTE assumes Gaussian beam pattern
    gamma_temporal = decorrelation(T_pp, T_corr_surface)
    gamma_SNR = SNR / (1 + SNR)

    gamma = gamma_temporal * gamma_SNR * gamma_velocity

    return gamma


def phase_uncertainty_rodriguez2018(gamma, N_L=1):
    """
    Calculates the pulse pair phase variance within a resolution cell due to coherence loss
    equation 14 of Rodriguez et al (2018) Estimating Ocean Vector Winds and Currents Using a Ka-Band Pencil-Beam Doppler Scatterometer

    NOTE valid in the high-coherence limit only

    Parameters
    ----------
    N_L : int
        Number of independent looks for given area
    gamma : scaler
        coherence

    Returns
    -------
    Scaler of estimates surface velocity variance
    """

    phase_var = 1 / (2 * N_L) * (1 - gamma**2) / gamma**2

    return phase_var


def generate_complex_speckle(noise_shape: tuple, random_state: int = 42):
    """
    Generates complex multiplicative speckle noise

    Assumes circular Gaussian distribution
    """
    np.random.seed(random_state)
    noise_real = np.random.randn(*noise_shape)
    noise_imag = np.random.randn(*noise_shape)
    speckle = np.array(
        [complex(a, b) for a, b in zip(noise_real.ravel(), noise_imag.ravel())]
    )

    return speckle.reshape(noise_shape)


def speckle_intensity(complex_speckle):
    """Intensity is obtained by (abs(speckle_complex)**2)/2, which has a mean and variance of 1"""
    return abs(complex_speckle) ** 2 / 2


def phase_error_generator(
    gamma,
    n_samples: Union[int, tuple],
    theta: float = 0,
    n_bins: int = 20001,
    random_state: Union[float, int, types.NoneType] = None,
):
    """
    generates samples from the 1-Look phase-difference probability density function (pdf) from equation 19 in:
    Jong-Sen Lee et al., (1994) "Statistics of phase difference and product magnitude of multi-look processed Gaussian signals"

    Input
    -----
    gamma: float,
        coherence of phase difference
    n_samples: Union[int, tuple],
        Number of samples to generate given as a number or as a shape within a tuple, e.g. 1000 or (20, 7)
    theta: float,
        phase offset, in radian
    n_bins: int,
        Number of discrete bins to generate pdf
    random_state: Union[int, float, types.NoneType],
        Fixes random state if float or int is provided

    Returns
    -------
    samples:
        Array with a phase error realisation of phase uncertainty
    """

    if type(random_state) is not type(None):
        random.seed(random_state)

    if type(n_samples) == tuple:
        target_shape = n_samples
        N = np.prod(target_shape)
    else:
        N = n_samples

    psi = np.linspace(-np.pi, np.pi, n_bins)
    beta = gamma * np.cos(psi - theta)
    pdf = (
        (1 - gamma**2) * (np.sqrt(1 - beta**2) + beta * (np.pi - (np.arccos(beta))))
    ) / (2 * np.pi * (1 - beta**2) ** (1.5))
    samples = np.array(random.choices(population=psi, weights=pdf, k=N))

    if type(n_samples) == tuple:
        samples = samples.reshape(target_shape)

    return samples
