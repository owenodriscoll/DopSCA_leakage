import numpy as np


def azimuth_beamwidth(Lambda, antenna_length):
    """
    Below from text just below Eq. 4.27 in IG. Cumming, FH. Wong (2005)


    input
    -----
    Lambda: float/int, radiowave wavelength, m
    antenna_length: float/int, length of the antenna in azimuth, m

    returns
    -------
    azimuth_beamwidth: float, ?, ? #FIXME
    """
    return 0.886*Lambda / antenna_length


def beam_pattern_oneway(theta, azimuth_beamwidth):
    """
    Eq. 4.27 from: IG. Cumming, FH. Wong (2005)


    input
    -----
    theta: float/int/array, angle from boresight, rad
    azimuth_beamwidth: 

    returns
    -------
    one-way beam pattern: float, int, array
    """
    
    return np.sinc(0.886*theta/azimuth_beamwidth)


if __name__ == "main":
    import matplotlib.pyplot as plt

    Lambda = 5.6e-2
    antenna_length = 10
    theta = np.linspace(-np.pi/6, np.pi/6)

    azimuth_beamwidth_out = azimuth_beamwidth(Lambda, antenna_length)
    beam_pattern_out = beam_pattern_oneway(theta, azimuth_beamwidth_out)
    print(1)

    plt.figure()
    plt.plot(theta, beam_pattern_out)
    plt.show()
