from typing import List, Union
from numpy.typing import ArrayLike
import numpy as np
from scipy.signal import correlate
from tqdm import tqdm



def chirp(t, f, k_c):
    b_c = k_c/2*t # FIXME whys divide by 2?
    signal = np.cos(2*np.pi*t*(f + b_c)) + 1j*np.sin(2*np.pi*t*(f + b_c))
    return signal

def energy_spec(signal):
    samples = len(signal)
    F = np.fft.fftshift(np.fft.fft(signal))
    F_mag = abs(F)**2
    E = F_mag/samples**2
    return E

def pulse_pairs(pulse, T_pulse: Union[int, float], T_interpulse: Union[int, float] = 0, n_pulses: int = 1):
    # TODO add assert that interpulse delay matches n_pulses - 1
    samples = len(pulse)
    sample_rate = T_pulse/samples

    # NOTE maybe T_interpulse is one index too long becase it takes a grid cell before and after as well
    samples_interpulse = int(np.ceil(T_interpulse/sample_rate))
    interpulse = np.zeros(samples_interpulse)

    signal = np.array(list(pulse) + (n_pulses -1) * (list(interpulse) + list(pulse)))
    T = np.linspace(0, (len(signal) -1 ) * sample_rate, len(signal))

    return signal, T, samples_interpulse

def random_receive(n_reflectors: int, return_window: int, signal: np.array, seed: int = 42) -> List[np.array]:
    # instantiate empty receive signal
    receive = np.zeros(return_window).astype(np.complex128) # NOTE instantiated as zeros as empty -> rounding errors

    np.random.seed(seed)
    # iteratively add random signals
    # TODO vectorize --> NOTE because the size of the matrixes involved, vectorization may lead to overload
    for i in tqdm(range(n_reflectors)):

        # reflection has random time delay and amplitude
        delta = np.random.randint(low = 0, high = return_window)
        A = abs(np.random.rand(1)[0]) + 0.01
        scatterer_phase = np.exp(1j*np.random.rand(1)*2*np.pi)

        # for given signals, apply delay and amplitude and combine for total received reflection
        reflection = A * signal * scatterer_phase
        
        # add to total received signal considering time delay
        reflection_length = len(receive[delta:delta +len(reflection)])
        receive[delta:delta+reflection_length] += reflection[:reflection_length]

    return receive

def delayed_correlation(signal_received, signal_length, inter_signal_delay):

    step_size = signal_length + inter_signal_delay
    corr_signals = []
    corr_peak = []
    corr_max_values = []

    # Calculate delayed correlation
    for i in tqdm(range(0, len(signal_received) - signal_length + 1, step_size)):
        # Get the current sample range
        sample_range = signal_received[i : i+signal_length]
        
        # Perform correlation with the next sample range for each step

        # NOTE maybe this should be a correlation with conjugate signal, as correlation weights edges???
        if i + step_size + signal_length <= len(signal_received):
            next_sample_range = signal_received[i+step_size : i+step_size+signal_length]

            # compute correlation between first sample range and next
            corr_result = correlate(sample_range, next_sample_range, mode = "same", method = "fft")
            corr_max = np.argmax(abs(corr_result)) # NOTE abs to find full correlation
            
            # store results
            corr_signals.append(corr_result)
            corr_peak.append(corr_max)
            corr_max_values.append([sample_range[corr_max], next_sample_range[corr_max]])

    corr_signals = np.array(corr_signals)
    corr_peak = np.array(corr_peak)
    corr_max_values = np.array(corr_max_values)

    return corr_signals, corr_peak, corr_max_values