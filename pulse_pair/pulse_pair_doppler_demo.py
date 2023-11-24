#%%
from typing import List, Union

import numpy as np
from matplotlib import pyplot as plt

from scipy.signal import correlate

#%%

# TODO add motion to point scatterers

T0 = 0e-3 # s
Te = 1e-3 # s
B_c = 1e6 # hz
f =  5.405e9 # hz
over_sample = 1 # -
receive_time = 30e-3 # Te*10
n_reflectors = 10000 #

T_pulse=Te-T0 # s
T_interpulse=0 # s (Te-T0)/4
n_pulses=2 # -

k_c = B_c/Te # hz/s
fs_nyq = 2*(f + 2*B_c) # hz, times 2 to follow Nyquist
fs = fs_nyq * over_sample # hz
s = 1/fs # stepsize
t = np.arange(T0, (Te + s), s) # s
receive_samples = int(receive_time*fs )

#%%

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

def random_receive(n_reflectors: int, return_window: int, signal: np.array) -> List[np.array]:
    # instantiate empty receive signal
    receive = np.zeros(return_window).astype(np.complex128) # 

    # iteratively add random signals
    # TODO vectorize --> because the size of the amtrixes involved, vectorization may lead to overload
    for i in tqdm(range(n_reflectors)):

        # reflection has random time delay and amplitude
        delta = np.random.randint(low = 0, high = return_window)
        A = abs(np.random.rand(1)[0]) + 0.01

        # for given signals, apply delay and amplitude and combine for total received reflection
        reflection = A * signal
        
        # add to total received signal considering time delay
        reflection_length = len(receive[delta:delta +len(reflection)])
        receive[delta:delta+reflection_length] += reflection[:reflection_length]

    return receive

def delayed_correlation(signal_received, signal_length, inter_signal_delay):

    step_size = signal_length + inter_signal_delay
    corr_signals = []
    corr_peak = []

    # Calculate delayed correlation
    for i in tqdm(range(0, len(signal_received) - signal_length + 1, step_size)):
        # Get the current sample range
        sample_range = signal_received[i : i+signal_length]
        
        # Perform correlation with the next sample range for each step

        # NOTE maybe this should be a correlation with conjugate signal, as correlation weights edges???
        if i + step_size + signal_length < len(signal_received):
            next_sample_range = signal_received[i+step_size : i+step_size+signal_length]

            corr_result = correlate(sample_range, next_sample_range, mode = "full", method = "fft")

            # store results
            corr_signals.append(corr_result)
            corr_peak.append(np.argmax(corr_result))

    corr_signals = np.array(corr_signals)
    corr_peak = np.array(corr_peak)

    return corr_signals, corr_peak

#%%

# generate a single pulse for the pulse pair
pulse = chirp(t, f, k_c) 

# combine multiple pulses into a pulse pair
signal, t_signal, samples_interpulse = pulse_pairs(
    pulse=pulse, 
    T_pulse=T_pulse, 
    T_interpulse=T_interpulse, 
    n_pulses=n_pulses)

phase = np.arctan2(signal.imag, signal.real)

# calculate the reflection of pulse pair from random arrangement of point scatterers
reflections = random_receive(
    n_reflectors = n_reflectors,
    return_window = receive_samples,
    signal = signal
    )
reflections = np.ravel(reflections)

# compute pulse compressed return
taper = np.hamming(len(pulse))

pulse_compress = correlate(reflections, taper*pulse, mode = "same", method = "fft")
# pulse_compress = np.correlate(reflections, taper*pulse, "same")

# calculate delayed cross correlation 
corr_pulse_compress, corr_peak_pulse_compress =  delayed_correlation(
    signal_received=pulse_compress, 
    signal_length=len(pulse), 
    inter_signal_delay=samples_interpulse)

# organize data to show in box plot
data = corr_peak_pulse_compress * 1/fs


#%%

# plt.figure()
# plt.title('Time domain real component signal')
# plt.plot(signal.real)

# plt.figure()
# plt.title('Phase [rad]')
# plt.plot(t_signal, phase)

# plt.figure()
# plt.title('Energy spectrum')
# plt.plot(energy_spec(signal))

# plt.figure()
# plt.title('Received signal amplitude')
# plt.plot(reflection_amp) # NOTE is intensity sqrt of imag**2 + real**2 ?

# plt.figure()
# plt.title('Received signal phase')
# plt.plot(reflection_phase)

plt.figure()
plt.title('Matched filter on received signal')
plt.plot(pulse_compress)
plt.xlabel("range samples"); plt.ylabel("Matched filter magnitude")

# NOTE in the figure below we alreadyremove the delay caused by the interpulse time
plt.figure()
plt.boxplot(data,
            vert=True,  # vertical box alignment
            patch_artist=True,  # fill with color
            labels = ['Complex']); 
            # labels = ['Complex', 'Phase', 'Amplitude']); 
# plt.ylim([T_pulse*0.8, T_pulse*1.2])
plt.ylabel(f"Interpulse correlation delay (input $\Delta T$: {T_pulse})")


# %%


#%%

import numpy as np
import dask.array as da
from dask import delayed
import dask

# Define the parameters
n = 10000  # Length of the array y (increased value)
L = 100   # Number of signals to add
m = 1000  # Length of each signal (large value)

# Determine an appropriate chunk size based on memory availability
chunk_size = 500  # Adjust this value as needed

# Create a Dask array of zeros with dtype complex128 and specified chunks
y = da.zeros(n, dtype=np.complex128, chunks=(chunk_size,))

# Define a delayed function for adding a single signal
@delayed
def add_signal(m):
    A = 1

    signal = np.random.rand(m) + 1j * np.random.rand(m)  # Create a complex signal
    index = np.random.randint(low = 0, high = n)
    receive = A * signal

    reflection_length = len(y[index:index + len(receive)])
    y[index:index+reflection_length] += receive[:reflection_length]


# Create a list of delayed computations for adding signals
delayed_computations = [add_signal(m) for _ in range(L)]

# Compute the delayed computations to add signals to y
dask.compute(*delayed_computations)

# Compute the final result as a NumPy array
final_result = y.compute()

print(final_result)




# %%
