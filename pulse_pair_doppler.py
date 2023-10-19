from numpy.typing import ArrayLike
import numpy as np
from scipy.signal import correlate, fftconvolve
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, Union, List, Dict, Any
from stereoid.oceans.GMF.cmod5n import cmod5n_forward

c = 3e8 # m/s

@dataclass
class pulse_pair_doppler:
    t_pulse: Union[int, float]
    t_receive: Union[int, float]
    t_interpulse: Union[int, float]
    n_pulses: int
    n_bursts: int
    bandwidth: Union[int, float]
    baseband: Union[int, float]
    seed: Union[int, float]
    n_reflectors: int = None
    oversample_retriev: Union[int, float] = 1
    range_cell_avg_factor: Union[int, float] = 2 # downsample factor after cross correlation
    range_cell_size_frac_pulse: Union[int, float] = 1
    Lambda = 0.05 # wavelength
    R = 700e3 # satellite elevation
    theta_min = 27 #minimum incidence angle, deg


    def __post_init__(self):
        self.k_c = self.bandwidth/self.t_pulse # hz/s
        fs_nyq = 2 * (self.baseband + self.bandwidth)  # hz, times 2 to follow Nyquist
        fs = fs_nyq * self.oversample_retriev # hz
        s = 1/fs # stepsize
        self.pulse_samples = int(np.round(self.t_pulse/s))
        self.t_pulse_vector = np.linspace(0, self.t_pulse, self.pulse_samples) # s
        self.receive_samples = int(np.round(self.t_receive*fs ))

        if self.seed == None: self.seed = 42

    def chirp(self, centre_around_baseband: bool = True):
        """

        """
        b_c = self.k_c / 2 * self.t_pulse_vector # FIXME whys divide by 2?
        self.pulse = np.cos(2*np.pi*self.t_pulse_vector*(self.baseband + b_c)) + 1j*np.sin(2*np.pi*self.t_pulse_vector*(self.baseband + b_c))

        # set bandwidth to centre around baseband
        if centre_around_baseband:
            freq_shift = -self.bandwidth/2
            self.pulse = self.pulse * np.exp(1j*2*np.pi*freq_shift * self.t_pulse_vector)

        return 

    def pulse_pairs(self):
        """
        
        """
        # TODO add assert that interpulse delay matches n_pulses - 1
        sample_rate = self.t_pulse/self.pulse_samples

        # NOTE maybe T_interpulse is one index too long becase it takes a grid cell before and after as well
        self.samples_interpulse = int(np.ceil(self.t_interpulse/sample_rate))
        interpulse = np.zeros(self.samples_interpulse)
        self.signal = np.array(list(self.pulse) + (self.n_pulses -1) * (list(interpulse) + list(self.pulse)))

        return 
    

    def _simulate_reflection(self, seed: Union[int,float, bool] = True, progress_bar_dissable = False):
        """

        """
        # fix randomness for reproducibility
        if seed == True:
            np.random.seed(self.seed)
        elif type(seed) != bool:
            np.random.seed(seed)
        
        # instantiate empty receive signal
        reflections = np.zeros(self.receive_samples).astype(np.complex128)

        # iteratively add random signals
        # TODO vectorize --> NOTE because the size of the matrixes involved, vectorization may lead to overload
        for i in tqdm(range(self.n_reflectors), desc="Reflector", disable=progress_bar_dissable):

            # reflection has random time delay and amplitude
            delta = np.random.randint(low = 0, high = self.receive_samples)
            A = np.random.rayleigh(2, 1)[0] 
            scatterer_phase = np.exp(1j*np.random.rand(1)*2*np.pi)

            # for given signals, apply delay and amplitude and combine for total received reflection
            reflection = A * self.signal * scatterer_phase
            
            # add to total received signal considering time delay
            reflection_length = len(reflections[delta:delta +len(reflection)])
            reflections[delta:delta+reflection_length] += reflection[:reflection_length]

        self.reflections = reflections

        return 
    
    # NOTE this does not take into account varying grid resolution as function of incidence angle
    def simulate_reflection(self, phi_avg: Union[int,float] = 90, v_avg: Union[int,float] = 6, seed: Union[int,float,bool] = True):
        """
        input
        -----
        phi_avg:
        v_avg:
        seed: 

        """
        # fix randomness for reproducibility
        if seed == True:
            np.random.seed(self.seed)
        elif type(seed) != bool:
            np.random.seed(seed)

        # compute change in backscatter from CMOD
        R1 = self.R / np.cos(np.deg2rad(self.theta_min))
        R2 = self.t_receive * c + R1
        R_n = np.linspace(R1, R2, self.receive_samples)
        theta = np.rad2deg(np.arccos(self.R/R_n))

        phi = np.ones(self.receive_samples) * phi_avg
        v = np.ones(self.receive_samples) * v_avg
        surface_amplitude_slope = cmod5n_forward(v=v, phi=phi, theta=theta)
        surface_amplitude_slope /= np.max(surface_amplitude_slope)
        surface_amplitude = np.random.rayleigh(2, self.receive_samples)
        # surface_amplitude = abs(np.random.randn(self.receive_samples))

        # generate uniform distribution of phase
        surface_phase = np.exp(1j*2*np.pi*np.random.rand(self.receive_samples))

        # filter to n_reflectors
        if self.n_reflectors is not None:
             
            n_reflector_filter = np.zeros(self.receive_samples)

            # Generate n unique random indices within the range
            random_indices = np.random.choice(self.receive_samples, self.n_reflectors, replace=False)

            # Set the selected indices to 1
            n_reflector_filter[random_indices] = 1
        else:
            n_reflector_filter = 1

        self.surface = surface_amplitude * surface_amplitude_slope * surface_phase * n_reflector_filter
        # self.surface = 1 * 1 * surface_phase * n_reflector_filter

        self.reflections = fftconvolve(self.surface, self.signal, mode = 'valid')

        return 
    
    def pulse_compress(self, window_function: bool = False):
        """
        
        """
        if window_function:
            taper = np.hamming(self.pulse_samples)
        else:
            taper = 1
        
        self.pulse_compressed = correlate(self.reflections, taper*self.pulse, mode = "valid", method = "fft")

        return 
    

    def delayed_autocorrelation(self, progress_bar_dissable = False):
        """
        
        """

        window_size = int(self.pulse_samples // (1 / self.range_cell_size_frac_pulse)) 
        step_size = self.pulse_samples + self.samples_interpulse
        
        corr = []

        # Calculate delayed correlation
        for i in tqdm(range(0, len(self.pulse_compressed) - 2 * window_size + 1, window_size), desc="Autocorrelation cell", disable=progress_bar_dissable):
            # Get the current sample range
            sample_range = self.pulse_compressed[i : i+window_size]
            
            # Perform correlation with the next sample range for each step
            if i + step_size + window_size <= len(self.pulse_compressed):

                # select next same range
                next_sample_range = self.pulse_compressed[i+step_size : i + step_size + window_size]

                # compute correlation between first sample range and next
                corr_result = correlate(sample_range, next_sample_range, mode = "full", method = "fft")
                
                # store results
                corr.append(corr_result)

        self.corr = np.array(corr)

        return 
    
    def phase_difference(self):
        """

        """

        corr_index = np.argmax(abs(self.corr), axis = 1)
        corr_max = self.corr[np.arange(len(self.corr)), corr_index]

        # calculate phase difference at autocorrelation peaks 
        self.phase_diff = np.arctan2(corr_max.imag, corr_max.real)

        # apply averaging of phase difference for successive autocorrelation cells
        downsamp_shape = self.phase_diff.shape[0]//self.range_cell_avg_factor
        self.phase_diff_avg_rg = np.mean(self.phase_diff[:downsamp_shape*self.range_cell_avg_factor]
                                         .reshape(downsamp_shape, self.range_cell_avg_factor), axis = 1)

        return
    

    def azimuth_average(self, window_function = False, progress_bar_dissable = True):
        """

        """
        self.chirp()
        self.pulse_pairs()

        np.random.seed(self.seed)
        corrs = []
        phase_diffs = []
        phase_diffs_avg_rg = []
        for i in tqdm(range(self.n_bursts), desc="Burst number"):

            self.simulate_reflection(seed=False)
            self.pulse_compress(window_function)
            self.delayed_autocorrelation(progress_bar_dissable = progress_bar_dissable)
            self.phase_difference()
            
            corrs.append(self.corr)
            phase_diffs.append(self.phase_diff)
            phase_diffs_avg_rg.append(self.phase_diff_avg_rg)

        # overwrite variables to store infor for all bursts
        self.corr = np.array(corrs)
        self.phase_diff = np.array(phase_diffs)
        self.phase_diff_avg_rg = np.array(phase_diffs_avg_rg)

        # calculate phase difference averaged over range and azimuth
        self.phase_diffs_avg_rg_az = np.mean(self.phase_diff_avg_rg, axis =0)

        return 