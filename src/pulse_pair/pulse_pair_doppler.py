from numpy.typing import ArrayLike
import numpy as np
import copy
from scipy.signal import correlate, fftconvolve
from scipy import stats
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, Union, List, Dict, Any
from stereoid.oceans.GMF.cmod5n import cmod5n_forward

c = 3e8 # m/s


# def temporal_decorr_phase_shift_calculator(target_coherence):
#     """

#     """
#     # constants from a least squares solution where A = [1, cos(phase_shift), cos(phase_shift)**2]
#     # NOTE might not work for all surfaces, least squares fit created on single surface
#     a, b, c = 0.40403017, 0.50515681, 0.09433892
#     phase_shift = np.arccos((-b + np.sqrt(b**2 - 4*c*(a-target_coherence))) / (2*c))

#     return phase_shift

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
    t_decor: Union[int, float] = None
    temporal_coherence: Union[int, float] = None
    n_reflectors: int = None
    oversample_retriev: Union[int, float] = 1
    range_cell_avg_factor: Union[int, float] = 2 # downsample factor after cross correlation
    range_cell_size_frac_pulse: Union[int, float] = 1
    Lambda = 0.05 # wavelength
    R = 700e3 # satellite elevation
    theta_min = 27 #minimum incidence angle, deg


    def __post_init__(self):
        if self.seed == None: self.seed = 42

        if ((type(self.temporal_coherence) != type(None)) | (type(self.t_decor) != type(None))) & (self.n_pulses < 2):
            raise Exception('Temporal decorrelation required at least two pulses')
        
        if (type(self.temporal_coherence) != type(None)) & (type(self.t_decor) != type(None)):
            raise Exception('Either select decorrelation time or temporal coherence, not both')
        
        elif type(self.t_decor) != type(None):
            self.temporal_coherence = self.decorr_coherence_calc(time_elapsed=self.t_pulse, time_decorr=self.t_decor)


    @staticmethod
    def calc_decorrelated_surface(coherence_target: float, surface):
        """
        # NOTE Only works if newly generated surface follwos same distribution as input surface
        We assume the input surface was generated with a rayleigh distribution of scale =1
        any other distribution will cause wrong values

        input
        -----

        return
        ------

        """
        warning_flag = False
        gamma = np.sqrt(coherence_target)
        N = surface.shape
        A = np.random.rayleigh(scale=1, size=N)
        phi = np.random.uniform(0,1, size=N)
        dist = A*np.exp(-1j*2*np.pi*phi)

        # test whether new distribution is similar to input
        if stats.ks_2samp(surface, dist).pvalue < 0.025:
             print('Generated surface distribution != input surface distribution (p < 0.025). Decorrelated surface likely too different')
             warning_flag = True
            #  raise Exception('Generated surface distribution != input surface distribution. Decorrelation will fail')

        surface_2 = gamma * surface + np.sqrt(1-abs(gamma)**2) * dist

        if warning_flag:
            corr = pulse_pair_doppler.coherence_calc(signal1=surface, signal2=surface_2)
            print(f'Coherence: {corr:.2f}')
        return surface_2
    
    @staticmethod
    def coherence_calc(signal1, signal2) -> float:
        """
        
        Yields similar performance as the mean output from scipy.signal.coherence using a 'hann' window
        """
        # Compute the cross-power spectral density (CSD)
        csd = np.mean(signal1 * np.conj(signal2))

        # Compute the power spectral density (PSD) for each signal
        psd_signal1 = np.mean(np.abs(signal1)**2)
        psd_signal2 = np.mean(np.abs(signal2)**2)

        # Calculate the coherence using the CSD and PSD values
        coherence = np.abs(csd)**2 / (psd_signal1 * psd_signal2)

        return coherence
    
    @staticmethod
    def decorr_coherence_calc(time_elapsed: Union[int, float, ArrayLike], time_decorr: Union[int, float, ArrayLike]) -> Union[float, ArrayLike]:
        """
        Calculates the coherence corresponding to elapsed time and a specified decorrelation time
        """
        return np.exp(-(time_elapsed**2/time_decorr**2)) # FIXME yields non-zero for time_elapsed = time_decorr


    def chirp(self, centre_around_baseband: bool = True):
        """

        """

        self.k_c = self.bandwidth/self.t_pulse # hz/s
        fs_nyq = 2 * (self.baseband + self.bandwidth)  # hz, times 2 to follow Nyquist
        self.fs = fs_nyq * self.oversample_retriev # hz
        s = 1/self.fs # stepsize
        
        self.pulse_samples = int(np.round(self.t_pulse/s))
        self.t_pulse_vector = np.linspace(0, self.t_pulse, self.pulse_samples) # s
        self.receive_samples = int(np.round(self.t_receive*self.fs ))

        b_c = self.k_c / 2 * self.t_pulse_vector # FIXME whys divide by 2?
        self.pulse = np.cos(2*np.pi*self.t_pulse_vector*(self.baseband + b_c)) + 1j*np.sin(2*np.pi*self.t_pulse_vector*(self.baseband + b_c))

        # set bandwidth to centre around baseband
        if centre_around_baseband:
            freq_shift = -self.bandwidth/2
            self.pulse = self.pulse * np.exp(1j*2*np.pi*freq_shift * self.t_pulse_vector)

        return self

    def pulse_pairs(self):
        """
        
        """
        # TODO add assert that interpulse delay matches n_pulses - 1
        sample_rate = self.t_pulse/self.pulse_samples

        # NOTE maybe T_interpulse is one index too long becase it takes a grid cell before and after as well
        self.interpulse_samples = int(np.ceil(self.t_interpulse/sample_rate))
        interpulse = np.zeros(self.interpulse_samples)
        self.signal = np.array(list(self.pulse) + (self.n_pulses -1) * (list(interpulse) + list(self.pulse)))

        return self
    
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

        # scatterers have rayleigh distributed amplitude and uniform phase
        surface_amplitude = np.random.rayleigh(1, self.receive_samples)
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

        self.surface = surface_amplitude  * surface_phase

        surface_amplitude_slope = cmod5n_forward(v=v, phi=phi, theta=theta)
        surface_amplitude_slope /= np.max(surface_amplitude_slope)
        filter = n_reflector_filter * surface_amplitude_slope

        if type(self.temporal_coherence) != type(None):

            self.subreflections = []
            len_subpulse = self.pulse_samples + self.interpulse_samples

            # instantiate decorrelated surface. or the first pulse this == original surface
            surface_decorr = copy.deepcopy(self.surface)

            for n in range(self.n_pulses):
                
                nth_pulse = np.zeros_like(self.signal).astype(np.complex128)
                nth_pulse[n * len_subpulse: (n+1) * len_subpulse] = self.signal[n * len_subpulse: (n+1) * len_subpulse]

                subreflection = fftconvolve(filter*surface_decorr, nth_pulse, mode = 'valid')
                self.subreflections.append(subreflection)

                # overwite decorrelated surface 
                surface_decorr = self.calc_decorrelated_surface(coherence_target=self.temporal_coherence, surface=surface_decorr)

            self.reflections = np.sum(self.subreflections, axis=0)

        else:

            self.reflections = fftconvolve(filter*self.surface, self.signal, mode = 'valid')  

        return self
    
    def pulse_compress(self, window_function: bool = False):
        """
        
        """
        if window_function:
            taper = np.hamming(self.pulse_samples)
        else:
            taper = 1
        
        self.pulse_compressed = correlate(self.reflections, taper*self.pulse, mode = "valid", method = "fft")

        return self
    

    def delayed_autocorrelation(self, progress_bar_dissable = False):
        """
        
        """

        window_size = int(self.pulse_samples // (1 / self.range_cell_size_frac_pulse)) 
        step_size = self.pulse_samples + self.interpulse_samples
        
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

        return self
    
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

        return self
    

    def azimuth_average(self, window_function = False, range_progress_bar_dissable = True, azimuth_progress_bar_dissable=False):
        """

        """
        self.chirp()
        self.pulse_pairs()

        np.random.seed(self.seed)
        corrs = []
        phase_diffs = []
        phase_diffs_avg_rg = []
        for i in tqdm(range(self.n_bursts), desc="Burst number", disable = azimuth_progress_bar_dissable):

            self.simulate_reflection(seed=False)
            self.pulse_compress(window_function)
            self.delayed_autocorrelation(progress_bar_dissable = range_progress_bar_dissable)
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

        return self