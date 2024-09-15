import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft

def plot_audio(x,y): 
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('wav!') 
    plt.show() 

def framing(sig, fs=16000, win_len=0.025, win_hop=0.01):
    """
    transform a signal into a series of overlapping frames.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.

    Returns:
        array of frames.
        frame length.
        number of frames.
    """
    # compute frame length and frame step (convert from seconds to samples)
    frame_length = win_len * fs
    frame_step = win_hop * fs
    signal_length = len(sig)
    frames_overlap = frame_length - frame_step

    # Make sure that we have at least 1 frame+
    num_frames = np.abs(signal_length - frames_overlap) // np.abs(frame_length - frames_overlap)
    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)

    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    if rest_samples != 0:
        pad_signal_length = int(frame_step - rest_samples)
        z = np.zeros((pad_signal_length))
        pad_signal = np.append(sig, z)
        num_frames += 1
    else:
        pad_signal = sig
    
    # make sure to use integers as indices
    frame_length = int(frame_length)
    frame_step = int(frame_step)
    num_frames = int(num_frames)

    # compute indices
    idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
    idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step),(frame_length, 1)).T
    indices = idx1 + idx2
    
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames, frame_length, num_frames

def haming_win(wav, frame_l):
    n = np.arange(0, frame_l)
    hw = 0.54 - (0.46 * np.cos(2*np.pi*n/frame_l))
    wav_windowed = wav * hw
    return wav_windowed

def windowing(frame_l, nframes, wav):
    windowed = []
    for i in range(nframes):
        w = haming_win(wav[i],frame_l)
        windowed.append(w)
    return windowed

def p_emphasis(signal, pre_emphasis):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal

def pre_emphasis(nframes, wav, pre_emphasis=0.96):
    emphasis = []
    for i in range(nframes):
        e = p_emphasis(wav[i], pre_emphasis)
        emphasis.append(e)
    return emphasis

def energy(wav, frame_l):
    sum=0
    for i in range(frame_l):
        sum += pow(wav[i], 2)
    return sum/frame_l

def signal_energy(frame_l, nframes, wav):
    sig_energy = []
    for i in range(nframes):
        e = energy(wav[i],frame_l)
        sig_energy.append(e)
    return sig_energy

def plot_energy(x,y):      
    plt.plot(x, y) 
    plt.xlabel('x - frames') 
    plt.ylabel('y - energy') 
    plt.title('energy plot!') 
    plt.show() 

def sgn(x):
    if x>=0: 
        return 1
    return -1

def ZCR(wav, frame_l):
    sum=0
    for i in range(1, frame_l):
        sum += (abs(sgn(wav[i]) - sgn(wav[i-1])) / 2)
    return sum/frame_l

def signal_zcr(frame_l, nframes, wav):
    sig_zcr = []
    for i in range(nframes):
        z = ZCR(wav[i],frame_l)
        sig_zcr.append(z)
    return sig_zcr

def plot_zcr(x,y): 
    plt.plot(x, y)
    plt.xlabel('x - frame')
    plt.ylabel('y - zcr')
    plt.title('zero crossing rate plot!') 
    plt.show()     

def autocorrelation_coefficient(signal):
    n = len(signal)
    autocorr = []
    
    for lag in range(n):
        sum_product = 0
        for i in range(n - lag):
            sum_product += signal[i] * signal[i + lag]
        autocorr.append(sum_product/n)
    
    return autocorr

def plot_autocorrelation(x,y): 
    plt.plot(x, y)
    plt.xlabel('x - lag')
    plt.ylabel('y - autocorrelation Coefficient')
    plt.title('autocorrelation Coefficient plot for 1 frame!') 
    plt.show()

def center_clip(S, cut):
    max_value = np.amax(S)
    c = cut * max_value

    C = []
    for value in S:
        if value > c: 
            C.append(value - c)
        elif value <-c: 
            C.append(value + c)
        C.append(0)
    return C

def threelevel_center_clip(S, cut):
    max_value = np.amax(S)
    c = cut * max_value

    C = []
    for value in S:
        if value > c: 
            C.append(1)
        elif value <-c: 
            C.append(-1)
        C.append(0)
    return C

def AMDF(signal):
    n = len(signal)
    amdf = []
    for lag in range(n):
        sum_product = 0
        for i in range(n - lag):
            sum_product += abs(signal[i] - signal[i + lag])
        amdf.append(sum_product/n)
    return amdf

def cepstral_analysis(signal):
    # FFT
    spectrum = []
    for frame in signal:
        spectrum.append(np.abs(fft(frame)))

    # Logarithm
    log_spectrum = np.log(np.array(spectrum))
    
    # IFFT
    cepstrum = []
    for frame in log_spectrum:
        cepstrum.append(np.real(ifft(frame)))
    
    return cepstrum

file_path = r'h_orig.wav'
sample_rate , samples = wavfile.read(file_path) 
#plot_audio(np.arange(len(samples)), samples)

#show signal that has been framed
print('-> signal that has been framed:')
wav_frame, frame_length, num_frames = framing(samples, fs=sample_rate)
#plot_audio(np.arange(len(wav_frame[185])), wav_frame[185])
#plot_audio(np.arange(len(wav_frame)), wav_frame)

#show signal that has been windowed(hamming)
print('-> signal that has been windowd by hamming function:')
wav_windowed = windowing(frame_length, num_frames, wav_frame)
#plot_audio(np.arange(len(wav_windowed[185])), wav_windowed[185])

#pre_emphasise the signal
print('-> signal that has been pre_emphasis:')
wav_emphasis = pre_emphasis(num_frames, wav_windowed, pre_emphasis=0.96)
#plot_audio(np.arange(len(wav_emphasis[185])), wav_emphasis[185])

#show energy of every frame in signal
print('-> energy of signal in every frame:')
wav_energy = signal_energy(frame_length, num_frames, wav_windowed)
#print(wav_energy,'\n')
#plot_energy(np.arange(len(wav_energy)), wav_energy)

#show zero crossing rate of every frame in signal
print('-> zero crosing rate of signal in every frame:')
wav_zcr = signal_zcr(frame_length, num_frames, wav_windowed)
#print(wav_zcr, '\n')
#plot_zcr(np.arange(len(wav_zcr)), wav_zcr)

#show autocorrelation of 1 random frame in signal
print('-> autocorrelation of signal in 1 frame:')
frame_autocorrelation = autocorrelation_coefficient(list(wav_emphasis[185]))
#print(frame_autocorrelation, '\n')
#plot_autocorrelation(np.arange(len(frame_autocorrelation)), frame_autocorrelation)

#center clipping of 1 random frame in signal
print('-> signal after center clipping:')
cc_frame = center_clip(wav_emphasis[185], 0.3)
#print(cc_frame, '\n')
#plot_audio(np.arange(len(cc_frame)), cc_frame)

#3 level center clipping of 1 random frame in signal
print('-> signal after 3 level center clipping:')
three_cc_frame = threelevel_center_clip(wav_emphasis[185], 0.3)
#print(three_cc_frame, '\n')
#plot_audio(np.arange(len(three_cc_frame)), three_cc_frame)

#show autocorrelation of 1 random frame in signal after center clipping
print('-> autocorrelation of signal in 1 frame after center clipping:')
frame_autocorrelation_cc = autocorrelation_coefficient(cc_frame)
#print(frame_autocorrelation_cc, '\n')
#plot_autocorrelation(np.arange(len(frame_autocorrelation_cc)), frame_autocorrelation_cc)


#show avarage magnitude differene function of 1 random frame in signal
print('-> varage magnitude differene function of signal in 1 frame:')
frame_amdf = AMDF(list(wav_emphasis[185]))
#print(frame_amdf, '\n')
#plot_audio(np.arange(len(frame_amdf)), frame_amdf)

#show Cepstral Coefficients of 1 random frame in signal
print('-> Cepstral Coefficients  of signal in 1 frame:')
cepstral = cepstral_analysis(wav_emphasis)
#print(cepstral_analysis, '\n')
#plot_audio(np.arange(len(cepstral[185])), cepstral[185])