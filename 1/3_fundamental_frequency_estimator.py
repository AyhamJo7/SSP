import importlib
import numpy as np
import matplotlib.pyplot as plt

fundamental_estimation_by_eye = importlib.import_module("1_fundamental_estimation_by_eye")
block_processing = importlib.import_module("2_block_processing")
v_signal1 = fundamental_estimation_by_eye.v_signal1
v_signal2 = fundamental_estimation_by_eye.v_signal2
sampling_rate1 = fundamental_estimation_by_eye.sampling_rate1
sampling_rate2 = fundamental_estimation_by_eye.sampling_rate2
v_time1 = fundamental_estimation_by_eye.v_time1
v_time2 = fundamental_estimation_by_eye.v_time2
my_windowing = block_processing.my_windowing


# (((3a))) Split the signal into 32 ms frames with 16 ms shift

m_frames1, v_time_frame1 = my_windowing(v_signal1, sampling_rate1, frame_length_ms=32, frame_shift_ms=16)
m_frames2, v_time_frame2 = my_windowing(v_signal2, sampling_rate2, frame_length_ms=32, frame_shift_ms=16)
# Import the my_windowing function from the other file
print(f"Number of frames for signal 1: {len(m_frames1)}")
print(f"Number of frames for signal 2: {len(m_frames2)}")


# (((3b))) Compute the ACF using np.convolve 
# (((3c))) Remove the lower half (negative lags)

def compute_acf_matrix(m_frames: np.ndarray):
    """
    Compute the autocorrelation function (ACF) for each frame.
    Returns only the positive-lag part of the ACF (lags >= 0).

    Parameters:
        m_frames : 2D array of frames (each row is a frame)

    Returns:
        acf_matrix : 2D array with ACFs (lags >= 0)
    """
    num_frames, frame_len = m_frames.shape
    acf_matrix = np.zeros((num_frames, frame_len))

    for i in range(num_frames):
        frame = m_frames[i]
        # Full ACF via convolution of frame with time-reversed version
        acf_full = np.convolve(frame, frame[::-1], mode='full')
        # Keep only lags >= 0 (centered at frame_len - 1)
        acf_matrix[i] = acf_full[frame_len - 1:]

    return acf_matrix

acf_matrix1 = compute_acf_matrix(m_frames1)
acf_matrix2 = compute_acf_matrix(m_frames2)


# (((3d))) Estimate f0 using ACF peak in valid lag range

def estimate_f0_from_acf(acf_matrix: np.ndarray, sampling_rate: int, fmin=80, fmax=400):
    """
    Estimate the fundamental frequency from ACF for each frame.

    Parameters:
        acf_matrix   : 2D array of ACFs (positive lags only)
        sampling_rate: Sampling rate in Hz
        fmin         : Minimum allowed frequency (Hz)
        fmax         : Maximum allowed frequency (Hz)

    Returns:
        v_f0         : 1D array of estimated fundamental frequencies per frame
    """
    num_frames, frame_len = acf_matrix.shape
    v_f0 = np.zeros(num_frames)

    # Convert frequency range to lag range
    min_lag = sampling_rate // fmax
    max_lag = sampling_rate // fmin

    for i in range(num_frames):
        acf = acf_matrix[i]
        search_region = acf[min_lag:max_lag]
        if len(search_region) > 0:
            peak_index = np.argmax(search_region) + min_lag
            v_f0[i] = sampling_rate / peak_index
        else:
            v_f0[i] = 0.0  # fallback if no peak found

    return v_f0

v_f0_1 = estimate_f0_from_acf(acf_matrix1, sampling_rate1)
v_f0_2 = estimate_f0_from_acf(acf_matrix2, sampling_rate2)


# (((3e))) Plot the estimated f0

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(v_time1, v_signal1, label="Speech Signal 1")
plt.plot(v_time_frame1, v_f0_1 / max(v_f0_1) * max(v_signal1), label="Estimated $f_0$ (scaled)")
plt.title("Estimated Fundamental Frequency - Signal 1")
plt.xlabel("Time [s]")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(v_time2, v_signal2, label="Speech Signal 2")
plt.plot(v_time_frame2, v_f0_2 / max(v_f0_2) * max(v_signal2), label="Estimated $f_0$ (scaled)")
plt.title("Estimated Fundamental Frequency - Signal 2")
plt.xlabel("Time [s]")
plt.legend()

plt.tight_layout()
plt.show()

"""
Question:
In which parts of the signal does the fundamental frequency estimator give reasonable results and why? Do the estimated frequencies match your findings from the first exercise in Section 1?

Answer:
The estimator gives good results in voiced regions (where the waveform is periodic).
In unvoiced/silent segments, the ACF is flat → the estimated f0 drops or becomes unstable.
The result matches well with visual estimates from Exercise 1
Signal 1 → ~180 Hz → Female voice ✅
Signal 2 → ~100 Hz → Male voice ✅
"""