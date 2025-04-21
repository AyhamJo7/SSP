# (((1a))) Load the wave files using librosa with original sampling rate

import matplotlib.pyplot as plt
import numpy as np
import librosa
import sounddevice as sd

# Load both signals without changing their sampling rate
v_signal1, sampling_rate1 = librosa.load("1/speech1.wav", sr=None)
v_signal2, sampling_rate2 = librosa.load("1/speech2.wav", sr=None)
v_time1 = np.arange(len(v_signal1)) / sampling_rate1
v_time2 = np.arange(len(v_signal2)) / sampling_rate2


def main():

    v_signal1, sampling_rate1 = librosa.load("1/speech1.wav", sr=None)
    v_signal2, sampling_rate2 = librosa.load("1/speech2.wav", sr=None)

    # Print the sampling rates
    print(f"Sampling Rate 1: {sampling_rate1} Hz")
    print(f"Sampling Rate 2: {sampling_rate2} Hz")

    """
    Question: 
    What is the sampling frequency of the signals?

    Answer:
    Sampling Rate 1: 16000 Hz
    Sampling Rate 2: 16000 Hz
    """

    # (((1b))) Plot the signal as a function of time

    # Create time vectors
    v_time1 = np.arange(len(v_signal1)) / sampling_rate1
    v_time2 = np.arange(len(v_signal2)) / sampling_rate2

    # Plot both waveforms
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(v_time1, v_signal1)
    plt.title("Speech Signal 1")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(v_time2, v_signal2)
    plt.title("Speech Signal 2")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    """
    Question: 
    Identify the voiced, unvoiced and silence regions in the waveform. Which criteria did you use to
    distinguish between the three signal types?

    Answer: 
    Voiced: Periodic structure with high amplitude
    Unvoiced: Noise-like and less regular, lower amplitude
    Silence: Near-zero flat line

    """

    # (((1c))) Select voiced segments manually based on waveform inspection

    # For signal 1, select around 1.0–1.05s
    start1 = int(0.40 * sampling_rate1)
    end1   = int(0.45 * sampling_rate1)
    voiced_segment1 = v_signal1[start1:end1]
    t_voiced1 = np.arange(start1, end1) / sampling_rate1

    # For signal 2, select around 1.2–1.25s
    start2 = int(0.40 * sampling_rate2)
    end2   = int(0.45 * sampling_rate2)
    voiced_segment2 = v_signal2[start2:end2]
    t_voiced2 = np.arange(start2, end2) / sampling_rate2

    # Plot both segments for visual inspection
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(t_voiced1, voiced_segment1)
    plt.title("Voiced Segment from Signal 1")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(t_voiced2, voiced_segment2)
    plt.title("Voiced Segment from Signal 2")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    """
    Question:
    Plot your selected segments and describe your procedure. Judging based on the measured fundamental
    frequencies, do the signals originate rather from a male or a female speaker?

    Answer:
    Judging based on the measured fundamental frequencies:
    the signals from speech1.wav originate rather from a Female voice, ( Higher Amplitude, more periodic structure, and higher frequency content )
    while the signals from speech2 originate rather from a Male voice. 
    To make sure let's listen to the selected segments.
    """

    print("🔊 Playing Speech Signal 1 (Female voice)...")
    sd.play(v_signal1, samplerate=sampling_rate1)
    sd.wait()

    print("🔊 Playing Speech Signal 2 (Male voice)...")
    sd.play(v_signal2, samplerate=sampling_rate2)
    sd.wait()

    """
    Question:
    Verify your findings by listening to the signals! (In the following exercises you should be able to listen to audio data contained in numpy (abbreviated with np) arrays.

    Answer:
    Our Judging was correct, the first signal is indeed from a Female voice, while the second signal is from a male voice.
    """

if __name__ == "__main__":
    main()
