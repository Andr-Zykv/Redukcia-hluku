import wave
import numpy as np
import matplotlib.pyplot as plt

def load_wav_numpy(path):
    with wave.open(path, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        raw_data = wf.readframes(num_frames)

    # Determine the dtype from sample width
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Convert the raw audio to numpy array
    audio = np.frombuffer(raw_data, dtype=dtype)

    # If stereo or multichannel → reshape
    if channels > 1:
        audio = audio.reshape(-1, channels)

    return sample_rate, audio, sample_width

def plot_waveform_origial(data):
    # Create time axis in seconds
    if data.ndim == 1:
        t = np.linspace(0, len(data) / sample_rate, num=len(data))
        plt.plot(t, data)
        plt.title("Waveform (Mono)")
    else:
        t = np.linspace(0, data.shape[0] / sample_rate, num=data.shape[0])
        for ch in range(data.shape[1]):
            plt.plot(t, data[:, ch], label=f"Channel {ch+1}")
        plt.legend()
        plt.title("Waveform (Multi-channel)")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def compute_fft(sample_rate, data):
    if data.ndim > 1:
        data = data[:, 0]

    N = len(data)
    fft_vals = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

    return freqs, fft_vals

def cut_noise(fft_vals):
    start = 300
    end = 8000
    # fft_vals[:start] = [0] * start
    # fft_vals[end:] = [0] * (len(fft_vals) - end)
    fft_vals[:start] *= np.linspace(0, 1, start)
    if (len(fft_vals) > end):
        fft_vals[end:] *= np.linspace(0, 1, len(fft_vals) - end)
    fft_vals[:] *= 2

def plot_fft(freqs, mags):
    plt.plot(freqs, mags)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Spectrum")
    plt.tight_layout()
    plt.show()

def inverse_fft(fft_vals):
    # Returns complex; take real part
    time_data = np.fft.irfft(fft_vals)
    return time_data

def plot_waveform_after_fft(sample_rate, data, title="Waveform"):
    t = np.linspace(0, len(data) / sample_rate, num=len(data))
    plt.plot(t, data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def save_wav(path, sample_rate, data, sample_width, channels = 1):
    # Clip samples to valid range
    if sample_width == 1:
        max_val = 127
        dtype = np.int8
    elif sample_width == 2:
        max_val = 32767
        dtype = np.int16
    elif sample_width == 4:
        max_val = 2147483647
        dtype = np.int32
    else:
        raise ValueError("Unsupported WAV sample width")

    # Normalize float to int
    data_int = np.clip(data, -max_val, max_val).astype(dtype)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(data_int.tobytes())

    print(f"Saved WAV: {path}")

def process_channel(data_ch, sample_rate):
    hop = int(0.03 * sample_rate)
    N = hop * 2 # 60 ms
    window = np.hanning(N)
    
    window_sum = np.zeros(len(data_ch))
    reconstructed = np.zeros(len(data_ch))

    for i in range (0, len(data_ch) - N, hop):
        freqs, fft_vals = compute_fft(sample_rate, data_ch[i:i + N] * window)
        # plot_fft(freqs, np.abs(fft_vals))
        cut_noise(fft_vals)
        chunk_out = inverse_fft(fft_vals) * window
        reconstructed[i:i + N] += chunk_out
        window_sum[i:i + N] += window ** 2

    reconstructed /= np.maximum(window_sum, 1e-8)
    return reconstructed

if __name__ == "__main__":
    sample_rate, data, sample_width = load_wav_numpy("input.wav")

    print("Sample rate:", sample_rate)
    print("Shape:", data.shape)

    # plot_waveform_origial(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]   

    num_samples, num_channels = data.shape

    processed = np.zeros_like(data, dtype=float)

    for ch in range(num_channels):
        processed[:, ch] = process_channel(data[:, ch], sample_rate)

    # plot_waveform_after_fft(sample_rate, reconstructed, title="Reconstructed Waveform (IFFT)")

    save_wav("output_reconstructed.wav", sample_rate, processed, sample_width, num_channels)