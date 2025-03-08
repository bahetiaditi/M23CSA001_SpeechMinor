import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import librosa
import librosa.display
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

# 1. Train/Test Split with random_state=45
def split_data(df, test_size=0.2, random_state=45):
    """
    Split the data into training and test sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing audio file information
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Controls the shuffling applied to the data before applying the split
        
    Returns:
    --------
    train_df : pandas.DataFrame
        Training set
    test_df : pandas.DataFrame
        Test set
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[['vowel', 'category']]  # Stratify by both vowel and category to ensure balanced splits
    )
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Check distribution of vowels in train and test sets
    train_vowel_dist = train_df['vowel'].value_counts(normalize=True)
    test_vowel_dist = test_df['vowel'].value_counts(normalize=True)
    
    print("\nVowel distribution in training set:")
    print(train_vowel_dist)
    print("\nVowel distribution in test set:")
    print(test_vowel_dist)
    
    return train_df, test_df


# 2. Preprocessing Functions
def preprocess_audio(file_path, target_sr=None, apply_preemphasis=True, preemphasis_coef=0.97):
    """
    Load and preprocess audio file.
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    target_sr : int or None
        Target sampling rate for resampling, None to keep original
    apply_preemphasis : bool
        Whether to apply pre-emphasis filter
    preemphasis_coef : float
        Pre-emphasis filter coefficient
        
    Returns:
    --------
    y : ndarray
        Audio signal
    sr : int
        Sampling rate
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=target_sr)
    
    # Apply pre-emphasis filter if requested
    if apply_preemphasis:
        y = librosa.effects.preemphasis(y, coef=preemphasis_coef)
    
    return y, sr


def frame_audio(y, sr, frame_length_ms=25, frame_shift_ms=10):
    """
    Segment audio into frames.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sampling rate
    frame_length_ms : int
        Frame length in milliseconds
    frame_shift_ms : int
        Frame shift in milliseconds
        
    Returns:
    --------
    frames : ndarray
        2D array with framed audio data, shape (num_frames, frame_length)
    """
    # Convert ms to samples
    frame_length = int(sr * frame_length_ms / 1000)
    frame_shift = int(sr * frame_shift_ms / 1000)
    
    # Calculate number of frames
    num_frames = 1 + int((len(y) - frame_length) / frame_shift)
    
    # Create empty array to hold frames
    frames = np.zeros((num_frames, frame_length))
    
    # Extract frames
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        frames[i, :] = y[start:end]
    
    return frames


def apply_window(frames, window_type='hamming'):
    """
    Apply window function to frames.
    
    Parameters:
    -----------
    frames : ndarray
        Framed audio data
    window_type : str
        Type of window ('hamming', 'hanning', etc.)
        
    Returns:
    --------
    windowed_frames : ndarray
        Windowed frames
    """
    frame_length = frames.shape[1]
    
    if window_type == 'hamming':
        window = np.hamming(frame_length)
    elif window_type == 'hanning':
        window = np.hanning(frame_length)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    
    windowed_frames = frames * window
    
    return windowed_frames


def remove_silence(y, sr, threshold_db=-40, min_silence_duration=0.1):
    """
    Remove silence or low-energy segments from audio.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sampling rate
    threshold_db : float
        Threshold for silence detection in dB
    min_silence_duration : float
        Minimum silence duration in seconds
        
    Returns:
    --------
    y_active : ndarray
        Audio signal with silence removed
    """
    # Compute RMS energy
    energy = librosa.feature.rms(y=y)[0]
    energy_db = librosa.amplitude_to_db(energy)
    
    # Find silent frames
    silent_frames = energy_db < threshold_db
    
    # Convert frames to samples
    hop_length = int(sr * 10 / 1000)
    silent_samples = librosa.frames_to_samples(np.where(silent_frames)[0], hop_length=hop_length)
    
    # Find continuous silent regions
    silent_regions = []
    current_region = []
    
    for sample in silent_samples:
        if not current_region or sample == current_region[-1] + 1:
            current_region.append(sample)
        else:
            if len(current_region) >= int(min_silence_duration * sr):
                silent_regions.append(current_region)
            current_region = [sample]
    
    if current_region and len(current_region) >= int(min_silence_duration * sr):
        silent_regions.append(current_region)
    
    # Create mask for active audio
    mask = np.ones_like(y, dtype=bool)
    for region in silent_regions:
        mask[region] = False
    
    # Apply mask
    y_active = y[mask]
    
    return y_active


# 3. Formant Extraction using LPC
def extract_formants_lpc(y, sr, num_formants=3, frame_length_ms=25, preemphasis=0.97):
    """
    Extract formant frequencies using Linear Predictive Coding (LPC).
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sampling rate
    num_formants : int
        Number of formants to extract
    frame_length_ms : int
        Frame length in milliseconds
    preemphasis : float
        Pre-emphasis filter coefficient
        
    Returns:
    --------
    formants : ndarray
        Array of formant frequencies, shape (num_frames, num_formants)
    lpc_coeffs : ndarray
        LPC coefficients for each frame
    """
    # Preprocess audio
    if preemphasis:
        y = librosa.effects.preemphasis(y, coef=preemphasis)
    
    # Frame length in samples
    frame_length = int(sr * frame_length_ms / 1000)
    
    # Determine LPC order
    lpc_order = int(2 + sr / 1000)
    
    # Extract frames
    frames = frame_audio(y, sr, frame_length_ms=frame_length_ms)
    
    # Apply window
    windowed_frames = apply_window(frames)
    
    # Initialize arrays for results
    num_frames = len(windowed_frames)
    formants = np.zeros((num_frames, num_formants))
    lpc_coeffs = np.zeros((num_frames, lpc_order + 1))
    
    # Process each frame
    for i, frame in enumerate(windowed_frames):
        # Calculate LPC coefficients
        a = librosa.lpc(frame, order=lpc_order)
        lpc_coeffs[i] = a
        
        # Find roots of LPC polynomial
        roots = np.roots(a)
        
        # Keep only roots with positive imaginary part (and inside unit circle)
        roots = roots[np.imag(roots) > 0]
        
        # Convert to frequencies
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * sr / (2 * np.pi)
        
        # Sort by frequency
        freqs = np.sort(freqs)
        
        # Store frequencies of the first num_formants formants
        formants[i, :min(len(freqs), num_formants)] = freqs[:min(len(freqs), num_formants)]
    
    # Remove NaN values by replacing with median values for each formant
    for j in range(num_formants):
        col = formants[:, j]
        median_val = np.median(col[col > 0])
        col[col == 0] = median_val
        formants[:, j] = col
    
    return formants, lpc_coeffs


def average_formants(formants, window_size=5):
    """
    Average formants across frames to find stable regions.
    
    Parameters:
    -----------
    formants : ndarray
        Array of formant frequencies, shape (num_frames, num_formants)
    window_size : int
        Window size for averaging
        
    Returns:
    --------
    avg_formants : ndarray
        Averaged formant frequencies
    """
    num_frames, num_formants = formants.shape
    
    # Use convolution for moving average
    avg_formants = np.zeros_like(formants)
    for i in range(num_formants):
        formant_values = formants[:, i]
        kernel = np.ones(window_size) / window_size
        avg_formants[:, i] = np.convolve(formant_values, kernel, mode='same')
    
    return avg_formants


def plot_lpc_spectrum(y, sr, lpc_order=None, ax=None):
    """
    Plot LPC spectrum alongside FFT spectrum.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sampling rate
    lpc_order : int or None
        LPC order, if None, use 2 + sr / 1000
    ax : matplotlib.axes.Axes or None
        Axes to plot on, if None, create new figure
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        Axes with plot
    """
    if lpc_order is None:
        lpc_order = int(2 + sr / 1000)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate LPC coefficients
    a = librosa.lpc(y, order=lpc_order)
    
    # Frequency response of the LPC filter
    freqs, h = signal.freqz(1.0, a, worN=2000, fs=sr)
    
    # Calculate FFT spectrum
    X = np.fft.rfft(y)
    freq = np.fft.rfftfreq(len(y), d=1/sr)
    
    # Plot FFT spectrum
    ax.plot(freq, 20 * np.log10(np.abs(X) + 1e-10), 'r', alpha=0.5, label='FFT Spectrum')
    
    # Plot LPC spectrum
    ax.plot(freqs, 20 * np.log10(np.abs(h)), 'b', linewidth=2, label='LPC Spectrum')
    
    # Find peaks in LPC spectrum (formants)
    peaks, _ = signal.find_peaks(20 * np.log10(np.abs(h)), height=-30, distance=int(sr/1000))
    ax.plot(freqs[peaks], 20 * np.log10(np.abs(h[peaks])), 'go', label='Formants')
    
    # For each peak, add a text label with the frequency
    for peak in peaks:
        ax.text(freqs[peak], 20 * np.log10(np.abs(h[peak])) + 3, f'{int(freqs[peak])} Hz', fontsize=8)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'LPC Spectrum (order={lpc_order}) and FFT Spectrum')
    ax.legend()
    ax.grid(True)
    
    return ax


# 4. Fundamental Frequency (F0) Extraction
def extract_f0_autocorr(y, sr, frame_length_ms=25, frame_shift_ms=10, f0_min=50, f0_max=500):
    """
    Extract fundamental frequency using autocorrelation method.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sampling rate
    frame_length_ms : int
        Frame length in milliseconds
    frame_shift_ms : int
        Frame shift in milliseconds
    f0_min : int
        Minimum F0 in Hz
    f0_max : int
        Maximum F0 in Hz
        
    Returns:
    --------
    f0 : ndarray
        Fundamental frequency for each frame
    """
    # Frame audio
    frames = frame_audio(y, sr, frame_length_ms, frame_shift_ms)
    windowed_frames = apply_window(frames)
    
    # Initialize array for results
    num_frames = len(frames)
    f0 = np.zeros(num_frames)
    
    # Convert min/max F0 to lags in samples
    min_lag = int(sr / f0_max)
    max_lag = int(sr / f0_min)
    
    # Process each frame
    for i, frame in enumerate(windowed_frames):
        # Compute autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:] # Keep only positive lags
        
        # Find peak in the desired range
        if max_lag < len(autocorr):
            peaks, _ = signal.find_peaks(autocorr[min_lag:max_lag])
            if len(peaks) > 0:
                peak_lag = min_lag + peaks[np.argmax(autocorr[min_lag:max_lag][peaks])]
                f0[i] = sr / peak_lag
            else:
                f0[i] = 0  # No peak found
        else:
            f0[i] = 0  # Not enough samples
    
    # Replace zeros with median value
    nonzero_f0 = f0[f0 > 0]
    if len(nonzero_f0) > 0:
        median_f0 = np.median(nonzero_f0)
        f0[f0 == 0] = median_f0
    
    return f0


def extract_f0_amdf(y, sr, frame_length_ms=25, frame_shift_ms=10, f0_min=50, f0_max=500):
    """
    Extract fundamental frequency using Average Magnitude Difference Function (AMDF).
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sampling rate
    frame_length_ms : int
        Frame length in milliseconds
    frame_shift_ms : int
        Frame shift in milliseconds
    f0_min : int
        Minimum F0 in Hz
    f0_max : int
        Maximum F0 in Hz
        
    Returns:
    --------
    f0 : ndarray
        Fundamental frequency for each frame
    """
    # Frame audio
    frames = frame_audio(y, sr, frame_length_ms, frame_shift_ms)
    windowed_frames = apply_window(frames)
    
    # Initialize array for results
    num_frames = len(frames)
    f0 = np.zeros(num_frames)
    
    # Convert min/max F0 to lags in samples
    min_lag = int(sr / f0_max)
    max_lag = int(sr / f0_min)
    
    # Process each frame
    for i, frame in enumerate(windowed_frames):
        # Compute AMDF
        amdf = np.zeros(max_lag - min_lag)
        for j in range(min_lag, max_lag):
            amdf[j - min_lag] = np.sum(np.abs(frame[:-j] - frame[j:]))
        
        # Find minimum in AMDF (which corresponds to period)
        if len(amdf) > 0:
            # Invert AMDF to find peaks
            inverted_amdf = np.max(amdf) - amdf
            peaks, _ = signal.find_peaks(inverted_amdf)
            if len(peaks) > 0:
                peak_lag = min_lag + peaks[np.argmax(inverted_amdf[peaks])]
                f0[i] = sr / peak_lag
            else:
                f0[i] = 0  # No peak found
        else:
            f0[i] = 0  # Not enough samples
    
    # Replace zeros with median value
    nonzero_f0 = f0[f0 > 0]
    if len(nonzero_f0) > 0:
        median_f0 = np.median(nonzero_f0)
        f0[f0 == 0] = median_f0
    
    return f0


# 5. Visualization Functions
def plot_waveform(y, sr, title=None):
    """
    Plot waveform of audio signal.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sampling rate
    title : str or None
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with plot
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Audio Waveform')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    
    return fig


def plot_spectrogram(y, sr, title=None):
    """
    Plot spectrogram of audio signal.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sampling rate
    title : str or None
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute and display spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax)
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Spectrogram')
    
    return fig


def plot_vowel_space_2d(formants, vowels, categories=None, title='Vowel Space (F1-F2)'):
    """
    Plot vowel space using F1 and F2.
    
    Parameters:
    -----------
    formants : ndarray
        Array of formant frequencies, shape (n_samples, n_formants)
    vowels : ndarray
        Array of vowel labels
    categories : ndarray or None
        Array of category labels
    title : str
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Reverse axes for conventional phonetic plots (F1 increases downward, F2 increases leftward)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Extract F1 and F2
    f1 = formants[:, 0]
    f2 = formants[:, 1]
    
    # Plot points with different colors for different vowels and markers for categories
    if categories is not None:
        # Create scatter plot for each vowel and category combination
        for vowel in np.unique(vowels):
            for category in np.unique(categories):
                mask = (vowels == vowel) & (categories == category)
                ax.scatter(f2[mask], f1[mask], alpha=0.7, label=f'{vowel}-{category}')
    else:
        # Create scatter plot for each vowel
        for vowel in np.unique(vowels):
            mask = vowels == vowel
            ax.scatter(f2[mask], f1[mask], alpha=0.7, label=vowel)
    
    # Add labels for vowel centroids
    for vowel in np.unique(vowels):
        mask = vowels == vowel
        f1_mean = np.mean(f1[mask])
        f2_mean = np.mean(f2[mask])
        ax.text(f2_mean, f1_mean, vowel, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('F2 (Hz)')
    ax.set_ylabel('F1 (Hz)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True)
    
    return fig


def plot_vowel_space_3d(formants, vowels, categories=None, title='Vowel Space (F1-F2-F3)'):
    """
    Plot 3D vowel space using F1, F2, and F3.
    
    Parameters:
    -----------
    formants : ndarray
        Array of formant frequencies, shape (n_samples, n_formants)
    vowels : ndarray
        Array of vowel labels
    categories : ndarray or None
        Array of category labels
    title : str
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with plot
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract F1, F2, and F3
    f1 = formants[:, 0]
    f2 = formants[:, 1]
    f3 = formants[:, 2]
    
    # Plot points with different colors for different vowels and markers for categories
    if categories is not None:
        # Create scatter plot for each vowel and category combination
        for vowel in np.unique(vowels):
            for category in np.unique(categories):
                mask = (vowels == vowel) & (categories == category)
                ax.scatter(f2[mask], f1[mask], f3[mask], alpha=0.7, label=f'{vowel}-{category}')
    else:
        # Create scatter plot for each vowel
        for vowel in np.unique(vowels):
            mask = vowels == vowel
            ax.scatter(f2[mask], f1[mask], f3[mask], alpha=0.7, label=vowel)
    
    # Add labels for vowel centroids
    for vowel in np.unique(vowels):
        mask = vowels == vowel
        f1_mean = np.mean(f1[mask])
        f2_mean = np.mean(f2[mask])
        f3_mean = np.mean(f3[mask])
        ax.text(f2_mean, f1_mean, f3_mean, vowel, fontsize=12, fontweight='bold')
    
    # Reverse x and y axes for conventional phonetic plots
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    ax.set_xlabel('F2 (Hz)')
    ax.set_ylabel('F1 (Hz)')
    ax.set_zlabel('F3 (Hz)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    return fig


def plot_f0_distribution(f0_values, vowels, categories=None, title='F0 Distribution by Vowel'):
    """
    Plot F0 distribution for each vowel.
    
    Parameters:
    -----------
    f0_values : ndarray
        Array of F0 values
    vowels : ndarray
        Array of vowel labels
    categories : ndarray or None
        Array of category labels
    title : str
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with plot
    """
    if categories is not None:
        # Create violin plot for each vowel and category combination
        df = pd.DataFrame({
            'F0': f0_values,
            'Vowel': vowels,
            'Category': categories
        })
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.violinplot(x='Vowel', y='F0', hue='Category', data=df, split=True, inner='quartile', ax=ax)
    else:
        # Create violin plot for each vowel
        df = pd.DataFrame({
            'F0': f0_values,
            'Vowel': vowels
        })
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(x='Vowel', y='F0', data=df, inner='quartile', ax=ax)
    
    ax.set_xlabel('Vowel')
    ax.set_ylabel('F0 (Hz)')
    ax.set_title(title)
    
    return fig


def plot_feature_correlation(features, feature_names, title='Feature Correlation Matrix'):
    """
    Plot correlation matrix of features.
    
    Parameters:
    -----------
    features : ndarray
        Array of feature values, shape (n_samples, n_features)
    feature_names : list
        List of feature names
    title : str
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with plot
    """
    # Create correlation matrix
    corr = np.corrcoef(features.T)
    
    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names)
    ax.set_yticklabels(feature_names)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add values in cells
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{corr[i, j]:.2f}', 
                           ha='center', va='center', 
                           color='white' if abs(corr[i, j]) > 0.5 else 'black')
    
    ax.set_title(title)
    fig.tight_layout()
    
    return fig


# 6. Feature extraction wrapper function
def extract_features(audio_file, category=None, vowel=None):
    """
    Extract all features from an audio file.
    
    Parameters:
    -----------
    audio_file : str
        Path to audio file
    category : str or None
        Category label
    vowel : str or None
        Vowel label
        
    Returns:
    --------
    features : dict
        Dictionary of extracted features
    """
    # Load and preprocess audio
    y, sr = preprocess_audio(audio_file, apply_preemphasis=True)
    
    # Remove silence
    y = remove_silence(y, sr)
    
    # Extract formants
    formants, lpc_coeffs = extract_formants_lpc(y, sr, num_formants=3)
    
    # Average formants
    avg_formants = average_formants(formants)
    
    # Extract F0 using autocorrelation
    f0_autocorr = extract_f0_autocorr(y, sr)
    
    # Extract F0 using AMDF for comparison
    f0_amdf = extract_f0_amdf(y, sr)
    
    # Compute mean and standard deviation of features
    formant_means = np.mean(formants, axis=0)
    formant_stds = np.std(formants, axis=0)
    
    f0_autocorr_mean = np.mean(f0_autocorr)
    f0_autocorr_std = np.std(f0_autocorr)
    
    f0_amdf_mean = np.mean(f0_amdf)
    f0_amdf_std = np.std(f0_amdf)
    
    # Create feature dictionary
    features = {
        'file_path': audio_file,
        'category': category,
        'vowel': vowel,
        'F1_mean': formant_means[0],
        'F2_mean': formant_means[1],
        'F3_mean': formant_means[2],
        'F1_std': formant_stds[0],
        'F2_std': formant_stds[1],
        'F3_std': formant_stds[2],
        'F0_autocorr_mean': f0_autocorr_mean,
        'F0_autocorr_std': f0_autocorr_std,
        'F0_amdf_mean': f0_amdf_mean,
        'F0_amdf_std': f0_amdf_std,
        'raw_audio': y,
        'sample_rate': sr,
        'formants': formants,
        'avg_formants': avg_formants,
        'f0_autocorr': f0_autocorr,
        'f0_amdf': f0_amdf
    }
    
    return features


# 7. Process all data and extract features
def process_dataset(df):
    """
    Process all files in the dataset and extract features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing audio file information
        
    Returns:
    --------
    features_df : pandas.DataFrame
        DataFrame containing extracted features
    """
    all_features = []
    
    # Process each file
    for i, row in df.iterrows():
        print(f"Processing file {i+1}/{len(df)}: {row['file_path']}")
        
        try:
            # Extract features
            features = extract_features(
                row['file_path'],
                category=row['category'],
                vowel=row['vowel']
            )
            
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {row['file_path']}: {e}")
    
    # Convert raw feature list to DataFrame
    features_df = pd.DataFrame([
        {k: v for k, v in feature.items() if not isinstance(v, np.ndarray)} 
        for feature in all_features
    ])
    
    # Store raw data and arrays separately (optional)
    raw_data = {i: {} for i in range(len(all_features))}
    for i, feature in enumerate(all_features):
        for k, v in feature.items():
            if isinstance(v, np.ndarray):
                raw_data[i][k] = v
    
    print(f"Extracted features from {len(features_df)} files")
    
    return features_df, raw_data

# 8. Additional visualization functions
def plot_interactive_vowel_space(formants, vowels, categories=None, title='Interactive Vowel Space'):
    """
    Create an interactive 3D vowel space plot using Plotly.
    
    Parameters:
    -----------
    formants : ndarray
        Array of formant frequencies, shape (n_samples, n_formants)
    vowels : ndarray
        Array of vowel labels
    categories : ndarray or None
        Array of category labels
    title : str
        Plot title
        
    Returns:
    --------
    fig : plotly.graph_objs.Figure
        Interactive Plotly figure
    """
    # Create DataFrame for Plotly
    df = pd.DataFrame({
        'F1': formants[:, 0],
        'F2': formants[:, 1],
        'F3': formants[:, 2] if formants.shape[1] > 2 else np.zeros(len(formants)),
        'Vowel': vowels
    })
    
    if categories is not None:
        df['Category'] = categories
    
    # Create figure
    if categories is not None:
        fig = px.scatter_3d(df, x='F2', y='F1', z='F3', color='Vowel', symbol='Category',
                         labels={'F1': 'F1 (Hz)', 'F2': 'F2 (Hz)', 'F3': 'F3 (Hz)'},
                         hover_data=['Vowel', 'Category'])
    else:
        fig = px.scatter_3d(df, x='F2', y='F1', z='F3', color='Vowel',
                         labels={'F1': 'F1 (Hz)', 'F2': 'F2 (Hz)', 'F3': 'F3 (Hz)'},
                         hover_data=['Vowel'])
    
    # Update layout
    fig.update_layout(
        title=title,
        legend_title='Vowel',
        scene=dict(
            xaxis_title='F2 (Hz)',
            yaxis_title='F1 (Hz)',
            zaxis_title='F3 (Hz)',
            # Reverse x and y axes for conventional phonetic plots
            xaxis=dict(autorange='reversed'),
            yaxis=dict(autorange='reversed')
        )
    )
    
    return fig


def plot_formant_trajectories(formants_list, times_list, vowels, title='Formant Trajectories'):
    """
    Plot formant trajectories for different vowels.
    
    Parameters:
    -----------
    formants_list : list of ndarrays
        List of formant frequency arrays, each of shape (n_frames, n_formants)
    times_list : list of ndarrays
        List of time arrays corresponding to frames
    vowels : list
        List of vowel labels
    title : str
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors for vowels
    vowel_colors = {vowel: plt.cm.tab10(i % 10) for i, vowel in enumerate(set(vowels))}
    
    # Plot F1, F2, and F3 trajectories
    for i, formant_idx in enumerate([0, 1, 2]):
        ax = axes[i]
        
        for formants, times, vowel in zip(formants_list, times_list, vowels):
            ax.plot(times, formants[:, formant_idx], color=vowel_colors[vowel], label=vowel)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'F{formant_idx+1} (Hz)')
        ax.set_title(f'F{formant_idx+1} Trajectories')
        
        # Add legend for first plot only
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
        
        ax.grid(True)
    
    fig.suptitle(title)
    fig.tight_layout()
    
    return fig


def plot_hierarchical_clustering(formants, vowels, categories=None, title='Hierarchical Clustering of Vowels'):
    """
    Create hierarchical clustering dendrogram of vowels based on formant values.
    
    Parameters:
    -----------
    formants : ndarray
        Array of formant frequencies, shape (n_samples, n_formants)
    vowels : ndarray
        Array of vowel labels
    categories : ndarray or None
        Array of category labels
    title : str
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with plot
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    
    # Create labels
    if categories is not None:
        labels = [f"{v}_{c}" for v, c in zip(vowels, categories)]
    else:
        labels = vowels
    
    # Compute linkage matrix
    Z = linkage(formants, method='ward')
    
    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(
        Z,
        labels=labels,
        leaf_font_size=10,
        ax=ax
    )
    
    ax.set_xlabel('Sample')
    ax.set_ylabel('Distance')
    ax.set_title(title)
    
    plt.tight_layout()
    
    return fig


def plot_voronoi_vowel_space(formants, vowels, title='Vowel Classification Regions'):
    """
    Create Voronoi diagram showing classification regions in F1-F2 space.
    
    Parameters:
    -----------
    formants : ndarray
        Array of formant frequencies, shape (n_samples, n_formants)
    vowels : ndarray
        Array of vowel labels
    title : str
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with plot
    """
    from scipy.spatial import Voronoi, voronoi_plot_2d
    
    # Extract F1 and F2
    f1 = formants[:, 0]
    f2 = formants[:, 1]
    
    # Compute vowel centroids
    centroids = []
    centroid_labels = []
    
    for vowel in np.unique(vowels):
        mask = vowels == vowel
        f1_mean = np.mean(f1[mask])
        f2_mean = np.mean(f2[mask])
        centroids.append([f2_mean, f1_mean])  # Note: x=F2, y=F1
        centroid_labels.append(vowel)
    
    centroids = np.array(centroids)
    
    # Compute Voronoi diagram
    vor = Voronoi(centroids)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Voronoi diagram
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_width=2, line_alpha=0.6, point_size=0)
    
    # Plot centroids and labels
    for i, (x, y) in enumerate(centroids):
        ax.scatter(x, y, color=plt.cm.tab10(i % 10), s=100, zorder=5)
        ax.text(x, y, centroid_labels[i], fontsize=12, fontweight='bold', ha='center', va='center', zorder=6)
    
    # Plot all data points
    for i, vowel in enumerate(np.unique(vowels)):
        mask = vowels == vowel
        ax.scatter(f2[mask], f1[mask], color=plt.cm.tab10(i % 10), alpha=0.3, label=vowel)
    
    # Reverse axes for conventional phonetic plots
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    ax.set_xlabel('F2 (Hz)')
    ax.set_ylabel('F1 (Hz)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True)
    
    return fig


# 9. Feature normalization functions
def normalize_features(features_df, method='z-score'):
    """
    Normalize features to account for physiological differences.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing features
    method : str
        Normalization method ('z-score', 'min-max', or 'bark')
        
    Returns:
    --------
    normalized_df : pandas.DataFrame
        DataFrame with normalized features
    """
    # Create copy of features DataFrame
    normalized_df = features_df.copy()
    
    # Select numerical columns to normalize
    feature_cols = ['F1_mean', 'F2_mean', 'F3_mean', 
                   'F1_std', 'F2_std', 'F3_std',
                   'F0_autocorr_mean', 'F0_autocorr_std']
    
    if method == 'z-score':
        # Z-score normalization
        for col in feature_cols:
            normalized_df[f'{col}_norm'] = (features_df[col] - features_df[col].mean()) / features_df[col].std()
    
    elif method == 'min-max':
        # Min-max normalization
        for col in feature_cols:
            normalized_df[f'{col}_norm'] = (features_df[col] - features_df[col].min()) / (features_df[col].max() - features_df[col].min())
    
    elif method == 'bark':
        # Bark scale for formant normalization (perceptual scale)
        # Formula: Bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f/7500)²)
        for col in ['F1_mean', 'F2_mean', 'F3_mean']:
            f = features_df[col]
            normalized_df[f'{col}_bark'] = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f/7500)**2)
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized_df


def calculate_speaker_normalization(features_df, method='lobanov'):
    """
    Apply speaker normalization to formant values.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing features
    method : str
        Normalization method ('lobanov', 'nearey', or 'wattfabricius')
        
    Returns:
    --------
    normalized_df : pandas.DataFrame
        DataFrame with normalized features
    """
    normalized_df = features_df.copy()
    
    if method == 'lobanov':
        # Lobanov normalization: z-score within each speaker
        for category in features_df['category'].unique():
            category_mask = features_df['category'] == category
            
            for formant in ['F1_mean', 'F2_mean', 'F3_mean']:
                formant_values = features_df.loc[category_mask, formant]
                mean = formant_values.mean()
                std = formant_values.std()
                
                normalized_df.loc[category_mask, f'{formant}_lobanov'] = (formant_values - mean) / std
    
    elif method == 'nearey':
        # Nearey normalization: log-mean normalization within each speaker
        for category in features_df['category'].unique():
            category_mask = features_df['category'] == category
            
            for formant in ['F1_mean', 'F2_mean', 'F3_mean']:
                formant_values = features_df.loc[category_mask, formant]
                log_values = np.log(formant_values)
                mean_log = log_values.mean()
                
                normalized_df.loc[category_mask, f'{formant}_nearey'] = log_values - mean_log
    
    elif method == 'wattfabricius':
        # Watt-Fabricius normalization
        for category in features_df['category'].unique():
            category_mask = features_df['category'] == category
            
            # Find reference vowels (typically /i/ for F2 max, /a/ for F1 max)
            category_data = features_df[category_mask]
            
            # Use /i/ (heed) as reference for max F2
            i_mask = (category_data['vowel'] == '/i/') & (category_data['word'] == 'heed')
            if sum(i_mask) > 0:
                i_f1 = category_data.loc[i_mask, 'F1_mean'].mean()
                i_f2 = category_data.loc[i_mask, 'F2_mean'].mean()
            else:
                # Fallback: use max F2 vowel
                max_f2_idx = category_data['F2_mean'].idxmax()
                i_f1 = category_data.loc[max_f2_idx, 'F1_mean']
                i_f2 = category_data.loc[max_f2_idx, 'F2_mean']
            
            # Use /a/ (had) as reference for max F1
            a_mask = (category_data['vowel'] == '/a/') & (category_data['word'] == 'had')
            if sum(a_mask) > 0:
                a_f1 = category_data.loc[a_mask, 'F1_mean'].mean()
                a_f2 = category_data.loc[a_mask, 'F2_mean'].mean()
            else:
                # Fallback: use max F1 vowel
                max_f1_idx = category_data['F1_mean'].idxmax()
                a_f1 = category_data.loc[max_f1_idx, 'F1_mean']
                a_f2 = category_data.loc[max_f1_idx, 'F2_mean']
            
            # Calculate centroid
            centroid_f1 = (i_f1 + a_f1) / 2
            centroid_f2 = (i_f2 + a_f2) / 2
            
            # Normalize
            f1_values = features_df.loc[category_mask, 'F1_mean']
            f2_values = features_df.loc[category_mask, 'F2_mean']
            
            normalized_df.loc[category_mask, 'F1_wattfabricius'] = (f1_values - centroid_f1) / (a_f1 - centroid_f1)
            normalized_df.loc[category_mask, 'F2_wattfabricius'] = (f2_values - centroid_f2) / (i_f2 - centroid_f2)
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized_df


# 10. Additional feature extraction for classification
def extract_mfccs(y, sr, n_mfcc=13):
    """
    Extract Mel-frequency cepstral coefficients.
    
    Parameters:
    -----------
    y : ndarray
        Audio signal
    sr : int
        Sampling rate
    n_mfcc : int
        Number of MFCCs to extract
        
    Returns:
    --------
    mfccs : ndarray
        Array of MFCCs, shape (n_mfcc, n_frames)
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Compute statistics
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    
    return mfccs, mfcc_means, mfcc_stds


def create_feature_vector(features, include_mfccs=True):
    """
    Create feature vector for classification.
    
    Parameters:
    -----------
    features : dict
        Dictionary of extracted features
    include_mfccs : bool
        Whether to include MFCCs
        
    Returns:
    --------
    feature_vector : ndarray
        Feature vector
    feature_names : list
        List of feature names
    """
    # Start with formants and F0
    feature_vector = [
        features['F1_mean'],
        features['F2_mean'],
        features['F3_mean'],
        features['F1_std'],
        features['F2_std'],
        features['F3_std'],
        features['F0_autocorr_mean'],
        features['F0_autocorr_std']
    ]
    
    feature_names = [
        'F1_mean',
        'F2_mean',
        'F3_mean',
        'F1_std',
        'F2_std',
        'F3_std',
        'F0_autocorr_mean',
        'F0_autocorr_std'
    ]
    
    # Add ratios
    f1 = features['F1_mean']
    f2 = features['F2_mean']
    f3 = features['F3_mean']
    
    feature_vector.extend([
        f2/f1,  # F2/F1 ratio
        f3/f2,  # F3/F2 ratio
    ])
    
    feature_names.extend([
        'F2/F1_ratio',
        'F3/F2_ratio'
    ])
    
    # Add MFCCs if requested
    if include_mfccs:
        # Calculate MFCCs
        _, mfcc_means, mfcc_stds = extract_mfccs(features['raw_audio'], features['sample_rate'])
        
        # Add mean and std of first few MFCCs
        for i in range(len(mfcc_means)):
            feature_vector.append(mfcc_means[i])
            feature_names.append(f'MFCC{i+1}_mean')
        
        for i in range(len(mfcc_stds)):
            feature_vector.append(mfcc_stds[i])
            feature_names.append(f'MFCC{i+1}_std')
    
    return np.array(feature_vector), feature_names


import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Run the complete vowel classification pipeline including feature extraction,
    visualization, and saving outputs in a user-specified directory.
    """
    # ----- Step 0: Ask user for output directory
    output_dir = '/Users/aditibaheti/Desktop/IITJ/Speech/M23CSA001_SpeechMinor/Ques_3'

    # ----- Step 1: Load data
    # Change the CSV path to your local CSV file path.
    data_csv_path = "/Users/aditibaheti/Desktop/IITJ/Speech/M23CSA001_SpeechMinor/Ques_3/Ques_3.csv"
    df = pd.read_csv(data_csv_path)
    
    # ----- Step 2: Split data (80:20 with random_state=45)
    train_df, test_df = split_data(df, test_size=0.2, random_state=45)
    
    # ----- Step 3: Process dataset and extract features
    print("Extracting features from training set...")
    train_features_df, train_raw_data = process_dataset(train_df)
    
    print("Extracting features from test set...")
    test_features_df, test_raw_data = process_dataset(test_df)
    
    # ----- Step 4: Normalize features (z-score + speaker normalization)
    train_features_norm = normalize_features(train_features_df, method='z-score')
    test_features_norm = normalize_features(test_features_df, method='z-score')
    train_features_norm = calculate_speaker_normalization(train_features_norm, method='lobanov')
    test_features_norm = calculate_speaker_normalization(test_features_norm, method='lobanov')
    
    # ----- Step 5: Create feature vectors for classification
    X_train = []
    y_train = []
    feature_names = None
    for i, row in train_features_df.iterrows():
        features = {
            'F1_mean': row['F1_mean'],
            'F2_mean': row['F2_mean'],
            'F3_mean': row['F3_mean'],
            'F1_std': row['F1_std'],
            'F2_std': row['F2_std'],
            'F3_std': row['F3_std'],
            'F0_autocorr_mean': row['F0_autocorr_mean'],
            'F0_autocorr_std': row['F0_autocorr_std'],
            'raw_audio': train_raw_data[i]['raw_audio'],
            'sample_rate': row['sample_rate']
        }
        feature_vector, feature_names = create_feature_vector(features)
        X_train.append(feature_vector)
        y_train.append(row['vowel'])
    
    X_test = []
    y_test = []
    for i, row in test_features_df.iterrows():
        features = {
            'F1_mean': row['F1_mean'],
            'F2_mean': row['F2_mean'],
            'F3_mean': row['F3_mean'],
            'F1_std': row['F1_std'],
            'F2_std': row['F2_std'],
            'F3_std': row['F3_std'],
            'F0_autocorr_mean': row['F0_autocorr_mean'],
            'F0_autocorr_std': row['F0_autocorr_std'],
            'raw_audio': test_raw_data[i]['raw_audio'],
            'sample_rate': row['sample_rate']
        }
        feature_vector, _ = create_feature_vector(features)
        X_test.append(feature_vector)
        y_test.append(row['vowel'])
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # ----- Step 6: Visualizations
    # a. Waveform and Spectrogram for a sample audio from training set
    sample_idx = 0  # Taking the first successfully processed training sample
    sample_raw = train_raw_data[sample_idx]['raw_audio']
    sample_sr = train_features_df.iloc[sample_idx]['sample_rate']

    
    fig_waveform = plot_waveform(sample_raw, sample_sr, title="Waveform Sample")
    fig_waveform.savefig(os.path.join(output_dir, "waveform_sample.png"))
    plt.close(fig_waveform)
    
    fig_spectrogram = plot_spectrogram(sample_raw, sample_sr, title="Spectrogram Sample")
    fig_spectrogram.savefig(os.path.join(output_dir, "spectrogram_sample.png"))
    plt.close(fig_spectrogram)
    
    # b. LPC Spectrum for the sample audio
    fig_lpc = plt.figure(figsize=(10, 6))
    ax = fig_lpc.add_subplot(111)
    plot_lpc_spectrum(sample_raw, sample_sr, ax=ax)
    fig_lpc.savefig(os.path.join(output_dir, "lpc_spectrum_sample.png"))
    plt.close(fig_lpc)
    
    # c. 2D Vowel Space (F1-F2) using training features
    # Use F1, F2 and F3 columns; the plotting function uses columns 0 and 1 as F1 and F2.
    formants_2d = train_features_df[['F1_mean', 'F2_mean', 'F3_mean']].to_numpy()
    vowels_train = train_features_df['vowel'].to_numpy()
    categories_train = train_features_df['category'].to_numpy() if 'category' in train_features_df.columns else None
    fig_vowel_space_2d = plot_vowel_space_2d(formants_2d, vowels_train, categories=categories_train, title="2D Vowel Space (F1-F2)")
    fig_vowel_space_2d.savefig(os.path.join(output_dir, "vowel_space_2d.png"))
    plt.close(fig_vowel_space_2d)
    
    # d. 3D Vowel Space (F1-F2-F3) using training features
    fig_vowel_space_3d = plot_vowel_space_3d(formants_2d, vowels_train, categories=categories_train, title="3D Vowel Space (F1-F2-F3)")
    fig_vowel_space_3d.savefig(os.path.join(output_dir, "vowel_space_3d.png"))
    plt.close(fig_vowel_space_3d)
    
    # e. F0 Distribution by Vowel
    # Aggregate F0 values from the raw_data across training samples.
    f0_values = []
    vowels_f0 = []
    for idx, data in train_raw_data.items():
        if 'f0_autocorr' in data:
            f0_vals = data['f0_autocorr']
            f0_values.extend(f0_vals.tolist())
            vowels_f0.extend([train_features_df.iloc[idx]['vowel']] * len(f0_vals))
    if f0_values:  # Ensure there is data to plot
        fig_f0 = plot_f0_distribution(np.array(f0_values), np.array(vowels_f0), categories=None, title="F0 Distribution by Vowel")
        fig_f0.savefig(os.path.join(output_dir, "f0_distribution.png"))
        plt.close(fig_f0)
    
    # f. Feature Correlation Matrix using selected features
    feature_cols = ['F1_mean', 'F2_mean', 'F3_mean', 'F1_std', 'F2_std', 'F3_std', 'F0_autocorr_mean', 'F0_autocorr_std']
    features_array = train_features_df[feature_cols].to_numpy()
    fig_corr = plot_feature_correlation(features_array, feature_cols, title="Feature Correlation Matrix")
    fig_corr.savefig(os.path.join(output_dir, "feature_correlation.png"))
    plt.close(fig_corr)
    
    # g. Interactive 3D Vowel Space using Plotly
    fig_interactive = plot_interactive_vowel_space(formants_2d, vowels_train, categories=categories_train, title="Interactive Vowel Space")
    fig_interactive.write_html(os.path.join(output_dir, "interactive_vowel_space.html"))
    
    # h. Formant Trajectories for a few training samples (e.g., first 3 samples)
    formants_list = []
    times_list = []
    vowels_traj = []
    for i in range(min(3, len(train_raw_data))):
        if 'formants' in train_raw_data[i]:
            formant_array = train_raw_data[i]['formants']
            formants_list.append(formant_array)
            num_frames = formant_array.shape[0]
            # Assuming a frame shift of 10ms:
            times = np.arange(num_frames) * 0.01  
            times_list.append(times)
            vowels_traj.append(train_features_df.iloc[i]['vowel'])
    if formants_list:
        fig_traj = plot_formant_trajectories(formants_list, times_list, vowels_traj, title="Formant Trajectories")
        fig_traj.savefig(os.path.join(output_dir, "formant_trajectories.png"))
        plt.close(fig_traj)
    
    # i. Hierarchical Clustering Dendrogram of Vowels
    fig_dendro = plot_hierarchical_clustering(formants_2d, vowels_train, categories=categories_train, title="Hierarchical Clustering of Vowels")
    fig_dendro.savefig(os.path.join(output_dir, "hierarchical_clustering.png"))
    plt.close(fig_dendro)
    
    # j. Voronoi Diagram of Vowel Classification Regions
    fig_voronoi = plot_voronoi_vowel_space(formants_2d, vowels_train, title="Voronoi Diagram of Vowel Classification Regions")
    fig_voronoi.savefig(os.path.join(output_dir, "voronoi_vowel_space.png"))
    plt.close(fig_voronoi)
    
    # ----- Step 7: Save features
    # Save training and test feature DataFrames as CSV files
    train_features_df.to_csv(os.path.join(output_dir, "train_features.csv"), index=False)
    test_features_df.to_csv(os.path.join(output_dir, "test_features.csv"), index=False)
    
    # Save the raw feature data (numpy arrays, etc.) as pickle files
    with open(os.path.join(output_dir, "train_raw_data.pkl"), "wb") as f:
        pickle.dump(train_raw_data, f)
    with open(os.path.join(output_dir, "test_raw_data.pkl"), "wb") as f:
        pickle.dump(test_raw_data, f)
    
    print("All outputs have been saved in:", output_dir)
    
    # ----- Step 8: Return data for further processing if needed
    return {
        'train_df': train_df,
        'test_df': test_df,
        'train_features_df': train_features_df,
        'test_features_df': test_features_df,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'train_raw_data': train_raw_data,
        'test_raw_data': test_raw_data
    }

if __name__ == "__main__":
    main()
