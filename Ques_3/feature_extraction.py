import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import pickle
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import Voronoi, voronoi_plot_2d

def split_data(df, test_size=0.2, random_state=45):
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['vowel'])

def extract_formants(signal, sr, order=13):
    pre_emphasis = 0.95
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    frame_length = int(0.025 * sr)  # 25ms
    frame_step = int(0.01 * sr)     # 10ms
    
    frames = []
    for i in range(0, len(emphasized_signal) - frame_length, frame_step):
        frames.append(emphasized_signal[i:i+frame_length])
    
    if not frames:
        return None, None
    
    formants_list = []
    for frame in frames:
        frame = frame * np.hamming(len(frame))
        
        a = librosa.lpc(frame, order=order)
        roots = np.roots(a)
        roots = roots[np.imag(roots) > 0]
        
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (sr / (2 * np.pi))
        
        formants = sorted(freqs)
        if len(formants) >= 3:
            formants_list.append(formants[:3])
    
    if not formants_list:
        return None, None
    
    formants_array = np.array(formants_list)
    return formants_array, frames

def extract_f0(signal, sr):
    frame_length = int(0.025 * sr)
    frame_step = int(0.01 * sr)
    
    frames = []
    for i in range(0, len(signal) - frame_length, frame_step):
        frames.append(signal[i:i+frame_length])
    
    if not frames:
        return None
    
    f0_values = []
    for frame in frames:
        frame = frame * np.hamming(len(frame))
        
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        min_lag = int(sr / 500)  # Max F0: 500Hz
        max_lag = int(sr / 60)   # Min F0: 60Hz
        
        if len(corr) <= max_lag:
            continue
        
        peak_idx = np.argmax(corr[min_lag:max_lag]) + min_lag
        if peak_idx > 0:
            f0 = sr / peak_idx
            f0_values.append(f0)
    
    return np.array(f0_values) if f0_values else None

def process_audio(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=None)
        
        formants_array, frames = extract_formants(signal, sr)
        f0_autocorr = extract_f0(signal, sr)
        
        if formants_array is None or f0_autocorr is None:
            return None
        
        raw_data = {
            'raw_audio': signal,
            'sample_rate': sr,
            'formants': formants_array,
            'f0_autocorr': f0_autocorr
        }
        
        features = {
            'F1_mean': np.mean(formants_array[:, 0]),
            'F2_mean': np.mean(formants_array[:, 1]),
            'F3_mean': np.mean(formants_array[:, 2]),
            'F1_std': np.std(formants_array[:, 0]),
            'F2_std': np.std(formants_array[:, 1]),
            'F3_std': np.std(formants_array[:, 2]),
            'F0_autocorr_mean': np.mean(f0_autocorr),
            'F0_autocorr_std': np.std(f0_autocorr),
            'sample_rate': sr
        }
        
        return features, raw_data
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset(df):
    features_list = []
    raw_data_dict = {}
    
    for idx, row in df.iterrows():
        result = process_audio(row['file_path'])
        
        if result is not None:
            features, raw_data = result
            
            features_row = {
                'vowel': row['vowel'],
                'category': row['category'] if 'category' in row else None,
                **features
            }
            
            features_list.append(features_row)
            raw_data_dict[len(features_list) - 1] = raw_data
    
    features_df = pd.DataFrame(features_list)
    return features_df, raw_data_dict

def normalize_features(features_df, method='z-score'):
    numeric_cols = ['F1_mean', 'F2_mean', 'F3_mean', 'F1_std', 'F2_std', 'F3_std', 
                    'F0_autocorr_mean', 'F0_autocorr_std']
    
    df_norm = features_df.copy()
    
    if method == 'z-score':
        scaler = StandardScaler()
        df_norm[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])
    
    return df_norm

def calculate_speaker_normalization(features_df, method='lobanov'):
    if method == 'lobanov':
        if 'category' in features_df.columns:
            normalized_df = features_df.copy()
            for category in features_df['category'].unique():
                category_mask = features_df['category'] == category
                
                for formant in ['F1_mean', 'F2_mean', 'F3_mean']:
                    values = features_df.loc[category_mask, formant]
                    mean = values.mean()
                    std = values.std()
                    normalized_df.loc[category_mask, formant] = (values - mean) / std
            
            return normalized_df
    
    return features_df

def create_feature_vector(features):
    feature_vector = np.array([
        features['F1_mean'], features['F2_mean'], features['F3_mean'],
        features['F1_std'], features['F2_std'], features['F3_std'],
        features['F0_autocorr_mean'], features['F0_autocorr_std']
    ])
    
    feature_names = [
        'F1_mean', 'F2_mean', 'F3_mean',
        'F1_std', 'F2_std', 'F3_std',
        'F0_autocorr_mean', 'F0_autocorr_std'
    ]
    
    return feature_vector, feature_names

def plot_waveform(signal, sr, title="Waveform"):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(signal)) / sr, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    return fig

def plot_spectrogram(signal, sr, title="Spectrogram"):
    fig = plt.figure(figsize=(10, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    plt.imshow(D, aspect='auto', origin='lower', extent=[0, len(signal)/sr, 0, sr/2])
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    return fig

def plot_lpc_spectrum(signal, sr, order=13, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    pre_emphasis = 0.95
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    frame_length = int(0.025 * sr)
    frame = emphasized_signal[:frame_length] * np.hamming(frame_length)
    
    a = librosa.lpc(frame, order=order)
    
    freqs = np.linspace(0, sr/2, 512)
    angles = 2 * np.pi * freqs / sr
    
    h = np.zeros(angles.shape, dtype=complex)
    for i, angle in enumerate(angles):
        h[i] = 1 / np.sum(a * np.exp(-1j * angle * np.arange(len(a))))
    
    ax.plot(freqs, 20 * np.log10(np.abs(h)))
    
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2 * np.pi))
    
    formants = sorted(freqs)
    
    for formant in formants[:3]:
        ax.axvline(x=formant, color='r', linestyle='--', alpha=0.5)
        ax.text(formant + 50, ax.get_ylim()[0] + 5, f"{formant:.0f} Hz", rotation=90)
    
    ax.set_title("LPC Spectrum with Formants")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, 4000)
    
    return ax.figure

def plot_vowel_space_2d(formants, vowels, categories=None, title="Vowel Space (F1-F2)"):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Reverse axes for phonetic convention (F1 increasing downward, F2 increasing leftward)
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    vowel_colors = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    
    if categories is not None:
        markers = {'Adult_Male': 'o', 'Adult_Female': 's', '7yo_Child': '^', 
                   '5yo_Child': 'D', '3yo_Child': 'p'}
        
        for vowel in np.unique(vowels):
            for category in np.unique(categories):
                mask = (vowels == vowel) & (categories == category)
                if np.any(mask):
                    ax.scatter(formants[mask, 1], formants[mask, 0], 
                              color=vowel_colors.get(vowel, 'gray'),
                              marker=markers.get(category, 'o'),
                              label=f"{vowel} - {category}")
    else:
        for vowel in np.unique(vowels):
            mask = vowels == vowel
            ax.scatter(formants[mask, 1], formants[mask, 0], 
                      color=vowel_colors.get(vowel, 'gray'),
                      label=vowel)
    
    # Add vowel labels at centroids
    for vowel in np.unique(vowels):
        mask = vowels == vowel
        centroid_x = np.mean(formants[mask, 1])
        centroid_y = np.mean(formants[mask, 0])
        ax.text(centroid_x, centroid_y, vowel, fontsize=16, fontweight='bold')
    
    plt.title(title)
    plt.tight_layout()
    
    # Add legend for first few elements if there are many categories
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 10:
        ax.legend(handles[:10], labels[:10], loc='upper right', title="Sample Legend")
    else:
        ax.legend(loc='upper right')
    
    return fig

def plot_vowel_space_3d(formants, vowels, categories=None, title="Vowel Space (F1-F2-F3)"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    vowel_colors = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    
    if categories is not None:
        markers = {'Adult_Male': 'o', 'Adult_Female': 's', '7yo_Child': '^', 
                   '5yo_Child': 'D', '3yo_Child': 'p'}
        
        for vowel in np.unique(vowels):
            for category in np.unique(categories):
                mask = (vowels == vowel) & (categories == category)
                if np.any(mask):
                    ax.scatter(formants[mask, 1], formants[mask, 0], formants[mask, 2], 
                              color=vowel_colors.get(vowel, 'gray'),
                              marker=markers.get(category, 'o'),
                              label=f"{vowel} - {category}")
    else:
        for vowel in np.unique(vowels):
            mask = vowels == vowel
            ax.scatter(formants[mask, 1], formants[mask, 0], formants[mask, 2], 
                      color=vowel_colors.get(vowel, 'gray'),
                      label=vowel)
    
    # Add vowel labels at centroids
    for vowel in np.unique(vowels):
        mask = vowels == vowel
        centroid_x = np.mean(formants[mask, 1])
        centroid_y = np.mean(formants[mask, 0])
        centroid_z = np.mean(formants[mask, 2])
        ax.text(centroid_x, centroid_y, centroid_z, vowel, fontsize=16, fontweight='bold')
    
    # Reverse x and y axes for phonetic convention
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    ax.set_xlabel('F2 (Hz)')
    ax.set_ylabel('F1 (Hz)')
    ax.set_zlabel('F3 (Hz)')
    ax.set_title(title)
    
    # Add legend for first few elements if there are many categories
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 10:
        ax.legend(handles[:10], labels[:10], loc='upper right', title="Sample Legend")
    else:
        ax.legend(loc='upper right')
    
    return fig

def plot_feature_correlation(features, feature_names, title="Feature Correlation Matrix"):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = np.corrcoef(features.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", 
                    ha="center", va="center", 
                    color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig

def plot_interactive_vowel_space(formants, vowels, categories=None, title="Interactive Vowel Space"):
    df = pd.DataFrame({
        'F1': formants[:, 0],
        'F2': formants[:, 1],
        'F3': formants[:, 2],
        'Vowel': vowels
    })
    
    if categories is not None:
        df['Category'] = categories
        fig = px.scatter_3d(df, x='F2', y='F1', z='F3', color='Vowel', symbol='Category',
                          title=title, labels={'F1': 'F1 (Hz)', 'F2': 'F2 (Hz)', 'F3': 'F3 (Hz)'})
    else:
        fig = px.scatter_3d(df, x='F2', y='F1', z='F3', color='Vowel',
                          title=title, labels={'F1': 'F1 (Hz)', 'F2': 'F2 (Hz)', 'F3': 'F3 (Hz)'})
    
    # Reverse axes for phonetic convention
    fig.update_layout(scene=dict(
        xaxis=dict(autorange="reversed"),
        yaxis=dict(autorange="reversed")
    ))
    
    return fig

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    return fig

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import pickle
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import Voronoi, voronoi_plot_2d
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings
warnings.filterwarnings("ignore")

def split_data(df, test_size=0.2, random_state=45):
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['vowel'])

def extract_formants(signal, sr, order=13, preemphasis=0.95, num_formants=3):
    # Apply pre-emphasis filter to enhance higher frequencies
    emphasized_signal = np.append(signal[0], signal[1:] - preemphasis * signal[:-1])
    
    # Frame the signal
    frame_length = int(0.025 * sr)  # 25ms
    frame_step = int(0.01 * sr)     # 10ms
    
    frames = []
    for i in range(0, len(emphasized_signal) - frame_length, frame_step):
        frames.append(emphasized_signal[i:i+frame_length])
    
    if not frames:
        return None, None
    
    formants_list = []
    for frame in frames:
        # Apply Hamming window
        frame = frame * np.hamming(len(frame))
        
        # LPC analysis
        a = librosa.lpc(frame, order=order)
        roots = np.roots(a)
        
        # Keep only roots with positive imaginary part (that's half of the poles)
        roots = roots[np.imag(roots) > 0]
        
        # Convert from angular frequency to Hz
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (sr / (2 * np.pi))
        
        # Sort and take only the first few as formants
        formants = sorted(freqs)
        if len(formants) >= num_formants:
            formants_list.append(formants[:num_formants])
    
    if not formants_list:
        return None, None
    
    formants_array = np.array(formants_list)
    return formants_array, frames

def extract_f0(signal, sr, min_freq=60, max_freq=500, method='autocorr'):
    frame_length = int(0.025 * sr)
    frame_step = int(0.01 * sr)
    
    frames = []
    for i in range(0, len(signal) - frame_length, frame_step):
        frames.append(signal[i:i+frame_length])
    
    if not frames:
        return None
    
    f0_values = []
    
    if method == 'autocorr':
        for frame in frames:
            frame = frame * np.hamming(len(frame))
            
            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            
            min_lag = int(sr / max_freq)  # Max F0: 500Hz by default
            max_lag = int(sr / min_freq)   # Min F0: 60Hz by default
            
            if len(corr) <= max_lag:
                continue
            
            # Find the highest peak after the first minimum
            # First find the first minimum after the zero lag
            dips = np.where(np.diff(corr[:min(len(corr), max_lag)]) > 0)[0]
            if len(dips) > 0:
                first_dip = dips[0]
                peak_idx = first_dip + np.argmax(corr[first_dip:max_lag])
            else:
                peak_idx = np.argmax(corr[min_lag:max_lag]) + min_lag
            
            if peak_idx > 0:
                f0 = sr / peak_idx
                # Only consider reasonable F0 values
                if min_freq <= f0 <= max_freq:
                    f0_values.append(f0)
    
    elif method == 'yin':
        # Implement YIN algorithm for better F0 estimation
        for frame in frames:
            frame = frame * np.hamming(len(frame))
            # Step 1: Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            
            # Step 2: Difference function
            length = len(frame)
            diff = np.zeros(length)
            for tau in range(length):
                for j in range(length-tau):
                    diff[tau] += (frame[j] - frame[j+tau])**2
            
            # Step 3: Cumulative normalization
            cum = np.zeros(length)
            cum[0] = 1.0
            for tau in range(1, length):
                cum[tau] = diff[tau] / (np.sum(diff[1:tau+1]) / tau)
            
            # Step 4: Absolute threshold
            thresh = 0.1
            tau = np.where(cum < thresh)[0]
            if len(tau) > 0:
                tau = tau[0]
                # Parabolic interpolation for better accuracy
                if tau > 0 and tau < length-1:
                    a = cum[tau-1]
                    b = cum[tau]
                    c = cum[tau+1]
                    tau = tau + 0.5 * (a - c) / (a - 2*b + c)
                
                f0 = sr / tau
                if min_freq <= f0 <= max_freq:
                    f0_values.append(f0)
    
    return np.array(f0_values) if f0_values else None

def extract_mfcc(signal, sr, n_mfcc=13):
    # Extract MFCC features for additional spectral shape information
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    return mfcc_means, mfcc_stds

def extract_spectral_features(signal, sr):
    # Extract additional spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    
    return {
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_std': np.std(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_bandwidth_std': np.std(spectral_bandwidth),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'spectral_rolloff_std': np.std(spectral_rolloff)
    }

def process_audio(file_path, use_advanced_features=True):
    try:
        signal, sr = librosa.load(file_path, sr=None)
        
        # Apply noise reduction if needed
        if use_advanced_features:
            # Simple noise reduction by spectral gating
            noise_frame = signal[:int(0.1*sr)]  # Use first 100ms as noise profile
            noise_power = np.mean(noise_frame**2)
            signal_power = np.mean(signal**2)
            
            # Only apply noise reduction if the signal-to-noise ratio is low
            if signal_power / noise_power < 20:
                noise_threshold = 2 * np.mean(np.abs(noise_frame))
                signal = np.where(np.abs(signal) < noise_threshold, 0, signal)
        
        # Ensure signal is of reasonable length
        if len(signal) < 0.1 * sr:
            print(f"Warning: Signal in {file_path} is too short.")
            return None
        
        # Extract formants with adjustable parameters
        formants_array, frames = extract_formants(signal, sr, order=16, preemphasis=0.97, num_formants=3)
        if formants_array is None:
            print(f"Warning: Could not extract formants from {file_path}")
            return None
        
        # More adaptive F0 extraction based on expected range
        is_child = 'child' in file_path.lower()
        min_f0 = 100 if is_child else 60
        max_f0 = 600 if is_child else 400
        
        # Try both F0 extraction methods
        f0_autocorr = extract_f0(signal, sr, min_freq=min_f0, max_freq=max_f0, method='autocorr')
        f0_yin = extract_f0(signal, sr, min_freq=min_f0, max_freq=max_f0, method='yin')
        
        # Pick the one with more valid values
        f0_values = f0_yin if (f0_yin is not None and (f0_autocorr is None or len(f0_yin) > len(f0_autocorr))) else f0_autocorr
        
        if f0_values is None or len(f0_values) < 3:
            print(f"Warning: Could not extract F0 from {file_path}")
            return None
        
        # Basic features
        features = {
            'F1_mean': np.mean(formants_array[:, 0]),
            'F2_mean': np.mean(formants_array[:, 1]),
            'F3_mean': np.mean(formants_array[:, 2]),
            'F1_std': np.std(formants_array[:, 0]),
            'F2_std': np.std(formants_array[:, 1]),
            'F3_std': np.std(formants_array[:, 2]),
            'F1_min': np.min(formants_array[:, 0]),
            'F2_min': np.min(formants_array[:, 1]),
            'F3_min': np.min(formants_array[:, 2]),
            'F1_max': np.max(formants_array[:, 0]),
            'F2_max': np.max(formants_array[:, 1]),
            'F3_max': np.max(formants_array[:, 2]),
            'F0_mean': np.mean(f0_values),
            'F0_std': np.std(f0_values),
            'F0_min': np.min(f0_values),
            'F0_max': np.max(f0_values),
            'F0_range': np.max(f0_values) - np.min(f0_values),
            'duration': len(signal) / sr,
            'sample_rate': sr
        }
        
        # Advanced features
        if use_advanced_features:
            # Extract MFCCs
            mfcc_means, mfcc_stds = extract_mfcc(signal, sr, n_mfcc=13)
            for i in range(len(mfcc_means)):
                features[f'mfcc{i+1}_mean'] = mfcc_means[i]
                features[f'mfcc{i+1}_std'] = mfcc_stds[i]
            
            # Extract spectral features
            spectral_features = extract_spectral_features(signal, sr)
            features.update(spectral_features)
            
            # Add formant ratios (useful for vowel discrimination)
            features['F2_F1_ratio'] = features['F2_mean'] / features['F1_mean']
            features['F3_F2_ratio'] = features['F3_mean'] / features['F2_mean']
            features['F3_F1_ratio'] = features['F3_mean'] / features['F1_mean']
            
            # Formant dispersion (correlated with vocal tract length)
            features['formant_dispersion'] = (features['F3_mean'] - features['F1_mean']) / 2
        
        raw_data = {
            'raw_audio': signal,
            'sample_rate': sr,
            'formants': formants_array,
            'f0_values': f0_values
        }
        
        return features, raw_data
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset(df, use_advanced_features=True):
    features_list = []
    raw_data_dict = {}
    
    start_time = time.time()
    total_files = len(df)
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            files_per_sec = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_files - idx - 1) / files_per_sec if files_per_sec > 0 else 0
            print(f"Processing file {idx+1}/{total_files} - ETA: {remaining:.1f}s")
        
        result = process_audio(row['file_path'], use_advanced_features)
        
        if result is not None:
            features, raw_data = result
            
            features_row = {
                'vowel': row['vowel'],
                'category': row['category'] if 'category' in row else None,
                **features
            }
            
            features_list.append(features_row)
            raw_data_dict[len(features_list) - 1] = raw_data
    
    print(f"Successfully processed {len(features_list)}/{total_files} files")
    features_df = pd.DataFrame(features_list)
    return features_df, raw_data_dict

def normalize_features(features_df, method='z-score', by_category=False):
    # Get all numeric columns
    numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Remove non-feature columns
    for col in ['vowel', 'category', 'sample_rate']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    
    df_norm = features_df.copy()
    
    if by_category and 'category' in features_df.columns:
        # Normalize separately for each category (useful for speaker normalization)
        for category in features_df['category'].unique():
            mask = features_df['category'] == category
            if method == 'z-score':
                scaler = StandardScaler()
                if sum(mask) > 1:  # Need at least 2 samples for std calculation
                    df_norm.loc[mask, numeric_cols] = scaler.fit_transform(features_df.loc[mask, numeric_cols])
            elif method == 'minmax':
                scaler = MinMaxScaler()
                df_norm.loc[mask, numeric_cols] = scaler.fit_transform(features_df.loc[mask, numeric_cols])
    else:
        # Global normalization
        if method == 'z-score':
            scaler = StandardScaler()
            df_norm[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df_norm[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])
    
    return df_norm

def calculate_speaker_normalization(features_df, method='lobanov'):
    if method == 'lobanov':
        # Lobanov normalization: (F - mean) / std for each speaker/category
        if 'category' in features_df.columns:
            normalized_df = features_df.copy()
            for category in features_df['category'].unique():
                category_mask = features_df['category'] == category
                
                for formant in ['F1_mean', 'F2_mean', 'F3_mean']:
                    if formant in features_df.columns:
                        values = features_df.loc[category_mask, formant]
                        mean = values.mean()
                        std = values.std()
                        if std > 0:  # Avoid division by zero
                            normalized_df.loc[category_mask, formant] = (values - mean) / std
            
            return normalized_df
    
    elif method == 'nearey':
        # Nearey normalization: log(F) - mean(log(F)) for each speaker/category
        if 'category' in features_df.columns:
            normalized_df = features_df.copy()
            for category in features_df['category'].unique():
                category_mask = features_df['category'] == category
                
                for formant in ['F1_mean', 'F2_mean', 'F3_mean']:
                    if formant in features_df.columns:
                        values = np.log(features_df.loc[category_mask, formant])
                        mean = values.mean()
                        normalized_df.loc[category_mask, formant] = np.exp(values - mean)
            
            return normalized_df
    
    elif method == 'wattfabricius':
        # Watt-Fabricius normalization
        # Specific to vowel spaces, more complex implementation
        if 'category' in features_df.columns and 'vowel' in features_df.columns:
            normalized_df = features_df.copy()
            for category in features_df['category'].unique():
                category_mask = features_df['category'] == category
                
                # Find reference vowels for normalization (typically /i/, /a/, /u/)
                vowel_means = {}
                for vowel in ['i', 'a', 'u']:
                    vowel_mask = (features_df['category'] == category) & (features_df['vowel'] == vowel)
                    if sum(vowel_mask) > 0:
                        vowel_means[vowel] = {
                            'F1': features_df.loc[vowel_mask, 'F1_mean'].mean(),
                            'F2': features_df.loc[vowel_mask, 'F2_mean'].mean()
                        }
                
                # Need at least /i/ and /a/ for the normalization
                if 'i' in vowel_means and 'a' in vowel_means:
                    # Calculate centroid if we have three reference vowels
                    if 'u' in vowel_means:
                        centroid_F1 = (vowel_means['i']['F1'] + vowel_means['a']['F1'] + vowel_means['u']['F1']) / 3
                        centroid_F2 = (vowel_means['i']['F2'] + vowel_means['a']['F2'] + vowel_means['u']['F2']) / 3
                    else:
                        centroid_F1 = (vowel_means['i']['F1'] + vowel_means['a']['F1']) / 2
                        centroid_F2 = (vowel_means['i']['F2'] + vowel_means['a']['F2']) / 2
                    
                    # Define reference points
                    ref_F1 = vowel_means['a']['F1']
                    ref_F2 = vowel_means['i']['F2']
                    
                    # Normalize
                    for formant in ['F1_mean', 'F2_mean']:
                        base_formant = formant[:2]  # 'F1' or 'F2'
                        if base_formant == 'F1':
                            normalized_df.loc[category_mask, formant] = features_df.loc[category_mask, formant] / ref_F1
                        elif base_formant == 'F2':
                            normalized_df.loc[category_mask, formant] = features_df.loc[category_mask, formant] / ref_F2
            
            return normalized_df
    
    return features_df

def create_feature_vector(features, feature_set='all'):
    basic_features = [
        'F1_mean', 'F2_mean', 'F3_mean',
        'F1_std', 'F2_std', 'F3_std',
        'F0_mean', 'F0_std'
    ]
    
    extended_features = [
        'F1_min', 'F2_min', 'F3_min',
        'F1_max', 'F2_max', 'F3_max',
        'F0_min', 'F0_max', 'F0_range',
        'F2_F1_ratio', 'F3_F2_ratio', 'F3_F1_ratio',
        'formant_dispersion'
    ]
    
    spectral_features = [
        'spectral_centroid_mean', 'spectral_centroid_std',
        'spectral_bandwidth_mean', 'spectral_bandwidth_std',
        'spectral_rolloff_mean', 'spectral_rolloff_std'
    ]
    
    mfcc_features = [f'mfcc{i+1}_mean' for i in range(13)] + [f'mfcc{i+1}_std' for i in range(13)]
    
    all_feature_names = {
        'basic': basic_features,
        'extended': basic_features + extended_features,
        'spectral': basic_features + extended_features + spectral_features,
        'mfcc': basic_features + mfcc_features,
        'all': basic_features + extended_features + spectral_features + mfcc_features
    }
    
    selected_features = all_feature_names.get(feature_set, all_feature_names['basic'])
    
    # Filter out features that don't exist in the input
    available_features = [f for f in selected_features if f in features]
    feature_vector = np.array([features[f] for f in available_features])
    
    return feature_vector, available_features

def select_best_features(X_train, y_train, feature_names, n_features=10):
    from sklearn.feature_selection import SelectKBest, f_classif
    
    selector = SelectKBest(f_classif, k=n_features)
    X_selected = selector.fit_transform(X_train, y_train)
    
    # Get scores and indices of selected features
    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Sort features by importance
    feature_importance = [(selected_features[i], scores[selected_indices[i]]) 
                          for i in range(len(selected_features))]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    return X_selected, selected_features, feature_importance

def perform_hyperparameter_tuning(X_train, y_train, classifier_type='knn', cv=5):
    if classifier_type == 'knn':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        clf = KNeighborsClassifier()
    
    elif classifier_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        clf = SVC(probability=True)
    
    elif classifier_type == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        clf = RandomForestClassifier(random_state=45)
    
    elif classifier_type == 'gb':
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        clf = GradientBoostingClassifier(random_state=45)
    
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    # Use stratified k-fold CV to handle class imbalance
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=45)
    
    # Perform grid search
    grid_search = GridSearchCV(
        clf, param_grid, cv=cv_splitter, 
        scoring='f1_macro',  # Use macro F1-score for imbalanced classes
        n_jobs=-1,  # Use all available processors
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {classifier_type}: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def train_ensemble_model(X_train, y_train, models, voting='soft'):
    from sklearn.ensemble import VotingClassifier
    
    # Create a voting classifier
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting=voting
    )
    
    ensemble.fit(X_train, y_train)
    return ensemble

def plot_waveform(signal, sr, title="Waveform"):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(signal)) / sr, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    return fig

def plot_spectrogram(signal, sr, title="Spectrogram"):
    fig = plt.figure(figsize=(10, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    plt.imshow(D, aspect='auto', origin='lower', extent=[0, len(signal)/sr, 0, sr/2])
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    return fig

def plot_lpc_spectrum(signal, sr, order=13, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    pre_emphasis = 0.95
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    frame_length = int(0.025 * sr)
    frame = emphasized_signal[:frame_length] * np.hamming(frame_length)
    
    a = librosa.lpc(frame, order=order)
    
    freqs = np.linspace(0, sr/2, 512)
    angles = 2 * np.pi * freqs / sr
    
    h = np.zeros(angles.shape, dtype=complex)
    for i, angle in enumerate(angles):
        h[i] = 1 / np.sum(a * np.exp(-1j * angle * np.arange(len(a))))
    
    ax.plot(freqs, 20 * np.log10(np.abs(h)))
    
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2 * np.pi))
    
    formants = sorted(freqs)
    
    for formant in formants[:3]:
        ax.axvline(x=formant, color='r', linestyle='--', alpha=0.5)
        ax.text(formant + 50, ax.get_ylim()[0] + 5, f"{formant:.0f} Hz", rotation=90)
    
    ax.set_title("LPC Spectrum with Formants")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, 4000)
    
    return ax.figure

def plot_vowel_space_2d(formants, vowels, categories=None, title="Vowel Space (F1-F2)"):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Reverse axes for phonetic convention (F1 increasing downward, F2 increasing leftward)
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    vowel_colors = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    
    if categories is not None:
        markers = {'Adult_Male': 'o', 'Adult_Female': 's', '7yo_Child': '^', 
                   '5yo_Child': 'D', '3yo_Child': 'p'}
        
        for vowel in np.unique(vowels):
            for category in np.unique(categories):
                mask = (vowels == vowel) & (categories == category)
                if np.any(mask):
                    ax.scatter(formants[mask, 1], formants[mask, 0], 
                              color=vowel_colors.get(vowel, 'gray'),
                              marker=markers.get(category, 'o'),
                              label=f"{vowel} - {category}")
    else:
        for vowel in np.unique(vowels):
            mask = vowels == vowel
            ax.scatter(formants[mask, 1], formants[mask, 0], 
                      color=vowel_colors.get(vowel, 'gray'),
                      label=vowel)
    
    # Optionally, add a legend or additional formatting here
    ax.legend(loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    
    # Add return statement to pass the figure to the caller
    return fig

    
def plot_vowel_space_3d(formants, vowels, categories=None, title="Vowel Space (F1-F2-F3)"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Reverse axes for phonetic convention (F1 increasing downward, F2 increasing leftward)
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.set_zlabel("F3 (Hz)")
    
    vowel_colors = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    
    if categories is not None:
        markers = {'Adult_Male': 'o', 'Adult_Female': 's', '7yo_Child': '^', 
                   '5yo_Child': 'D', '3yo_Child': 'p'}
        
        for vowel in np.unique(vowels):
            for category in np.unique(categories):
                mask = (vowels == vowel) & (categories == category)
                if np.any(mask):
                    ax.scatter(formants[mask, 1], formants[mask, 0], formants[mask, 2],
                              color=vowel_colors.get(vowel, 'gray'),
                              marker=markers.get(category, 'o'),
                              label=f"{vowel} - {category}")
    else:
        for vowel in np.unique(vowels):
            mask = vowels == vowel
            ax.scatter(formants[mask, 1], formants[mask, 0], formants[mask, 2],
                      color=vowel_colors.get(vowel, 'gray'),
                      label=vowel)
    
    # Add a legend with only one entry per vowel and per category
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    seen_labels = set()
    for handle, label in zip(handles, labels):
        if label not in seen_labels:
            seen_labels.add(label)
            unique_labels.append(label)
            unique_handles.append(handle)
    
    # Place the legend outside the plot to avoid overcrowding
    ax.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Set the title
    ax.set_title(title)
    
    # Create centroids for each vowel (average formant values)
    for vowel in np.unique(vowels):
        mask = vowels == vowel
        if np.any(mask):
            centroid = np.mean(formants[mask], axis=0)
            ax.text(centroid[1], centroid[0], centroid[2], vowel.upper(), 
                   fontsize=14, fontweight='bold', color=vowel_colors.get(vowel, 'gray'))
    
    plt.tight_layout()
    return fig

def plot_feature_correlation(features_array, feature_names, title="Feature Correlation Matrix"):
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(features_array.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(feature_names)
    
    # Loop over data dimensions and create text annotations
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                          ha="center", va="center", 
                          color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
    
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    fig.tight_layout()
    return fig

# Advanced analysis functions 
def run_advanced_analysis(train_features_df, train_raw_data, test_features_df, test_raw_data, X_train, y_train, X_test, y_test, feature_names, output_dir):
    """
    Perform advanced analysis on the vowel classification data
    """
    print("Running advanced analysis...")
    
    # 1. Dimension Reduction with PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Plot PCA results
    fig, ax = plt.subplots(figsize=(10, 8))
    vowel_colors = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    
    for vowel in np.unique(y_train):
        mask = np.array(y_train) == vowel
        ax.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], 
                 color=vowel_colors.get(vowel, 'gray'),
                 label=vowel)
    
    ax.set_title("PCA Projection of Vowel Features")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.legend()
    
    fig.savefig(os.path.join(output_dir, "pca_projection.png"))
    plt.close(fig)
    
    # 2. Feature importance analysis
    X_selected, selected_features, feature_importance = select_best_features(X_train, y_train, feature_names)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(12, 6))
    features = [f[0] for f in feature_importance]
    scores = [f[1] for f in feature_importance]
    
    ax.bar(features, scores)
    ax.set_title("Feature Importance for Vowel Classification")
    ax.set_xlabel("Features")
    ax.set_ylabel("F-score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    fig.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close(fig)
    
    # 3. Hyperparameter tuning for multiple classifiers
    best_models = {}
    for clf_type in ['knn', 'svm', 'rf']:
        print(f"Tuning {clf_type}...")
        best_model, best_params, best_score = perform_hyperparameter_tuning(
            X_train, y_train, classifier_type=clf_type, cv=5
        )
        best_models[clf_type] = best_model
        
        # Save best parameters
        with open(os.path.join(output_dir, f"{clf_type}_best_params.txt"), "w") as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best CV score: {best_score:.4f}\n")
    
    # 4. Ensemble model
    ensemble = train_ensemble_model(X_train, y_train, best_models)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred))
    
    # Save ensemble results
    with open(os.path.join(output_dir, "ensemble_results.txt"), "w") as f:
        f.write(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, ensemble_pred))
    
    # 5. T-SNE visualization for better cluster separation
    tsne = TSNE(n_components=2, random_state=42)
    X_train_tsne = tsne.fit_transform(X_train)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for vowel in np.unique(y_train):
        mask = np.array(y_train) == vowel
        ax.scatter(X_train_tsne[mask, 0], X_train_tsne[mask, 1], 
                 color=vowel_colors.get(vowel, 'gray'),
                 label=vowel)
    
    ax.set_title("t-SNE Projection of Vowel Features")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend()
    
    fig.savefig(os.path.join(output_dir, "tsne_projection.png"))
    plt.close(fig)
    
    # 6. Plot sample waveform and LPC spectrum for each vowel
    for vowel in np.unique(y_train):
        # Find the first occurrence of this vowel in the training set
        sample_idx = train_features_df[train_features_df['vowel'] == vowel].index[0]
        sample_raw = train_raw_data[sample_idx]['raw_audio']
        sample_sr = train_features_df.iloc[sample_idx]['sample_rate']
        
        # Plot waveform
        fig_wave = plot_waveform(sample_raw, sample_sr, title=f"Waveform - Vowel '{vowel}'")
        fig_wave.savefig(os.path.join(output_dir, f"waveform_vowel_{vowel}.png"))
        plt.close(fig_wave)
        
        # Plot spectrogram
        fig_spec = plot_spectrogram(sample_raw, sample_sr, title=f"Spectrogram - Vowel '{vowel}'")
        fig_spec.savefig(os.path.join(output_dir, f"spectrogram_vowel_{vowel}.png"))
        plt.close(fig_spec)
        
        # Plot LPC spectrum
        fig_lpc, ax_lpc = plt.subplots(figsize=(10, 6))
        plot_lpc_spectrum(sample_raw, sample_sr, order=16, ax=ax_lpc)
        ax_lpc.set_title(f"LPC Spectrum - Vowel '{vowel}'")
        fig_lpc.savefig(os.path.join(output_dir, f"lpc_spectrum_vowel_{vowel}.png"))
        plt.close(fig_lpc)
    
    return {
        'best_models': best_models,
        'ensemble': ensemble,
        'ensemble_accuracy': ensemble_accuracy,
        'ensemble_pred': ensemble_pred,
        'feature_importance': feature_importance,
        'selected_features': selected_features
    }

# Update the main function to include the advanced analysis
def main():
    # Define paths
    output_dir = './vowel_classification_output'
    os.makedirs(output_dir, exist_ok=True)
    
    data_csv_path = "./Ques_3.csv"
    df = pd.read_csv(data_csv_path)
    
    # Split data
    train_df, test_df = split_data(df, test_size=0.2, random_state=45)
    
    print("Extracting features from training set...")
    train_features_df, train_raw_data = process_dataset(train_df)
    
    print("Extracting features from test set...")
    test_features_df, test_raw_data = process_dataset(test_df)
    
    # Feature normalization
    train_features_norm = normalize_features(train_features_df, method='z-score')
    test_features_norm = normalize_features(test_features_df, method='z-score')
    
    # Speaker normalization (Lobanov)
    train_features_norm = calculate_speaker_normalization(train_features_norm, method='lobanov')
    test_features_norm = calculate_speaker_normalization(test_features_norm, method='lobanov')
    
    # Create feature vectors
    X_train = []
    y_train = []
    feature_names = None
    for i, row in train_features_df.iterrows():
        features = {col: row[col] for col in train_features_df.columns if col not in ['vowel', 'category']}
        feature_vector, feature_names = create_feature_vector(features, feature_set='all')
        X_train.append(feature_vector)
        y_train.append(row['vowel'])
    
    X_test = []
    y_test = []
    for i, row in test_features_df.iterrows():
        features = {col: row[col] for col in test_features_df.columns if col not in ['vowel', 'category']}
        feature_vector, _ = create_feature_vector(features, feature_set='all')
        X_test.append(feature_vector)
        y_test.append(row['vowel'])
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # Train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate key visualizations
    
    # 1. Vowel space visualization (F1-F2)
    formants_2d = train_features_df[['F1_mean', 'F2_mean', 'F3_mean']].to_numpy()
    vowels_train = train_features_df['vowel'].to_numpy()
    categories_train = train_features_df['category'].to_numpy() if 'category' in train_features_df.columns else None
    
    fig_vowel_space_2d = plot_vowel_space_2d(formants_2d, vowels_train, categories=categories_train, 
                                           title="2D Vowel Space (F1-F2)")
    fig_vowel_space_2d.savefig(os.path.join(output_dir, "vowel_space_2d.png"))
    plt.close(fig_vowel_space_2d)
    
    # 2. 3D Vowel Space (F1-F2-F3)
    fig_vowel_space_3d = plot_vowel_space_3d(formants_2d, vowels_train, categories=categories_train, 
                                           title="3D Vowel Space (F1-F2-F3)")
    fig_vowel_space_3d.savefig(os.path.join(output_dir, "vowel_space_3d.png"))
    plt.close(fig_vowel_space_3d)
    
    # 3. Sample audio spectrogram
    sample_idx = 0
    sample_raw = train_raw_data[sample_idx]['raw_audio']
    sample_sr = train_features_df.iloc[sample_idx]['sample_rate']
    
    fig_spectrogram = plot_spectrogram(sample_raw, sample_sr, title="Spectrogram Sample")
    fig_spectrogram.savefig(os.path.join(output_dir, "spectrogram_sample.png"))
    plt.close(fig_spectrogram)
    
    # 4. Feature correlation matrix
    feature_cols = ['F1_mean', 'F2_mean', 'F3_mean', 'F1_std', 'F2_std', 'F3_std']
    if 'F0_mean' in train_features_df.columns:
        feature_cols.extend(['F0_mean', 'F0_std'])
    elif 'F0_autocorr_mean' in train_features_df.columns:
        feature_cols.extend(['F0_autocorr_mean', 'F0_autocorr_std'])
    
    features_array = train_features_df[feature_cols].to_numpy()
    
    fig_corr = plot_feature_correlation(features_array, feature_cols, title="Feature Correlation Matrix")
    fig_corr.savefig(os.path.join(output_dir, "feature_correlation.png"))
    plt.close(fig_corr)
    
    # 5. Confusion matrix for classification results
    fig_cm = plot_confusion_matrix(y_test, y_pred, title="Vowel Classification Confusion Matrix")
    fig_cm.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(fig_cm)
    
    # Save results
    train_features_df.to_csv(os.path.join(output_dir, "train_features.csv"), index=False)
    test_features_df.to_csv(os.path.join(output_dir, "test_features.csv"), index=False)
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    with open(os.path.join(output_dir, "classification_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    print("All outputs have been saved in:", output_dir)
    
    # Run advanced analysis
    advanced_results = run_advanced_analysis(
        train_features_df, train_raw_data, 
        test_features_df, test_raw_data,
        X_train, y_train, X_test, y_test, 
        feature_names, output_dir
    )
    
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
        'accuracy': accuracy,
        'y_pred': y_pred,
        'advanced_results': advanced_results
    }

# Function to create an interactive visualization dashboard
def create_interactive_dashboard(results, output_dir):
    """
    Create an interactive HTML dashboard with Plotly visualizations
    """
    train_features_df = results['train_features_df']
    
    # Create directory for interactive visualizations
    interactive_dir = os.path.join(output_dir, 'interactive')
    os.makedirs(interactive_dir, exist_ok=True)
    
    # 1. Interactive 3D vowel space
    fig_3d = px.scatter_3d(
        train_features_df, 
        x='F2_mean', y='F1_mean', z='F3_mean',
        color='vowel',
        hover_name='vowel',
        labels={'F1_mean': 'F1 (Hz)', 'F2_mean': 'F2 (Hz)', 'F3_mean': 'F3 (Hz)'},
        title='3D Vowel Space'
    )
    
    # Customize the layout
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='F2 (Hz)',
            yaxis_title='F1 (Hz)',
            zaxis_title='F3 (Hz)',
            xaxis=dict(autorange='reversed'),
            yaxis=dict(autorange='reversed')
        )
    )
    
    fig_3d.write_html(os.path.join(interactive_dir, 'vowel_space_3d_interactive.html'))
    
    # 2. Interactive 2D vowel space with F1-F2
    fig_2d = px.scatter(
        train_features_df,
        x='F2_mean', y='F1_mean',
        color='vowel',
        hover_name='vowel',
        labels={'F1_mean': 'F1 (Hz)', 'F2_mean': 'F2 (Hz)'},
        title='2D Vowel Space (F1-F2)'
    )
    
    # Customize the layout
    fig_2d.update_layout(
        xaxis=dict(autorange='reversed'),
        yaxis=dict(autorange='reversed')
    )
    
    fig_2d.write_html(os.path.join(interactive_dir, 'vowel_space_2d_interactive.html'))
    
    # 3. Feature correlation heatmap
    feature_cols = [col for col in train_features_df.columns 
                   if col.startswith(('F0_', 'F1_', 'F2_', 'F3_', 'formant_', 'spec'))]
    
    corr_matrix = train_features_df[feature_cols].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Heatmap'
    )
    
    fig_heatmap.write_html(os.path.join(interactive_dir, 'feature_correlation_interactive.html'))
    
    # 4. Confusion matrix
    cm = confusion_matrix(results['y_test'], results['y_pred'])
    cm_df = pd.DataFrame(cm, 
                        index=np.unique(results['y_test']), 
                        columns=np.unique(results['y_test']))
    
    fig_cm = px.imshow(
        cm_df,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=cm_df.columns,
        y=cm_df.index,
        text_auto=True,
        color_continuous_scale='Blues',
        title='Confusion Matrix'
    )
    
    fig_cm.write_html(os.path.join(interactive_dir, 'confusion_matrix_interactive.html'))
    
    # Create a simple index HTML page that links to all visualizations
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vowel Classification Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
            h2 { color: #3498db; }
            a { display: block; margin: 10px 0; color: #2980b9; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Vowel Classification Dashboard</h1>
        <div class="container">
            <div class="card">
                <h2>Vowel Space Visualizations</h2>
                <a href="vowel_space_2d_interactive.html" target="_blank">2D Vowel Space (F1-F2)</a>
                <a href="vowel_space_3d_interactive.html" target="_blank">3D Vowel Space (F1-F2-F3)</a>
            </div>
            <div class="card">
                <h2>Model Analysis</h2>
                <a href="feature_correlation_interactive.html" target="_blank">Feature Correlation Heatmap</a>
                <a href="confusion_matrix_interactive.html" target="_blank">Confusion Matrix</a>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(interactive_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    
    print(f"Interactive dashboard created at: {os.path.join(interactive_dir, 'index.html')}")

# Update the main function to include the interactive dashboard
if __name__ == "__main__":
    results = main()
    create_interactive_dashboard(results, './vowel_classification_output')