
"""
Excited/Happy: "I'm thrilled to announce that I've been accepted into the AI research program at IIT Jodhpur with a full scholarship!"
Calm/Informative: "Artificial neural networks consist of interconnected layers of nodes that process and transform input data."
Sad/Disappointed: "Unfortunately, my research paper was rejected due to insufficient experimental data despite months of hard work."
Angry/Frustrated: "The computing cluster crashed again right before my deadline, losing three days of critical processing time!"
Curious/Questioning: "I wonder how quantum computing might revolutionize the field of machine learning in the next decade?"
Urgent/Commanding: "Submit your assignment by midnight tonight or you'll lose 30% of the total grade - no exceptions!"
Nostalgic/Reflective: "Growing up in India, I never imagined I would one day be studying cutting-edge artificial intelligence algorithms."
Surprised/Astonished: "Wait, you're telling me our model achieved 98% accuracy on the test dataset without any fine-tuning?!"
Worried/Concerned: "I'm concerned about the ethical implications of deploying autonomous AI systems without proper regulation."
Dreamy/Aspirational: "Someday, I hope to develop AI solutions that can help solve the most pressing challenges facing rural communities in India.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal
import pandas as pd
import os
from IPython.display import Audio
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Create directories for plots
os.makedirs('plots/waveforms', exist_ok=True)
os.makedirs('plots/spectrograms', exist_ok=True)
os.makedirs('plots/mfcc', exist_ok=True)
os.makedirs('plots/pitch', exist_ok=True)
os.makedirs('plots/energy', exist_ok=True)
os.makedirs('plots/formants', exist_ok=True)
os.makedirs('plots/comparative', exist_ok=True)
os.makedirs('plots/3d_visualizations', exist_ok=True)

# Function to analyze audio file
def analyze_audio(audio_path, sample_id):
    print(f"\nAnalyzing sample {sample_id}: {os.path.basename(audio_path)}")
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Number of samples: {len(y)}")
    
    # Create time array for plotting
    time = np.arange(0, len(y)) / sr
    
    # Compute amplitude values
    abs_amplitude = np.abs(y)
    max_amplitude = np.max(abs_amplitude)
    min_amplitude = np.min(abs_amplitude)
    mean_amplitude = np.mean(abs_amplitude)
    std_amplitude = np.std(abs_amplitude)
    
    print(f"Maximum amplitude: {max_amplitude:.4f}")
    print(f"Minimum amplitude: {min_amplitude:.4f}")
    print(f"Mean amplitude: {mean_amplitude:.4f}")
    print(f"Standard deviation of amplitude: {std_amplitude:.4f}")
    
    # Compute RMS energy
    frame_length = 1024
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    rms_max = np.max(rms)
    
    print(f"Mean RMS energy: {rms_mean:.4f}")
    print(f"Std RMS energy: {rms_std:.4f}")
    print(f"Max RMS energy: {rms_max:.4f}")
    
    # Extract pitch (fundamental frequency)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                 fmin=librosa.note_to_hz('C2'),
                                                 fmax=librosa.note_to_hz('C7'),
                                                 sr=sr)
    # Remove NaN values for statistics
    f0_cleaned = f0[~np.isnan(f0)]
    
    if len(f0_cleaned) > 0:
        mean_f0 = np.mean(f0_cleaned)
        std_f0 = np.std(f0_cleaned)
        min_f0 = np.min(f0_cleaned)
        max_f0 = np.max(f0_cleaned)
        
        print(f"Mean fundamental frequency (pitch): {mean_f0:.2f} Hz")
        print(f"Std fundamental frequency: {std_f0:.2f} Hz")
        print(f"Min fundamental frequency: {min_f0:.2f} Hz")
        print(f"Max fundamental frequency: {max_f0:.2f} Hz")
    else:
        print("Could not detect fundamental frequency")
        mean_f0 = std_f0 = min_f0 = max_f0 = 0
    
    # Calculate spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_spectral_centroid = np.mean(spectral_centroids)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    mean_spectral_bandwidth = np.mean(spectral_bandwidth)
    
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    mean_spectral_flatness = np.mean(spectral_flatness)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mean_spectral_rolloff = np.mean(spectral_rolloff)
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    mean_zero_crossing_rate = np.mean(zero_crossing_rate)
    
    print(f"Mean spectral centroid: {mean_spectral_centroid:.2f} Hz")
    print(f"Mean spectral bandwidth: {mean_spectral_bandwidth:.2f} Hz")
    print(f"Mean spectral flatness: {mean_spectral_flatness:.6f}")
    print(f"Mean spectral rolloff: {mean_spectral_rolloff:.2f} Hz")
    print(f"Mean zero crossing rate: {mean_zero_crossing_rate:.6f}")
    
    # Extract MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Compute spectrogram
    D = librosa.stft(y, n_fft=2048, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Compute chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Estimate formants (using LPC method)
    try:
        n_formants = 4  # Number of formants to estimate
        n_lpc = 2 + 2 * n_formants  # Rule of thumb for LPC order
        lpc_coeffs = librosa.lpc(y, order=n_lpc)
        
        # Get the roots of the LPC polynomial
        roots = np.roots(lpc_coeffs)
        
        # Keep only roots with positive imaginary part and compute angle
        roots = roots[np.imag(roots) > 0]
        angles = np.angle(roots)
        
        # Convert to frequency
        formants = angles * (sr / (2 * np.pi))
        formants = np.sort(formants)[:n_formants]
        
        formant_str = ", ".join([f"F{i+1}: {f:.1f} Hz" for i, f in enumerate(formants)])
        print(f"Estimated formants: {formant_str}")
    except:
        formants = np.zeros(n_formants)
        print("Could not estimate formants")
    
    # 1. Plot Waveform with envelope
    plt.figure(figsize=(12, 4))
    plt.plot(time, y, color='blue', alpha=0.6)
    
    # Add amplitude envelope
    frame_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    envelope = librosa.util.normalize(rms) * max_amplitude
    plt.plot(frame_time, envelope, color='red', linewidth=2, label='Amplitude Envelope')
    plt.plot(frame_time, -envelope, color='red', linewidth=2)
    
    plt.title(f'Waveform and Amplitude Envelope - Sample {sample_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/waveforms/waveform_{sample_id}.png', dpi=300)
    plt.close()
    
    # 2. Plot enhanced Spectrogram
    plt.figure(figsize=(12, 6))
    
    # Create a custom colormap with better contrast
    colors = [(1, 1, 1), (0, 0, 0.8), (0, 0, 0.5), (0.8, 0, 0)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors)
    
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap=custom_cmap)
    
    # Add pitch contour on spectrogram
    times = librosa.times_like(f0, sr=sr)
    plt.plot(times, f0, color='yellow', linewidth=2, alpha=0.7, label='Pitch Contour')
    
    # Add formant tracks if available
    if np.any(formants > 0):
        for i, formant in enumerate(formants):
            plt.axhline(y=formant, color='lime', linestyle='--', linewidth=1, alpha=0.7, label=f'F{i+1}' if i == 0 else None)
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram with Pitch and Formants - Sample {sample_id}')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'plots/spectrograms/spectrogram_{sample_id}.png', dpi=300)
    plt.close()
    
    # 3. Plot Mel-frequency spectrogram and MFCCs
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - Sample {sample_id}')
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='coolwarm')
    plt.colorbar(format='%+2.0f')
    plt.title(f'MFCCs - Sample {sample_id}')
    plt.tight_layout()
    plt.savefig(f'plots/mfcc/mfcc_{sample_id}.png', dpi=300)
    plt.close()
    
    # 4. Plot Pitch with violin plot for distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(times, f0, label='F0', color='blue', alpha=0.8)
    plt.axhline(y=mean_f0, color='r', linestyle='-', label=f'Mean: {mean_f0:.1f} Hz')
    plt.fill_between(times, 
                     mean_f0 - std_f0, 
                     mean_f0 + std_f0, 
                     color='red', 
                     alpha=0.2, 
                     label=f'±σ: {std_f0:.1f} Hz')
    plt.title(f'Pitch Contour - Sample {sample_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if len(f0_cleaned) > 0:
        # Only plot if we have enough non-NaN values
        sns.violinplot(y=f0_cleaned, inner='quartile', color='lightblue')
        plt.axhline(y=mean_f0, color='r', linestyle='-', label=f'Mean: {mean_f0:.1f} Hz')
        plt.title(f'Pitch Distribution - Sample {sample_id}')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/pitch/pitch_{sample_id}.png', dpi=300)
    plt.close()
    
    # 5. Plot RMS Energy with histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(frame_time, rms, color='green')
    plt.axhline(y=rms_mean, color='r', linestyle='-', label=f'Mean: {rms_mean:.4f}')
    plt.fill_between(frame_time, 
                     np.maximum(0, rms_mean - rms_std), 
                     rms_mean + rms_std, 
                     color='red', 
                     alpha=0.2, 
                     label=f'±σ: {rms_std:.4f}')
    plt.title(f'RMS Energy - Sample {sample_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Energy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(rms, kde=True, color='green')
    plt.axvline(x=rms_mean, color='r', linestyle='-', label=f'Mean: {rms_mean:.4f}')
    plt.title(f'RMS Energy Distribution - Sample {sample_id}')
    plt.xlabel('RMS Energy')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/energy/energy_{sample_id}.png', dpi=300)
    plt.close()
    
    # 6. Plot Formants if available
    if np.any(formants > 0):
        plt.figure(figsize=(8, 6))
        formant_labels = [f'F{i+1}' for i in range(len(formants))]
        bars = plt.bar(formant_labels, formants, color='purple')
        
        # Add the values on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{height:.1f} Hz', ha='center', va='bottom')
        
        plt.title(f'Estimated Formant Frequencies - Sample {sample_id}')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'plots/formants/formants_{sample_id}.png', dpi=300)
        plt.close()
    
    # Return the results as a dictionary
    results = {
        'Sample ID': sample_id,
        'Filename': os.path.basename(audio_path),
        'Duration (s)': duration,
        'Max Amplitude': max_amplitude,
        'Mean Amplitude': mean_amplitude,
        'Std Amplitude': std_amplitude,
        'Mean RMS Energy': rms_mean,
        'Std RMS Energy': rms_std,
        'Max RMS Energy': rms_max,
        'Mean Pitch (Hz)': mean_f0,
        'Std Pitch (Hz)': std_f0,
        'Min Pitch (Hz)': min_f0,
        'Max Pitch (Hz)': max_f0,
        'Pitch Range (Hz)': max_f0 - min_f0 if len(f0_cleaned) > 0 else 0,
        'Spectral Centroid (Hz)': mean_spectral_centroid,
        'Spectral Bandwidth (Hz)': mean_spectral_bandwidth,
        'Spectral Flatness': mean_spectral_flatness,
        'Spectral Rolloff (Hz)': mean_spectral_rolloff,
        'Zero Crossing Rate': mean_zero_crossing_rate
    }
    
    # Add formants to results if available
    if np.any(formants > 0):
        for i, formant in enumerate(formants):
            results[f'Formant F{i+1} (Hz)'] = formant
    
    # Add MFCC means to results
    for i, mfcc_mean in enumerate(mfccs_mean):
        results[f'MFCC {i+1} Mean'] = mfcc_mean
    
    # Add notes about the emotional content if provided
    # (placeholder for manual annotation or metadata from filename)
    
    return results

# Main analysis function
def analyze_audio_samples(audio_folder):
    results = []
    
    # List all audio files in the folder
    audio_files = sorted([os.path.join(audio_folder, f) for f in os.listdir(audio_folder) 
                  if f.endswith(('.wav', '.mp3', '.m4a'))])
    
    if not audio_files:
        print("No audio files found in the specified folder.")
        return
    
    # Analyze each audio file
    for i, audio_path in enumerate(audio_files, 1):
        result = analyze_audio(audio_path, i)
        results.append(result)
    
    # Create a dataframe with all results
    df = pd.DataFrame(results)
    
    # Save results to CSV
    df.to_csv('audio_analysis_results.csv', index=False)
    print("\nAnalysis completed! Results saved to 'audio_analysis_results.csv'")
    
    # Display comparative visualizations
    create_comparative_visualizations(df)
    
def create_comparative_visualizations(df):
    # Set a common style for plots
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    # Define a color palette for the samples
    palette = sns.color_palette("viridis", len(df))
    
    # 1. Compare mean and range of pitch across samples
    plt.figure(figsize=(14, 8))
    
    # Plot each sample with its own color
    for i, row in df.iterrows():
        plt.errorbar(row['Sample ID'], row['Mean Pitch (Hz)'],
                    yerr=[[row['Mean Pitch (Hz)'] - row['Min Pitch (Hz)']], 
                          [row['Max Pitch (Hz)'] - row['Mean Pitch (Hz)']]], 
                    fmt='o', capsize=5, ecolor='black', markersize=8,
                    color=palette[i], alpha=0.7)
    
    for i, row in df.iterrows():
        plt.text(row['Sample ID'], row['Mean Pitch (Hz)'] + 5, 
                f"{row['Mean Pitch (Hz)']:.1f} Hz", 
                ha='center', fontsize=9)
    
    plt.title('Mean Pitch and Range Comparison Across Samples', fontsize=16)
    plt.xlabel('Sample ID', fontsize=14)
    plt.ylabel('Pitch (Hz)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(df['Sample ID'])
    
    # Add a subtle background color to alternating samples for easier reading
    for i in range(0, len(df), 2):
        if i < len(df):
            plt.axvspan(df['Sample ID'].iloc[i] - 0.5, 
                       df['Sample ID'].iloc[i] + 0.5, 
                       color='gray', alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('plots/comparative/comparative_pitch_range.png', dpi=300)
    plt.close()
    
    # 2. Create a multi-feature comparison heatmap
    plt.figure(figsize=(16, 10))
    
    # Select relevant features for heatmap
    features_for_heatmap = [
        'Mean Pitch (Hz)', 'Pitch Range (Hz)', 'Mean RMS Energy', 
        'Max Amplitude', 'Spectral Centroid (Hz)', 'Spectral Flatness',
        'Zero Crossing Rate'
    ]
    
    # Add formants if they exist
    formant_cols = [col for col in df.columns if 'Formant' in col]
    if formant_cols:
        features_for_heatmap.extend(formant_cols)
    
    # Create a copy of the data for normalization
    heatmap_data = df[features_for_heatmap].copy()
    
    # Normalize each feature column to 0-1 range for better visualization
    for col in heatmap_data.columns:
        if heatmap_data[col].std() > 0:  # Avoid division by zero
            heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / \
                               (heatmap_data[col].max() - heatmap_data[col].min())
    
    # Add sample IDs as index
    heatmap_data.index = [f"Sample {i}" for i in df['Sample ID']]
    
    # Create the heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", 
                    linewidths=0.5, cbar_kws={"label": "Normalized Value"})
    
    plt.title('Normalized Audio Features Across Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/comparative/feature_heatmap.png', dpi=300)
    plt.close()
    
    # 3. Create a correlation matrix of audio features
    plt.figure(figsize=(14, 12))
    
    # Select all numeric features for correlation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Remove ID and non-relevant columns
    exclude_cols = ['Sample ID', 'Duration (s)']
    corr_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Compute correlation matrix
    corr_matrix = df[corr_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create the heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, 
               annot=True, fmt=".2f", linewidths=0.5, square=True)
    
    plt.title('Correlation Matrix of Audio Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/comparative/feature_correlation.png', dpi=300)
    plt.close()
    
    # 4. Create feature-pair relationship plots for key features
    key_features = ['Mean Pitch (Hz)', 'Mean RMS Energy', 'Spectral Centroid (Hz)', 'Zero Crossing Rate']
    
    # Create pairwise plots
    g = sns.PairGrid(df, vars=key_features, hue='Sample ID', palette=palette, height=2.5)
    g.map_diag(sns.histplot, kde=True)
    g.map_offdiag(sns.scatterplot)
    g.add_legend(title="Sample ID", bbox_to_anchor=(1, 0.5), loc='center left')
    
    plt.tight_layout()
    plt.savefig('plots/comparative/feature_pairplot.png', dpi=300)
    plt.close()
    
    # 5. Create a dynamic range visualization (amplitude vs energy)
    plt.figure(figsize=(12, 8))
    
    plt.scatter(df['Mean RMS Energy'], df['Max Amplitude'], 
               s=100, c=df['Mean Pitch (Hz)'], cmap='plasma', alpha=0.7)
    
    # Add sample IDs as labels
    for i, row in df.iterrows():
        plt.annotate(f"Sample {row['Sample ID']}",
                    (row['Mean RMS Energy'], row['Max Amplitude']),
                    xytext=(5, 5), textcoords='offset points')
    
    cbar = plt.colorbar()
    cbar.set_label('Mean Pitch (Hz)')
    
    plt.title('Dynamic Range Analysis: RMS Energy vs Maximum Amplitude', fontsize=16)
    plt.xlabel('Mean RMS Energy', fontsize=14)
    plt.ylabel('Maximum Amplitude', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/comparative/dynamic_range.png', dpi=300)
    plt.close()
    
    # 6. Create radar charts for feature comparison
    create_enhanced_radar_charts(df)
    
    # 7. Create PCA visualization of the samples
    create_pca_visualization(df)
    
    # 8. Create 3D feature space visualization
    create_3d_visualization(df)
    
    # 9. Create tempo and rhythm analysis (if applicable)
    create_tempo_analysis(df)

def create_enhanced_radar_charts(df):
    # Select features for radar charts
    radar_features = [
        'Mean Amplitude', 'Mean RMS Energy', 'Mean Pitch (Hz)', 
        'Spectral Centroid (Hz)', 'Spectral Flatness', 'Zero Crossing Rate'
    ]
    
    # Normalize the data for radar chart
    radar_data = df[radar_features].copy()
    for feature in radar_features:
        if radar_data[feature].std() > 0:  # Avoid division by zero
            radar_data[feature] = (radar_data[feature] - radar_data[feature].min()) / \
                                 (radar_data[feature].max() - radar_data[feature].min())
    
    # Number of features
    n_features = len(radar_features)
    
    # Angles for radar chart
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Function to create radar chart for a subset of samples
    def plot_radar_subset(sample_indices, title):
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add lines for each sample
        for i, idx in enumerate(sample_indices):
            sample_id = df.iloc[idx]['Sample ID']
            values = radar_data.iloc[idx][radar_features].values.tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                    label=f'Sample {sample_id}', alpha=0.8)
            ax.fill(angles, values, alpha=0.1)
        
        # Add feature labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace(' (Hz)', '').replace('Mean ', '') for f in radar_features])
        
        # Add labels at each tick
        for angle, feature in zip(angles[:-1], radar_features):
            ha = 'center'
            if angle < np.pi/2 or angle > 3*np.pi/2:
                ha = 'left'
            elif np.pi/2 < angle < 3*np.pi/2:
                ha = 'right'
            
            ax.text(angle, 1.2, feature.replace(' (Hz)', '').replace('Mean ', ''), 
                   ha=ha, va='center', fontsize=12, weight='bold')
        
        # Add gridlines and set limits
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add title and legend
        plt.title(title, size=16, pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f'plots/comparative/radar_chart_{title.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()
    
    # Create radar charts for different sample groups
    if len(df) >= 3:
        plot_radar_subset(range(3), 'First 3 Samples')
    
    if len(df) >= 5:
        plot_radar_subset(range(5), 'First 5 Samples')
    
    if len(df) >= 10:
        # All 10 samples together would be cluttered, so create meaningful subsets
        # For example, compare highest and lowest pitch samples
        high_pitch_idx = df['Mean Pitch (Hz)'].nlargest(3).index.tolist()
        low_pitch_idx = df['Mean Pitch (Hz)'].nsmallest(3).index.tolist()
        plot_radar_subset(high_pitch_idx, 'Highest Pitch Samples')
        plot_radar_subset(low_pitch_idx, 'Lowest Pitch Samples')
        
        # Compare highest and lowest energy samples
        high_energy_idx = df['Mean RMS Energy'].nlargest(3).index.tolist()
        low_energy_idx = df['Mean RMS Energy'].nsmallest(3).index.tolist()
        plot_radar_subset(high_energy_idx, 'Highest Energy Samples')
        plot_radar_subset(low_energy_idx, 'Lowest Energy Samples')

def create_pca_visualization(df):
    # Select features for PCA
    pca_features = [col for col in df.columns if col not in 
                   ['Sample ID', 'Filename', 'Duration (s)']]
    
    # Ensure all features are numeric
    pca_features = [f for f in pca_features if df[f].dtype in ['float64', 'int64']]
    
    # Standardize the data
    X = df[pca_features].dropna(axis=1)  # Drop columns with NaN values
    X_std = StandardScaler().fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_std)
    
    # Create DataFrame with principal components
    pca_df = pd.DataFrame(data=principal_components, 
                          columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['Sample ID'] = df['Sample ID']
    
    # Create PCA plot
    plt.figure(figsize=(12, 10))
    
    # Use a color map that works with numeric values
    scatter = plt.scatter(pca_df['Principal Component 1'], 
                         pca_df['Principal Component 2'],
                         c=pca_df['Sample ID'],  # Use Sample ID for coloring
                         cmap='viridis',        # Use a colormap
                         s=200, 
                         alpha=0.7)
    
    # Add sample labels
    for i, row in pca_df.iterrows():
        plt.annotate(f"Sample {int(row['Sample ID'])}",
                    (row['Principal Component 1'], row['Principal Component 2']),
                    xytext=(5, 5), textcoords='offset points')
    
    # Add explained variance information
    explained_variance = pca.explained_variance_ratio_
    plt.title(f'PCA of Audio Features\nPC1: {explained_variance[0]*100:.1f}% variance, PC2: {explained_variance[1]*100:.1f}% variance', 
             fontsize=16)
    
    # Add colorbar
    plt.colorbar(scatter, label='Sample ID')
    
    # Add loading vectors
    feature_names = X.columns
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='red', alpha=0.5, 
                 head_width=0.05, head_length=0.05)
        plt.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, 
                feature.replace(' (Hz)', '').replace('Mean ', ''),
                color='red', ha='center', va='center')
    
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/comparative/pca_visualization.png', dpi=300)
    plt.close()

def create_3d_visualization(df):
    # Select 3 important features for 3D visualization
    features_3d = ['Mean Pitch (Hz)', 'Spectral Centroid (Hz)', 'Mean RMS Energy']
    
    # Check if we have all the features
    if not all(feature in df.columns for feature in features_3d):
        print("Missing features for 3D visualization, skipping...")
        return
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each sample
    scatter = ax.scatter(df['Mean Pitch (Hz)'], 
                         df['Spectral Centroid (Hz)'], 
                         df['Mean RMS Energy'],
                         c=df['Sample ID'], 
                         cmap='viridis', 
                         s=100, 
                         alpha=0.7)
    
    # Add sample labels
    for i, row in df.iterrows():
        ax.text(row['Mean Pitch (Hz)'], 
                row['Spectral Centroid (Hz)'], 
                row['Mean RMS Energy'], 
                f"Sample {int(row['Sample ID'])}")
    
    # Set labels and title
    ax.set_xlabel('Mean Pitch (Hz)', fontsize=12)
    ax.set_ylabel('Spectral Centroid (Hz)', fontsize=12)
    ax.set_zlabel('Mean RMS Energy', fontsize=12)
    plt.title('3D Feature Space Visualization', fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Sample ID')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('plots/3d_visualizations/3d_feature_space.png', dpi=300)
    plt.close()
    
    # Create additional 3D visualizations with different feature combinations if enough samples
    if len(df) >= 5:
        alt_features = ['Zero Crossing Rate', 'Spectral Flatness', 'Spectral Bandwidth (Hz)']
        
        # Check if we have all the features
        if all(feature in df.columns for feature in alt_features):
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(df[alt_features[0]], 
                                df[alt_features[1]], 
                                df[alt_features[2]],
                                c=df['Mean Pitch (Hz)'], 
                                cmap='plasma', 
                                s=100, 
                                alpha=0.7)
            
            # Add sample labels
            for i, row in df.iterrows():
                ax.text(row[alt_features[0]], 
                        row[alt_features[1]], 
                        row[alt_features[2]], 
                        f"Sample {int(row['Sample ID'])}")
            
            # Set labels and title
            ax.set_xlabel(alt_features[0], fontsize=12)
            ax.set_ylabel(alt_features[1], fontsize=12)
            ax.set_zlabel(alt_features[2], fontsize=12)
            plt.title('Alternative 3D Feature Space', fontsize=16)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Mean Pitch (Hz)')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig('plots/3d_visualizations/alt_3d_feature_space.png', dpi=300)
            plt.close()

def create_tempo_analysis(df):
    # Note: This is a simplified tempo analysis that would work better with the actual audio files
    # For a more detailed analysis, we would need to extract tempo and rhythm features during the analyze_audio function
    
    # Check if we have Zero Crossing Rate and RMS Energy which can be related to rhythm
    if 'Zero Crossing Rate' in df.columns and 'Mean RMS Energy' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with Zero Crossing Rate vs RMS Energy
        # (higher ZCR and energy often correlate with faster tempo/more rhythmic content)
        plt.scatter(df['Zero Crossing Rate'], df['Mean RMS Energy'], 
                   s=100, c=df['Sample ID'], cmap='viridis', alpha=0.7)
        
        # Add sample labels
        for i, row in df.iterrows():
            plt.annotate(f"Sample {int(row['Sample ID'])}",
                        (row['Zero Crossing Rate'], row['Mean RMS Energy']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Rhythm Analysis Proxy: Zero Crossing Rate vs RMS Energy', fontsize=16)
        plt.xlabel('Zero Crossing Rate (higher = more transients)', fontsize=14)
        plt.ylabel('Mean RMS Energy (volume/intensity)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Sample ID')
        plt.tight_layout()
        plt.savefig('plots/comparative/rhythm_analysis.png', dpi=300)
        plt.close()

# Run the analysis function
if __name__ == "__main__":
    # Set the folder containing audio files
    audio_folder = "/Users/aditibaheti/Desktop/IITJ/Speech/M23CSA001_SpeechMinor/Ques_1/audio_samples"
    analyze_audio_samples(audio_folder)
    
    print("\nAnalysis complete!")
    print("Visualizations have been saved in the 'plots' directory")
    print("Results are available in 'audio_analysis_results.csv'")