
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
import warnings
warnings.filterwarnings('ignore')

# Create directories for plots
os.makedirs('plots/waveforms', exist_ok=True)
os.makedirs('plots/spectrograms', exist_ok=True)
os.makedirs('plots/pitch', exist_ok=True)
os.makedirs('plots/energy', exist_ok=True)

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
    
    print(f"Maximum amplitude: {max_amplitude:.4f}")
    print(f"Minimum amplitude: {min_amplitude:.4f}")
    print(f"Mean amplitude: {mean_amplitude:.4f}")
    
    # Compute RMS energy
    frame_length = 1024
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = np.mean(rms)
    rms_max = np.max(rms)
    
    print(f"Mean RMS energy: {rms_mean:.4f}")
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
        min_f0 = np.min(f0_cleaned)
        max_f0 = np.max(f0_cleaned)
        
        print(f"Mean fundamental frequency (pitch): {mean_f0:.2f} Hz")
        print(f"Min fundamental frequency: {min_f0:.2f} Hz")
        print(f"Max fundamental frequency: {max_f0:.2f} Hz")
    else:
        print("Could not detect fundamental frequency")
        mean_f0 = min_f0 = max_f0 = 0
    
    # Calculate spectral centroid (brightness of sound)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_spectral_centroid = np.mean(spectral_centroids)
    print(f"Mean spectral centroid: {mean_spectral_centroid:.2f} Hz")
    
    # Compute spectrogram
    D = librosa.stft(y, n_fft=2048, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # 1. Plot Waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, y)
    plt.title(f'Waveform - Sample {sample_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/waveforms/waveform_{sample_id}.png')
    plt.close()
    
    # 2. Plot Spectrogram
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - Sample {sample_id}')
    plt.tight_layout()
    plt.savefig(f'plots/spectrograms/spectrogram_{sample_id}.png')
    plt.close()
    
    # 3. Plot Pitch
    times = librosa.times_like(f0, sr=sr)
    plt.figure(figsize=(12, 4))
    plt.plot(times, f0, label='f0', color='blue', alpha=0.8)
    plt.axhline(y=mean_f0, color='r', linestyle='-', label=f'Mean: {mean_f0:.1f} Hz')
    plt.title(f'Pitch Contour - Sample {sample_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'plots/pitch/pitch_{sample_id}.png')
    plt.close()
    
    # 4. Plot RMS Energy
    frames = np.arange(len(rms))
    frame_time = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(12, 4))
    plt.plot(frame_time, rms)
    plt.axhline(y=rms_mean, color='r', linestyle='-', label=f'Mean: {rms_mean:.4f}')
    plt.title(f'RMS Energy - Sample {sample_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Energy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'plots/energy/energy_{sample_id}.png')
    plt.close()
    
    # Return the results as a dictionary
    return {
        'Sample ID': sample_id,
        'Filename': os.path.basename(audio_path),
        'Duration (s)': duration,
        'Max Amplitude': max_amplitude,
        'Mean Amplitude': mean_amplitude,
        'Mean RMS Energy': rms_mean,
        'Max RMS Energy': rms_max,
        'Mean Pitch (Hz)': mean_f0,
        'Min Pitch (Hz)': min_f0,
        'Max Pitch (Hz)': max_f0,
        'Spectral Centroid (Hz)': mean_spectral_centroid
    }

# Main analysis function
def analyze_audio_samples(audio_folder):
    results = []
    
    # List all audio files in the folder
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) 
                  if f.endswith(('.wav', '.mp3', '.m4a'))]
    
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
    # 1. Compare mean pitch across samples
    plt.figure(figsize=(12, 6))
    plt.bar(df['Sample ID'], df['Mean Pitch (Hz)'], color='skyblue')
    plt.title('Mean Pitch Comparison Across Samples')
    plt.xlabel('Sample ID')
    plt.ylabel('Mean Pitch (Hz)')
    plt.grid(True, alpha=0.3)
    plt.xticks(df['Sample ID'])
    plt.tight_layout()
    plt.savefig('plots/comparative_pitch.png')
    plt.close()
    
    # 2. Compare RMS energy across samples
    plt.figure(figsize=(12, 6))
    plt.bar(df['Sample ID'], df['Mean RMS Energy'], color='lightgreen')
    plt.title('Mean RMS Energy Comparison Across Samples')
    plt.xlabel('Sample ID')
    plt.ylabel('Mean RMS Energy')
    plt.grid(True, alpha=0.3)
    plt.xticks(df['Sample ID'])
    plt.tight_layout()
    plt.savefig('plots/comparative_energy.png')
    plt.close()
    
    # 3. Compare max amplitude across samples
    plt.figure(figsize=(12, 6))
    plt.bar(df['Sample ID'], df['Max Amplitude'], color='salmon')
    plt.title('Maximum Amplitude Comparison Across Samples')
    plt.xlabel('Sample ID')
    plt.ylabel('Maximum Amplitude')
    plt.grid(True, alpha=0.3)
    plt.xticks(df['Sample ID'])
    plt.tight_layout()
    plt.savefig('plots/comparative_amplitude.png')
    plt.close()
    
    # 4. Create scatter plot of pitch vs. energy
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Mean Pitch (Hz)'], df['Mean RMS Energy'], c=df['Sample ID'], 
                cmap='viridis', s=100, alpha=0.7)
    
    # Add sample IDs as labels
    for i, txt in enumerate(df['Sample ID']):
        plt.annotate(txt, (df['Mean Pitch (Hz)'].iloc[i], df['Mean RMS Energy'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.colorbar(label='Sample ID')
    plt.title('Relationship Between Pitch and Energy')
    plt.xlabel('Mean Pitch (Hz)')
    plt.ylabel('Mean RMS Energy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/pitch_vs_energy.png')
    plt.close()
    
    # 5. Create a multi-feature radar chart for each sample
    # Normalize the data for radar chart
    features = ['Mean Amplitude', 'Mean RMS Energy', 'Mean Pitch (Hz)', 
                'Max Amplitude', 'Spectral Centroid (Hz)']
    
    # Check if there are enough samples to create radar charts
    if len(df) >= 3:
        # Create radar charts for first 3 samples
        plot_radar_chart(df.iloc[:3], features, 'First 3 Samples')
    
    if len(df) >= 5:
        # Create radar charts for first 5 samples
        plot_radar_chart(df.iloc[:5], features, 'First 5 Samples')
    
    if len(df) >= 10:
        # Create radar charts comparing samples 1, 5, and 10
        indices = [0, 4, 9]  # 0-indexed, so 1st, 5th, and 10th samples
        plot_radar_chart(df.iloc[indices], features, 'Samples 1, 5, and 10')

def plot_radar_chart(df_subset, features, title):
    # Number of samples and features
    n_samples = len(df_subset)
    n_features = len(features)
    
    # Normalize data for radar chart
    df_norm = df_subset[features].copy()
    for feature in features:
        df_norm[feature] = (df_subset[feature] - df_subset[feature].min()) / \
                           (df_subset[feature].max() - df_subset[feature].min())
    
    # Angles for radar chart
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add lines for each sample
    colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
    for i, idx in enumerate(df_subset.index):
        sample_id = df_subset.loc[idx, 'Sample ID']
        values = df_norm.loc[idx, features].values.tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], 
                label=f'Sample {sample_id}', alpha=0.8)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Add feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace(' (Hz)', '').replace('Mean ', '') for f in features])
    
    # Add gridlines
    ax.set_ylim(0, 1)
    ax.grid(True)
    
    # Add title and legend
    plt.title(f'Audio Feature Comparison - {title}', size=15, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(f'plots/radar_chart_{title.replace(" ", "_").lower()}.png')
    plt.close()

if __name__ == "__main__":
    # Specify the folder containing your audio samples
    audio_folder = "/Users/aditibaheti/Desktop/IITJ/Speech/M23CSA001_SpeechMinor/Ques_1/audio_samples"  # Change this to your folder path
    
    # Run the analysis
    print("Starting audio analysis...")
    analyze_audio_samples(audio_folder)