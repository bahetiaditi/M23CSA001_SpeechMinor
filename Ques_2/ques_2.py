import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd
import os
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

class SpeechAnalyzer:
    def __init__(self, file_path=None, sample_rate=None):
        """
        Initialize the speech analyzer with an audio file
        
        Parameters:
        -----------
        file_path : str
            Path to the audio file
        sample_rate : int, optional
            Target sample rate for resampling. If None, original sample rate is used.
        """
        self.file_path = file_path
        self.target_sr = sample_rate
        self.features = {}
        
        if file_path:
            self.load_audio(file_path, sample_rate)
    
    def load_audio(self, file_path, sample_rate=None):
        """
        Load an audio file using librosa
        
        Parameters:
        -----------
        file_path : str
            Path to the audio file
        sample_rate : int, optional
            Target sample rate for resampling
        """
        try:
            self.audio, self.sr = librosa.load(file_path, sr=sample_rate, mono=True)
            self.file_path = file_path
            self.target_sr = self.sr
            self.duration = librosa.get_duration(y=self.audio, sr=self.sr)
            print(f"Loaded audio file: {file_path}")
            print(f"Duration: {self.duration:.2f} seconds")
            print(f"Sample rate: {self.sr} Hz")
            print(f"Number of samples: {len(self.audio)}")
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def preprocess_audio(self, normalize=True, trim_silence=True, noise_reduction=False):
        """
        Preprocess the audio signal
        
        Parameters:
        -----------
        normalize : bool
            Whether to normalize the audio signal
        trim_silence : bool
            Whether to trim leading and trailing silence
        noise_reduction : bool
            Whether to apply noise reduction (simple high-pass filter)
        """
        if not hasattr(self, 'audio'):
            print("No audio loaded. Please load audio first.")
            return
        
        # Store original audio for comparison
        self.original_audio = self.audio.copy()
        
        # Normalize the audio
        if normalize:
            self.audio = librosa.util.normalize(self.audio)
        
        # Trim silence
        if trim_silence:
            self.audio, _ = librosa.effects.trim(self.audio, top_db=30)
        
        # Simple noise reduction using high-pass filter
        if noise_reduction:
            b, a = signal.butter(5, 100/(self.sr/2), 'highpass')
            self.audio = signal.filtfilt(b, a, self.audio)
        
        print("Audio preprocessing completed")
    
    def extract_zcr(self, frame_length=2048, hop_length=512):
        """
        Extract Zero Crossing Rate (ZCR)
        
        Parameters:
        -----------
        frame_length : int
            The length of each frame in samples
        hop_length : int
            The number of samples between successive frames
            
        Returns:
        --------
        zcr : numpy.ndarray
            Zero Crossing Rate for each frame
        """
        if not hasattr(self, 'audio'):
            print("No audio loaded. Please load audio first.")
            return None
        
        zcr = librosa.feature.zero_crossing_rate(
            y=self.audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )
        
        self.features['zcr'] = zcr[0]
        self.features['zcr_mean'] = np.mean(zcr)
        self.features['zcr_std'] = np.std(zcr)
        self.features['zcr_frame_length'] = frame_length
        self.features['zcr_hop_length'] = hop_length
        
        return zcr[0]
    
    def extract_ste(self, frame_length=2048, hop_length=512):
        """
        Extract Short-Time Energy (STE)
        
        Parameters:
        -----------
        frame_length : int
            The length of each frame in samples
        hop_length : int
            The number of samples between successive frames
            
        Returns:
        --------
        ste : numpy.ndarray
            Short-Time Energy for each frame
        """
        if not hasattr(self, 'audio'):
            print("No audio loaded. Please load audio first.")
            return None
        
        # Calculate energy using librosa's rms function (Root Mean Square energy)
        rms = librosa.feature.rms(
            y=self.audio,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # Convert RMS to energy by squaring
        ste = rms[0] ** 2
        
        self.features['ste'] = ste
        self.features['ste_mean'] = np.mean(ste)
        self.features['ste_std'] = np.std(ste)
        self.features['ste_frame_length'] = frame_length
        self.features['ste_hop_length'] = hop_length
        
        return ste
    
    def extract_mfcc(self, n_mfcc=13, frame_length=2048, hop_length=512):
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCCs)
        
        Parameters:
        -----------
        n_mfcc : int
            Number of MFCCs to extract
        frame_length : int
            The length of each frame in samples
        hop_length : int
            The number of samples between successive frames
            
        Returns:
        --------
        mfcc : numpy.ndarray
            MFCCs for each frame
        """
        if not hasattr(self, 'audio'):
            print("No audio loaded. Please load audio first.")
            return None
        
        mfcc = librosa.feature.mfcc(
            y=self.audio, 
            sr=self.sr, 
            n_mfcc=n_mfcc,
            n_fft=frame_length,
            hop_length=hop_length
        )
        
        self.features['mfcc'] = mfcc
        self.features['mfcc_mean'] = np.mean(mfcc, axis=1)
        self.features['mfcc_std'] = np.std(mfcc, axis=1)
        self.features['mfcc_n_mfcc'] = n_mfcc
        self.features['mfcc_frame_length'] = frame_length
        self.features['mfcc_hop_length'] = hop_length
        
        return mfcc
    
    def extract_all_features(self, frame_length=2048, hop_length=512, n_mfcc=13):
        """
        Extract all features: ZCR, STE, and MFCC
        
        Parameters:
        -----------
        frame_length : int
            The length of each frame in samples
        hop_length : int
            The number of samples between successive frames
        n_mfcc : int
            Number of MFCCs to extract
            
        Returns:
        --------
        features : dict
            Dictionary containing all extracted features
        """
        self.extract_zcr(frame_length=frame_length, hop_length=hop_length)
        self.extract_ste(frame_length=frame_length, hop_length=hop_length)
        self.extract_mfcc(n_mfcc=n_mfcc, frame_length=frame_length, hop_length=hop_length)
        
        return self.features
    
    def get_feature_summary(self):
        """
        Get a summary of the extracted features
        
        Returns:
        --------
        summary : dict
            Dictionary containing summary statistics of the features
        """
        if not self.features:
            print("No features extracted. Please extract features first.")
            return None
        
        summary = {
            'zcr_mean': self.features.get('zcr_mean'),
            'zcr_std': self.features.get('zcr_std'),
            'ste_mean': self.features.get('ste_mean'),
            'ste_std': self.features.get('ste_std')
        }
        
        # Add MFCC means and stds
        if 'mfcc_mean' in self.features:
            for i, mean_val in enumerate(self.features['mfcc_mean']):
                summary[f'mfcc{i+1}_mean'] = mean_val
            
            for i, std_val in enumerate(self.features['mfcc_std']):
                summary[f'mfcc{i+1}_std'] = std_val
        
        return summary
    
    def plot_waveform(self, ax=None, title=None):
        """
        Plot the waveform of the audio signal
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None, a new figure is created
        title : str, optional
            Title for the plot
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes containing the plot
        """
        if not hasattr(self, 'audio'):
            print("No audio loaded. Please load audio first.")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        
        librosa.display.waveshow(self.audio, sr=self.sr, ax=ax)
        ax.set_title(title or f"Waveform: {os.path.basename(self.file_path)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        
        return ax
    
    def plot_zcr(self, ax=None, title=None):
        """
        Plot the Zero Crossing Rate
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None, a new figure is created
        title : str, optional
            Title for the plot
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes containing the plot
        """
        if 'zcr' not in self.features:
            print("ZCR not extracted. Please extract ZCR first.")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        
        times = librosa.times_like(self.features['zcr'], sr=self.sr, hop_length=self.features['zcr_hop_length'])
        ax.plot(times, self.features['zcr'])
        ax.set_title(title or "Zero Crossing Rate")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ZCR")
        
        return ax
    
    def plot_ste(self, ax=None, title=None):
        """
        Plot the Short-Time Energy
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None, a new figure is created
        title : str, optional
            Title for the plot
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes containing the plot
        """
        if 'ste' not in self.features:
            print("STE not extracted. Please extract STE first.")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        
        times = librosa.times_like(self.features['ste'], sr=self.sr, hop_length=self.features['ste_hop_length'])
        ax.plot(times, self.features['ste'])
        ax.set_title(title or "Short-Time Energy")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy")
        
        return ax
    
    def plot_mfcc(self, ax=None, title=None):
        """
        Plot the MFCCs
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None, a new figure is created
        title : str, optional
            Title for the plot
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes containing the plot
        """
        if 'mfcc' not in self.features:
            print("MFCC not extracted. Please extract MFCC first.")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create a mappable object for the colorbar
        img = librosa.display.specshow(
            self.features['mfcc'], 
            sr=self.sr, 
            hop_length=self.features['mfcc_hop_length'],
            x_axis='time',
            ax=ax
        )
        ax.set_title(title or "MFCC")
        fig = ax.figure
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        return ax
    
    def plot_all_features(self, figsize=(12, 16)):
        """
        Plot all features: waveform, ZCR, STE, and MFCC
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure containing the plots
        """
        if not hasattr(self, 'audio'):
            print("No audio loaded. Please load audio first.")
            return None
        
        if not self.features:
            print("No features extracted. Please extract features first.")
            return None
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        self.plot_waveform(axes[0], title=f"Waveform: {os.path.basename(self.file_path)}")
        self.plot_zcr(axes[1], title="Zero Crossing Rate")
        self.plot_ste(axes[2], title="Short-Time Energy")
        self.plot_mfcc(axes[3], title="MFCC")
        
        plt.tight_layout()
        return fig
    
    def compare_features(self, other_analyzer, figsize=(16, 20)):
        """
        Compare features with another SpeechAnalyzer instance
        
        Parameters:
        -----------
        other_analyzer : SpeechAnalyzer
            Another SpeechAnalyzer instance to compare with
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure containing the comparison plots
        """
        if not hasattr(self, 'audio') or not hasattr(other_analyzer, 'audio'):
            print("Audio not loaded in one or both analyzers.")
            return None
        
        if not self.features or not other_analyzer.features:
            print("Features not extracted in one or both analyzers.")
            return None
        
        file1_name = os.path.basename(self.file_path)
        file2_name = os.path.basename(other_analyzer.file_path)
        
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        
        # Waveforms
        self.plot_waveform(axes[0, 0], title=f"Waveform: {file1_name}")
        other_analyzer.plot_waveform(axes[0, 1], title=f"Waveform: {file2_name}")
        
        # ZCR
        self.plot_zcr(axes[1, 0], title=f"ZCR: {file1_name}")
        other_analyzer.plot_zcr(axes[1, 1], title=f"ZCR: {file2_name}")
        
        # STE
        self.plot_ste(axes[2, 0], title=f"STE: {file1_name}")
        other_analyzer.plot_ste(axes[2, 1], title=f"STE: {file2_name}")
        
        # MFCC
        self.plot_mfcc(axes[3, 0], title=f"MFCC: {file1_name}")
        other_analyzer.plot_mfcc(axes[3, 1], title=f"MFCC: {file2_name}")
        
        plt.tight_layout()
        return fig
    
    def compare_feature_stats(self, other_analyzer):
        """
        Compare feature statistics with another SpeechAnalyzer instance
        
        Parameters:
        -----------
        other_analyzer : SpeechAnalyzer
            Another SpeechAnalyzer instance to compare with
            
        Returns:
        --------
        comparison_df : pandas.DataFrame
            DataFrame containing the comparison of feature statistics
        """
        if not self.features or not other_analyzer.features:
            print("Features not extracted in one or both analyzers.")
            return None
        
        file1_name = os.path.basename(self.file_path)
        file2_name = os.path.basename(other_analyzer.file_path)
        
        summary1 = self.get_feature_summary()
        summary2 = other_analyzer.get_feature_summary()
        
        # Create a DataFrame for comparison
        comparison = {
            'Feature': [],
            file1_name: [],
            file2_name: [],
            'Difference': [],
            'Percent Difference (%)': []
        }
        
        # Add all features to the comparison
        for feature in summary1.keys():
            if feature in summary2:
                val1 = summary1[feature]
                val2 = summary2[feature]
                
                if val1 != 0:
                    percent_diff = (val2 - val1) / abs(val1) * 100
                else:
                    percent_diff = float('inf') if val2 != 0 else 0
                
                comparison['Feature'].append(feature)
                comparison[file1_name].append(val1)
                comparison[file2_name].append(val2)
                comparison['Difference'].append(val2 - val1)
                comparison['Percent Difference (%)'].append(percent_diff)
        
        comparison_df = pd.DataFrame(comparison)
        return comparison_df
    
    def plot_feature_comparison_bar(self, other_analyzer, features_to_plot=None, figsize=(14, 8)):
        """
        Plot a bar chart comparing feature statistics
        
        Parameters:
        -----------
        other_analyzer : SpeechAnalyzer
            Another SpeechAnalyzer instance to compare with
        features_to_plot : list, optional
            List of features to plot. If None, plots ZCR and STE means
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure containing the bar chart
        """
        if not self.features or not other_analyzer.features:
            print("Features not extracted in one or both analyzers.")
            return None
        
        if features_to_plot is None:
            features_to_plot = ['zcr_mean', 'ste_mean']
        
        file1_name = os.path.basename(self.file_path)
        file2_name = os.path.basename(other_analyzer.file_path)
        
        summary1 = self.get_feature_summary()
        summary2 = other_analyzer.get_feature_summary()
        
        # Create lists for bar chart
        features = []
        values1 = []
        values2 = []
        
        for feature in features_to_plot:
            if feature in summary1 and feature in summary2:
                features.append(feature)
                values1.append(summary1[feature])
                values2.append(summary2[feature])
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(features))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, values1, width, label=file1_name)
        rects2 = ax.bar(x + width/2, values2, width, label=file2_name)
        
        ax.set_ylabel('Value')
        ax.set_title('Feature Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend()
        
        # Add value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        return fig
    
    def plot_mfcc_comparison(self, other_analyzer, figsize=(14, 10)):
        """
        Plot a comparison of MFCC means
        
        Parameters:
        -----------
        other_analyzer : SpeechAnalyzer
            Another SpeechAnalyzer instance to compare with
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure containing the MFCC comparison plot
        """
        if not self.features or not other_analyzer.features:
            print("Features not extracted in one or both analyzers.")
            return None
        
        if 'mfcc_mean' not in self.features or 'mfcc_mean' not in other_analyzer.features:
            print("MFCC not extracted in one or both analyzers.")
            return None
        
        file1_name = os.path.basename(self.file_path)
        file2_name = os.path.basename(other_analyzer.file_path)
        
        # Get MFCC means
        mfcc_means1 = self.features['mfcc_mean']
        mfcc_means2 = other_analyzer.features['mfcc_mean']
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(mfcc_means1))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, mfcc_means1, width, label=file1_name)
        rects2 = ax.bar(x + width/2, mfcc_means2, width, label=file2_name)
        
        ax.set_ylabel('Mean Value')
        ax.set_title('MFCC Coefficient Means Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'MFCC {i+1}' for i in range(len(mfcc_means1))])
        ax.legend()
        
        plt.tight_layout()
        return fig

# Example usage - Demo code
def analyze_speeches(calm_speech_path, energetic_speech_path):
    """
    Analyze and compare two speech recordings: one calm, one energetic
    
    Parameters:
    -----------
    calm_speech_path : str
        Path to the calm speech audio file
    energetic_speech_path : str
        Path to the energetic speech audio file
        
    Returns:
    --------
    calm_analyzer, energetic_analyzer : tuple of SpeechAnalyzer
        The two SpeechAnalyzer instances for further analysis
    """
    print("=== Analyzing Calm Speech ===")
    calm_analyzer = SpeechAnalyzer(calm_speech_path)
    calm_analyzer.preprocess_audio()
    calm_analyzer.extract_all_features()
    
    print("\n=== Analyzing Energetic Speech ===")
    energetic_analyzer = SpeechAnalyzer(energetic_speech_path)
    energetic_analyzer.preprocess_audio()
    energetic_analyzer.extract_all_features()
    
    # Display individual features
    print("\n=== Feature Visualizations ===")
    print("Generating feature plots for calm speech...")
    calm_analyzer.plot_all_features()
    plt.savefig("calm_speech_features.png")
    
    print("Generating feature plots for energetic speech...")
    energetic_analyzer.plot_all_features()
    plt.savefig("energetic_speech_features.png")
    
    # Compare features
    print("\n=== Feature Comparison ===")
    print("Generating comparison visualizations...")
    fig = calm_analyzer.compare_features(energetic_analyzer)
    plt.savefig("speech_comparison.png")
    
    # Compare feature statistics
    print("Generating statistical comparison...")
    comparison_df = calm_analyzer.compare_feature_stats(energetic_analyzer)
    print(comparison_df)
    comparison_df.to_csv("feature_comparison.csv")
    
    # Generate bar charts for easy comparison
    print("Generating comparison bar charts...")
    
    # Compare ZCR and STE
    basic_features = ['zcr_mean', 'zcr_std', 'ste_mean', 'ste_std']
    fig_basic = calm_analyzer.plot_feature_comparison_bar(energetic_analyzer, basic_features)
    plt.savefig("basic_feature_comparison.png")
    
    # Compare MFCC coefficients
    fig_mfcc = calm_analyzer.plot_mfcc_comparison(energetic_analyzer)
    plt.savefig("mfcc_comparison.png")
    
    print("\n=== Analysis Complete ===")
    print("Visualizations saved as PNG files.")
    print("Feature comparison saved as 'feature_comparison.csv'.")
    
    return calm_analyzer, energetic_analyzer


# Main execution
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    calm_speech_path = "/Users/aditibaheti/Downloads/Speeches of leaders/Indira_Gandhi.mp3"
    energetic_speech_path = "/Users/aditibaheti/Downloads/Speeches of leaders/Lalu_Yadav.mp3"
    
    analyze_speeches(calm_speech_path, energetic_speech_path)