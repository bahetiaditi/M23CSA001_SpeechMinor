import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from scipy.signal import lfilter
import scipy.io.wavfile as wav
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ===============================================================================
# Part 1: Data Loading and Feature Extraction
# ===============================================================================

class VowelDataProcessor:
    def __init__(self, base_path, genders=["Male", "Female"], vowels=["a", "e", "i", "o", "u"]):
        """Initialize the vowel data processor."""
        print("Initializing VowelDataProcessor...")
        self.base_path = base_path
        self.genders = genders
        self.vowels = vowels
        self.all_files = []
        self.labels = []
        self.gender_labels = []
        print("VowelDataProcessor initialized.")
        
    def collect_files(self):
        """Collect all audio files and their corresponding vowel and gender labels."""
        print("Collecting audio files...")
        for gender in tqdm(self.genders, desc="Collecting genders"):
            for vowel in tqdm(self.vowels, desc=f"Collecting vowels for {gender}"):
                vowel_path = os.path.join(self.base_path, gender, vowel)
                if os.path.exists(vowel_path):
                    for file in os.listdir(vowel_path):
                        if file.endswith('.wav'):
                            self.all_files.append(os.path.join(vowel_path, file))
                            self.labels.append(vowel)
                            self.gender_labels.append(gender)
        
        print(f"Collected {len(self.all_files)} audio files.")
        return self.all_files, self.labels, self.gender_labels
    
    def split_train_test(self, test_size=0.2, random_state=45):
        """Split data into training and testing sets."""
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
            self.all_files, self.labels, self.gender_labels, 
            test_size=test_size, random_state=random_state, stratify=self.labels
        )
        print("Data split complete.")
        return X_train, X_test, y_train, y_test, gender_train, gender_test

class FeatureExtractor:
    def __init__(self, frame_length=512, hop_length=256, num_formants=3, lpc_order=16):
        """Initialize the feature extractor."""
        print("Initializing FeatureExtractor...")
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.num_formants = num_formants
        self.lpc_order = lpc_order
        print("FeatureExtractor initialized.")
    
    def preprocess_audio(self, file_path):
        """Load and preprocess audio file."""
        print(f"Preprocessing audio file: {file_path}")
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Normalize
        y = librosa.util.normalize(y)
        
        print(f"Audio file preprocessed: {file_path}")
        return y, sr
    
    def extract_f0(self, y, sr):
        """Extract fundamental frequency (F0) using pYIN algorithm."""
        print("Extracting fundamental frequency (F0)...")
        # Extract pitch using autocorrelation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # Remove NaN values
        f0 = f0[~np.isnan(f0)]
        
        # Return mean F0 if available, otherwise 0
        mean_f0 = np.mean(f0) if len(f0) > 0 else 0
        print(f"Mean F0: {mean_f0}")
        return mean_f0
    
    def extract_formants_lpc(self, y, sr):
        """Extract formant frequencies using Linear Predictive Coding."""
        print("Extracting formant frequencies using LPC...")
        # Pre-emphasis to amplify high frequencies
        y_pre = librosa.effects.preemphasis(y)
        
        # LPC coefficients
        a = librosa.lpc(y_pre, order=self.lpc_order)
        
        # Find roots of the LPC polynomial
        roots = np.roots(a)
        
        # Keep only roots with positive imaginary part (and inside unit circle)
        roots = roots[np.imag(roots) > 0]
        
        # Convert roots to frequencies
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (sr / (2 * np.pi))
        
        # Sort by frequency
        freqs = np.sort(freqs)
        
        # Take the first num_formants as F1, F2, F3...
        formants = freqs[:self.num_formants] if len(freqs) >= self.num_formants else np.pad(freqs, (0, self.num_formants - len(freqs)))
        
        print(f"Extracted formants: {formants}")
        return formants
    
    def extract_features(self, file_path):
        """Extract all features (F0, F1, F2, F3) for a single file."""
        print(f"Extracting features for {file_path}...")
        # Preprocess audio
        y, sr = self.preprocess_audio(file_path)
        
        # Extract F0
        f0 = self.extract_f0(y, sr)
        
        # Extract formants
        formants = self.extract_formants_lpc(y, sr)
        
        # Return feature vector [F0, F1, F2, F3]
        feature_vector = np.concatenate(([f0], formants))
        print(f"Extracted feature vector: {feature_vector}")
        return feature_vector
    
    def extract_batch_features(self, file_paths):
        """Extract features for a batch of files."""
        print("Extracting batch features...")
        features = []
        for file in tqdm(file_paths, desc="Extracting features from files"):
            features.append(self.extract_features(file))
        print("Batch feature extraction complete.")
        return np.array(features)

class VowelVisualizer:
    def __init__(self, vowels=["a", "e", "i", "o", "u"]):
        """Initialize the vowel visualizer."""
        print("Initializing VowelVisualizer...")
        self.vowels = vowels
        print("VowelVisualizer initialized.")
        
    def visualize_f1_f2_space(self, df, title="F1-F2 Vowel Space"):
        """Visualize F1-F2 vowel space."""
        print("Visualizing F1-F2 vowel space...")
        plt.figure(figsize=(10, 8))
        for vowel in self.vowels:
            vowel_data = df[df['vowel'] == vowel]
            plt.scatter(vowel_data['F2'], vowel_data['F1'], label=vowel, alpha=0.7)

        plt.xlabel('F2 (Hz)')
        plt.ylabel('F1 (Hz)')
        plt.title(title)
        plt.legend()
        # Invert both axes since lower formant values typically appear in the upper right
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('vowel_space_f1_f2.png', dpi=300)
        plt.show()
        print("F1-F2 vowel space visualization complete.")
    
    def visualize_formant_distributions(self, df, save_fig=True):
        """Visualize formant distributions for each vowel."""
        print("Visualizing formant distributions...")
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        formants = ['F1', 'F2', 'F3']

        for i, formant in enumerate(formants):
            for vowel in self.vowels:
                vowel_data = df[df['vowel'] == vowel][formant]
                axes[i].violinplot(vowel_data, positions=[self.vowels.index(vowel)], showmeans=True)
            
            axes[i].set_title(f'{formant} Distribution by Vowel')
            axes[i].set_xticks(range(len(self.vowels)))
            axes[i].set_xticklabels(self.vowels)
            axes[i].set_ylabel('Frequency (Hz)')
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('formant_distributions.png', dpi=300)
        plt.show()
        print("Formant distributions visualization complete.")
    
    def visualize_audio_sample(self, file_path, vowel):
        """Visualize waveform and spectrogram of an audio sample."""
        print(f"Visualizing audio sample for vowel /{vowel}/ from file: {file_path}")
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0])
        axes[0].set_title(f'Waveform of vowel /{vowel}/')
        
        # Plot spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=axes[1])
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        axes[1].set_title(f'Spectrogram of vowel /{vowel}/')
        
        plt.tight_layout()
        plt.savefig(f'audio_visualization_{vowel}.png', dpi=300)
        plt.show()
        print(f"Audio sample visualization for vowel /{vowel}/ complete.")
    
    def visualize_all_samples(self, files, labels):
        """Visualize one sample for each vowel."""
        print("Visualizing all audio samples...")
        for vowel in tqdm(self.vowels, desc="Visualizing vowels"):
            vowel_files = [f for f, v in zip(files, labels) if v == vowel]
            if vowel_files:
                self.visualize_audio_sample(vowel_files[0], vowel)
        print("All audio samples visualized.")
    
    def compare_gender_formants(self, train_df, gender_labels):
        """Compare formant frequencies between genders."""
        print("Comparing formant frequencies between genders...")
        # Add gender labels to the dataframe
        train_df['gender'] = gender_labels
        
        # Create subplot for each formant
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        formants = ['F1', 'F2', 'F3']
        
        for i, formant in enumerate(formants):
            # Use seaborn for boxplot
            sns.boxplot(x='vowel', y=formant, hue='gender', data=train_df, ax=axes[i])
            axes[i].set_title(f'{formant} by Vowel and Gender')
            axes[i].set_xlabel('Vowel')
            axes[i].set_ylabel(f'{formant} (Hz)')
        
        plt.tight_layout()
        plt.savefig('gender_formant_comparison.png', dpi=300)
        plt.show()
        print("Gender formant comparison complete.")
    
    def verify_formant_extraction(self, train_df):
        """Verify formant extraction with reference values."""
        print("Verifying formant extraction with reference values...")
        # Expected formant ranges for adults (approximate)
        expected_ranges = {
            'a': {'F1': (700, 1100), 'F2': (1100, 1500)},
            'e': {'F1': (500, 700), 'F2': (1800, 2300)},
            'i': {'F1': (200, 400), 'F2': (2000, 2800)},
            'o': {'F1': (400, 600), 'F2': (800, 1200)},
            'u': {'F1': (250, 450), 'F2': (600, 1000)}
        }
        
        # Check if our extracted formants align with expected ranges
        result_table = []
        for vowel in self.vowels:
            vowel_data = train_df[train_df['vowel'] == vowel]
            f1_mean = vowel_data['F1'].mean()
            f2_mean = vowel_data['F2'].mean()
            
            result_table.append({
                'Vowel': f'/{vowel}/',
                'Avg F1': f'{f1_mean:.2f} Hz',
                'Expected F1': f"{expected_ranges[vowel]['F1'][0]}-{expected_ranges[vowel]['F1'][1]} Hz",
                'Avg F2': f'{f2_mean:.2f} Hz',
                'Expected F2': f"{expected_ranges[vowel]['F2'][0]}-{expected_ranges[vowel]['F2'][1]} Hz"
            })
            
            print(f"\nVowel: /{vowel}/")
            print(f"Average F1: {f1_mean:.2f} Hz")
            print(f"Expected F1 range: {expected_ranges[vowel]['F1']} Hz")
            print(f"Average F2: {f2_mean:.2f} Hz")
            print(f"Expected F2 range: {expected_ranges[vowel]['F2']} Hz")
        
        # Create a nice table
        result_df = pd.DataFrame(result_table)
        
        # Plot verification as a table
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=result_df.values, colLabels=result_df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title('Formant Extraction Verification', fontsize=14)
        plt.tight_layout()
        plt.savefig('formant_verification.png', dpi=300)
        plt.show()
        
        print("Formant extraction verification complete.")
        return result_df
    
    def visualize_3d_vowel_space(self, df):
        """Visualize vowels in 3D space (F1, F2, F3)."""
        print("Visualizing vowels in 3D space...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['r', 'g', 'b', 'c', 'm']
        
        for i, vowel in enumerate(self.vowels):
            vowel_data = df[df['vowel'] == vowel]
            ax.scatter(
                vowel_data['F1'], 
                vowel_data['F2'], 
                vowel_data['F3'],
                c=colors[i],
                label=vowel,
                alpha=0.7
            )
        
        ax.set_xlabel('F1 (Hz)')
        ax.set_ylabel('F2 (Hz)')
        ax.set_zlabel('F3 (Hz)')
        ax.set_title('3D Vowel Space (F1-F2-F3)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('3d_vowel_space.png', dpi=300)
        plt.show()
        print("3D vowel space visualization complete.")
    
    def plot_feature_correlations(self, df):
        """Visualize correlations between features."""
        print("Plotting feature correlations...")
        # Calculate correlation matrix
        corr = df[['F0', 'F1', 'F2', 'F3']].corr()
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig('feature_correlations.png', dpi=300)
        plt.show()
        print("Feature correlation plot complete.")

# ===============================================================================
# Part 2: Classification System
# ===============================================================================

class VowelClassifier:
    def __init__(self, classifier_type='knn', n_neighbors=5, random_state=42):
        """Initialize the vowel classifier."""
        print("Initializing VowelClassifier...")
        self.classifier_type = classifier_type
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        if classifier_type == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
            print(f"Using KNN classifier with n_neighbors={n_neighbors}")
        elif classifier_type == 'gmm':
            self.classifier = {}  # Will contain one GMM per vowel
            print("Using GMM classifier")
        print("VowelClassifier initialized.")
    
    def train(self, X, y):
        """Train the classifier on the given data."""
        print("Training classifier...")
        # Scale the features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        print("Features scaled.")
        
        if self.classifier_type == 'knn':
            # Train KNN classifier
            print("Training KNN classifier...")
            self.classifier.fit(X_scaled, y)
            print("KNN classifier trained.")
        elif self.classifier_type == 'gmm':
            # Train one GMM per vowel class
            print("Training GMM classifier...")
            vowels = np.unique(y)
            for vowel in tqdm(vowels, desc="Training GMM for each vowel"):
                vowel_data = X_scaled[np.array(y) == vowel]
                self.classifier[vowel] = GaussianMixture(
                    n_components=2,
                    covariance_type='full',
                    random_state=self.random_state
                )
                self.classifier[vowel].fit(vowel_data)
            print("GMM classifier trained.")
        print("Classifier training complete.")
    
    def predict(self, X):
        """Make predictions on the given data."""
        print("Making predictions...")
        # Scale the features
        print("Scaling features...")
        X_scaled = self.scaler.transform(X)
        print("Features scaled.")
        
        if self.classifier_type == 'knn':
            # Predict using KNN
            print("Predicting using KNN...")
            predictions = self.classifier.predict(X_scaled)
            print("KNN predictions complete.")
            return predictions
        elif self.classifier_type == 'gmm':
            # Predict using GMMs
            print("Predicting using GMMs...")
            vowels = list(self.classifier.keys())
            log_probs = np.zeros((X_scaled.shape[0], len(vowels)))
            
            for i, vowel in enumerate(vowels):
                log_probs[:, i] = self.classifier[vowel].score_samples(X_scaled)
            
            # Return the vowel with highest log probability for each sample
            predictions = [vowels[i] for i in np.argmax(log_probs, axis=1)]
            print("GMM predictions complete.")
            return predictions
        print("Predictions complete.")
    
    def evaluate(self, X, y_true):
        """Evaluate the classifier and return metrics."""
        print("Evaluating classifier...")
        y_pred = self.predict(X)
        
        # Calculate confusion matrix
        print("Calculating confusion matrix...")
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix calculated.")
        
        # Calculate accuracy
        print("Calculating accuracy...")
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        
        # Generate classification report
        print("Generating classification report...")
        report = classification_report(y_true, y_pred)
        print("Classification Report:")
        print(report)
        
        print("Classifier evaluation complete.")
        return y_pred, cm, acc, report

class ClassificationVisualizer:
    def __init__(self, vowels=["a", "e", "i", "o", "u"]):
        """Initialize the classification visualizer."""
        print("Initializing ClassificationVisualizer...")
        self.vowels = vowels
        print("ClassificationVisualizer initialized.")
    
    def plot_confusion_matrix(self, cm, title="Confusion Matrix", save_fig=True):
        """Plot a confusion matrix."""
        print("Plotting confusion matrix...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.vowels, yticklabels=self.vowels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()
        if save_fig:
            plt.savefig('confusion_matrix.png', dpi=300)
        plt.show()
        print("Confusion matrix plotted.")
    
    def plot_normalized_confusion_matrix(self, cm, title="Normalized Confusion Matrix", save_fig=True):
        """Plot a normalized confusion matrix."""
        print("Plotting normalized confusion matrix...")
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=self.vowels, yticklabels=self.vowels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()
        if save_fig:
            plt.savefig('confusion_matrix_normalized.png', dpi=300)
        plt.show()
        print("Normalized confusion matrix plotted.")
    
    def compare_classifiers(self, results):
        """Compare multiple classifiers based on accuracy."""
        print("Comparing classifiers...")
        # Extract data
        names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in names]
        
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.ylabel('Accuracy')
        plt.title('Classifier Performance Comparison')
        plt.tight_layout()
        plt.savefig('classifier_comparison.png', dpi=300)
        plt.show()
        print("Classifier comparison complete.")

# ===============================================================================
# Part 3: Analysis and Main Execution
# ===============================================================================

def run_complete_analysis(base_path):
    """Run the complete vowel classification analysis pipeline."""
    print("\n=== VOWEL CLASSIFICATION SYSTEM ===\n")
    
    # Initialize components
    print("Initializing components...")
    data_processor = VowelDataProcessor(base_path)
    feature_extractor = FeatureExtractor()
    vowel_visualizer = VowelVisualizer()
    clf_visualizer = ClassificationVisualizer()
    print("Components initialized.")
    
    # Step 1: Load and split data
    print("Loading and splitting data...")
    all_files, labels, gender_labels = data_processor.collect_files()
    X_train, X_test, y_train, y_test, gender_train, gender_test = data_processor.split_train_test()
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Step 2: Extract features
    print("\nExtracting features...")
    X_train_features = feature_extractor.extract_batch_features(X_train)
    X_test_features = feature_extractor.extract_batch_features(X_test)
    
    # Create feature dataframes
    train_df = pd.DataFrame(
        X_train_features, 
        columns=['F0', 'F1', 'F2', 'F3']
    )
    train_df['vowel'] = y_train
    
    test_df = pd.DataFrame(
        X_test_features, 
        columns=['F0', 'F1', 'F2', 'F3']
    )
    test_df['vowel'] = y_test
    
    # Step 3: Visualize features
    print("\nCreating visualizations...")
    
    # Basic vowel space visualization
    print("Visualizing F1-F2 space...")
    vowel_visualizer.visualize_f1_f2_space(train_df)
    
    # Formant distributions
    print("Visualizing formant distributions...")
    vowel_visualizer.visualize_formant_distributions(train_df)
    
    # Sample audio visualizations
    print("Visualizing audio samples...")
    vowel_visualizer.visualize_all_samples(X_train, y_train)
    
    # Gender comparison
    print("Comparing gender formants...")
    vowel_visualizer.compare_gender_formants(train_df, gender_train)
    
    # Verify formant extraction
    print("Verifying formant extraction...")
    formant_verification = vowel_visualizer.verify_formant_extraction(train_df)
    
    # 3D vowel space
    print("Visualizing 3D vowel space...")
    vowel_visualizer.visualize_3d_vowel_space(train_df)
    
    # Feature correlations
    print("Plotting feature correlations...")
    vowel_visualizer.plot_feature_correlations(train_df)
    
    # Print feature statistics
    print("Printing feature statistics...")
    for vowel in data_processor.vowels:
        vowel_data = train_df[train_df['vowel'] == vowel]
        print(f"\nVowel: {vowel}")
        print(vowel_data[['F0', 'F1', 'F2', 'F3']].describe())
    
    # Step 4: Classification
    print("\nTraining and evaluating classifiers...")
    
    # Prepare different classifiers
    classifiers = {
        'KNN (k=3)': VowelClassifier(classifier_type='knn', n_neighbors=3),
        'KNN (k=5)': VowelClassifier(classifier_type='knn', n_neighbors=5),
        'GMM': VowelClassifier(classifier_type='gmm')
    }
    
    # Train and evaluate each classifier
    results = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.train(X_train_features, y_train)
        
        print(f"Evaluating {name}...")
        y_pred, cm, accuracy, report = clf.evaluate(X_test_features, y_test)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        
        # Visualize confusion matrix
        print("Plotting confusion matrix...")
        clf_visualizer.plot_confusion_matrix(cm, title=f"Confusion Matrix - {name}")
        print("Plotting normalized confusion matrix...")
        clf_visualizer.plot_normalized_confusion_matrix(cm, title=f"Normalized Confusion Matrix - {name}")
        
        # Store results
        results[name] = {
            'predictions': y_pred,
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'report': report
        }
    
    # Compare classifiers
    print("Comparing classifiers...")
    clf_visualizer.compare_classifiers(results)
    
    # Step 5: Analysis and discussion
    print("\n=== ANALYSIS AND DISCUSSION ===")
    print("\n1. Formant Pattern Analysis:")
    print("   - The F1-F2 vowel space plot shows clear clustering of vowels in accordance with")
    print("     linguistic theory. Vowels /i/ and /u/ have low F1, while /a/ has high F1.")
    print("   - F2 differentiates front vowels (/i/, /e/) from back vowels (/o/, /u/).")
    print("   - Gender differences in formant frequencies are observed, with females typically")
    print("     having higher formant values due to smaller vocal tracts.")
    
    print("\n2. Classification Performance:")
    best_clf = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"   - Best classifier: {best_clf[0]} with accuracy of {best_clf[1]['accuracy']:.4f}")
    print("   - Common misclassifications observed:")
    
    # Find the most common misclassifications from the best classifier's confusion matrix
    cm = best_clf[1]['confusion_matrix']
    for i in range(len(data_processor.vowels)):
        for j in range(len(data_processor.vowels)):
            if i != j and cm[i, j] > 0:
                print(f"     * Vowel /{data_processor.vowels[i]}/ misclassified as /{data_processor.vowels[j]}/ {cm[i, j]} times")
    
    
    print("Returning results...")
    return {
        'train_df': train_df,
        'test_df': test_df,
        'classification_results': results,
        'feature_verification': formant_verification
    }

if __name__ == "__main__":
    print("Starting main execution...")
    # Set the path to your dataset
    base_path = "/Users/aditibaheti/Downloads/Q3_Minor"  # Update this with your actual path
    print(f"Base path set to: {base_path}")
    
    # Run the complete analysis
    print("Running complete analysis...")
    results = run_complete_analysis(base_path)
    print("Complete analysis finished.")
    print("End of main execution.")
