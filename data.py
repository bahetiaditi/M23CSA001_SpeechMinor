import os
import pandas as pd
import numpy as np
import librosa
from glob import glob

def load_vowel_dataset(base_dir):
    # Define vowel mapping - map each word to its vowel sound
    vowel_mapping = {
        'had': 'a',    # The primary vowel sound is "a"
        'hawed': 'o',  # The primary vowel sound is "o"
        'hayed': 'e',  # The primary vowel sound is "e" in the diphthong "eɪ"
        'head': 'e',   # The primary vowel sound is "e"
        'heed': 'e',   # The primary vowel sound is "i", but since it's often pronounced more like "ee", it can be mapped to "e"
        'herd': 'e',   # The primary vowel sound is "e" in the schwa-like sound "ɜ"
        'hid': 'i',    # The primary vowel sound is "i"
        'hod': 'o',    # The primary vowel sound is "o"
        'hoed': 'o',   # The primary vowel sound is "o" in the diphthong "oʊ"
        'hood': 'o',   # The primary vowel sound is "o" (though it's more like "u" in some pronunciations)
        'hud': 'u',    # The primary vowel sound is more like "u" in the sound "ʌ"
        'whod': 'u'    # The primary vowel sound is "u"
    }

    
    # Define categories based on subdirectories
    categories = {
        'adult_male': 'adult_male',
        'adult_female': 'adult_female',
        'child_7yo': 'child_7yo',
        'child_5yo': 'child_5yo',
        'child_3yo': 'child_3yo'
    }
    
    data = []
    
    # Walk through the directory structure
    for category_id, category_name in categories.items():
        category_dir = os.path.join(base_dir, category_name)
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory {category_dir} not found!")
            continue
            
        # Get all word directories
        word_dirs = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
        
        for word in word_dirs:
            if word not in vowel_mapping:
                print(f"Warning: Unknown word {word}, skipping...")
                continue
                
            vowel = vowel_mapping[word]
            word_dir = os.path.join(category_dir, word)
            
            # Get all audio files in this word directory
            audio_files = glob(os.path.join(word_dir, "*.wav"))
            
            for audio_path in audio_files:
                filename = os.path.basename(audio_path)
                sample_id = os.path.splitext(filename)[0]  # Remove extension
                
                # Try to load the audio file to verify it works
                try:
                    y, sr = librosa.load(audio_path, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    # Add to our data list
                    data.append({
                        'file_path': audio_path,
                        'category': category_id,
                        'category_name': category_name,
                        'word': word,
                        'vowel': vowel,
                        'sample_id': sample_id,
                        'duration': duration,
                        'sample_rate': sr
                    })
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.to_csv('Ques_3.csv')
    print(f"Loaded {len(df)} audio files across {df['category'].nunique()} categories and {df['vowel'].nunique()} vowels")
    
    return df

base_dir='/Users/aditibaheti/Downloads/speechques3'
df = load_vowel_dataset(base_dir)

