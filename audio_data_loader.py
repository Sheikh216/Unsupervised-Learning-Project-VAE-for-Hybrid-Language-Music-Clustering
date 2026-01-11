"""
Audio and music dataset loading utilities.
Supports GTZAN, Million Song Dataset, Jamendo, MIR-1K, Lakh MIDI, and lyrics datasets.
"""

import numpy as np
import os
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')


class AudioDataLoader:
    """Handles loading and preprocessing of audio/music datasets."""
    
    def __init__(self, dataset_name='gtzan', data_dir='./music_data', allow_fallback: bool = True):
        """
        Initialize the audio data loader.
        
        Args:
            dataset_name: Name of dataset ('gtzan', 'msd', 'jamendo', 'mir1k', 'lakh_midi')
            data_dir: Directory containing the dataset
            allow_fallback: If False, raises when data is missing instead of generating synthetic samples
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.allow_fallback = allow_fallback
        os.makedirs(data_dir, exist_ok=True)
        
    def extract_audio_features(self, audio_path, sr=22050, n_mfcc=20, n_fft=2048, hop_length=512, duration=10.0):
        """Extract lightweight, stable features (MFCC mean/std only to avoid shape issues)."""
        try:
            y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
            if y is None or y.size == 0:
                return None
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            features = np.concatenate([mfccs_mean, mfccs_std]).astype(np.float32)
            return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def load_gtzan(self, features_file=None):
        """
        Load GTZAN Genre Collection dataset.
        
        Args:
            features_file: Pre-computed features file (optional)
            
        Returns:
            (X_train, y_train), (X_test, y_test)
        """
        print("Loading GTZAN Genre Collection...")
        
        # If pre-computed features exist, load them
        if features_file and os.path.exists(features_file):
            print(f"Loading pre-computed features from {features_file}")
            with open(features_file, 'rb') as f:
                data = pickle.load(f)
            X = data['features']
            y = data['labels']
        else:
            # Extract features from audio files
            gtzan_path = os.path.join(self.data_dir, 'gtzan', 'genres')
            
            if not os.path.exists(gtzan_path):
                msg = (
                    f"GTZAN dataset not found at {gtzan_path}. "
                    "Download from http://marsyas.info/downloads/datasets.html or supply pre-computed features."
                )
                if not self.allow_fallback:
                    raise FileNotFoundError(msg)
                print(msg)
                print("\nGenerating sample data for demonstration...")
                return self._generate_sample_music_data(n_samples=1000, n_features=43, n_classes=10)
            
            genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                     'jazz', 'metal', 'pop', 'reggae', 'rock']
            
            features_list = []
            labels_list = []
            
            print("Extracting audio features (this may take a while)...")
            for genre_idx, genre in enumerate(genres):
                genre_path = os.path.join(gtzan_path, genre)
                if not os.path.exists(genre_path):
                    continue
                
                audio_files = [f for f in os.listdir(genre_path) if f.lower().endswith(('.wav', '.au'))]
                
                for audio_file in audio_files[:100]:  # Limit to 100 per genre
                    audio_path = os.path.join(genre_path, audio_file)
                    features = self.extract_audio_features(audio_path)
                    
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(genre_idx)
                
                print(f"  Processed {genre}: {len([l for l in labels_list if l == genre_idx])} samples")
            
            X = np.array(features_list)
            y = np.array(labels_list)
            
            # Save features for future use
            features_file = os.path.join(self.data_dir, 'gtzan_features.pkl')
            with open(features_file, 'wb') as f:
                pickle.dump({'features': X, 'labels': y}, f)
            print(f"Features saved to {features_file}")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # One-hot encode labels
        from tensorflow import keras
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"Dataset loaded: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        print(f"Feature dimension: {X_train.shape[1]}")
        
        return (X_train, y_train), (X_test, y_test)
    
    def load_million_song_subset(self, csv_file=None):
        """
        Load Million Song Dataset (subset with features).
        
        Args:
            csv_file: Path to CSV file with pre-extracted features
            
        Returns:
            (X_train, y_train), (X_test, y_test)
        """
        print("Loading Million Song Dataset...")
        
        if csv_file and os.path.exists(csv_file):
            print(f"Loading from {csv_file}")
            df = pd.read_csv(csv_file)
            
            # Separate features and labels
            feature_cols = [col for col in df.columns if col not in ['track_id', 'genre', 'artist', 'title']]
            X = df[feature_cols].values
            
            if 'genre' in df.columns:
                le = LabelEncoder()
                y = le.fit_transform(df['genre'].values)
            else:
                y = np.zeros(len(X))  # No labels
            
        else:
            msg = "MSD file not found; provide a CSV with features."
            if not self.allow_fallback:
                raise FileNotFoundError(msg)
            print(msg)
            print("Generating sample data...")
            return self._generate_sample_music_data(n_samples=5000, n_features=50, n_classes=8)
        
        # Split and normalize
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        from tensorflow import keras
        n_classes = len(np.unique(y))
        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)
        
        print(f"Dataset loaded: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        return (X_train, y_train), (X_test, y_test)
    
    def load_jamendo(self, csv_file=None):
        """
        Load Jamendo music dataset.
        
        Args:
            csv_file: Path to CSV with metadata and features
            
        Returns:
            (X_train, y_train), (X_test, y_test)
        """
        print("Loading Jamendo Dataset...")
        
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            # Process similar to MSD
            feature_cols = [col for col in df.columns if 'feature' in col.lower() or 'mfcc' in col.lower()]
            X = df[feature_cols].values
            
            if 'genre' in df.columns:
                le = LabelEncoder()
                y = le.fit_transform(df['genre'].values)
            else:
                y = np.zeros(len(X))
        else:
            msg = "Jamendo file not found; provide metadata/features CSV."
            if not self.allow_fallback:
                raise FileNotFoundError(msg)
            print(msg)
            print("Generating sample data...")
            return self._generate_sample_music_data(n_samples=3000, n_features=45, n_classes=12)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        from tensorflow import keras
        n_classes = len(np.unique(y))
        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)
        
        print(f"Dataset loaded: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        return (X_train, y_train), (X_test, y_test)
    
    def _generate_sample_music_data(self, n_samples=1000, n_features=43, n_classes=10):
        """
        Generate sample music data for demonstration when real data is not available.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            
        Returns:
            (X_train, y_train), (X_test, y_test)
        """
        print(f"\nGenerating {n_samples} sample music features with {n_features} dimensions...")
        
        # Generate realistic-looking music features
        X = np.random.randn(n_samples, n_features)
        
        # Add some structure to make it more realistic
        # MFCCs typically have certain patterns
        for i in range(min(13, n_features)):
            X[:, i] = X[:, i] * (13 - i) / 13  # Decreasing variance for higher MFCCs
        
        # Generate labels
        y = np.random.randint(0, n_classes, n_samples)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # One-hot encode
        from tensorflow import keras
        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)
        
        print(f"Sample data generated: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        print(f"Feature dimension: {X_train.shape[1]}, Classes: {n_classes}")
        
        return (X_train, y_train), (X_test, y_test)
    
    def load_data(self, **kwargs):
        if self.dataset_name == 'gtzan':
            return self.load_gtzan(kwargs.get('features_file'))
        elif self.dataset_name == 'msd':
            return self.load_million_song_subset(kwargs.get('csv_file'))
        elif self.dataset_name == 'jamendo':
            return self.load_jamendo(kwargs.get('csv_file'))
        elif self.dataset_name == 'spotify':  # ADD THIS
            return self.load_spotify(kwargs.get('csv_file'))
        else:
            print(f"Dataset {self.dataset_name} not recognized. Using sample data.")
            return self._generate_sample_music_data()

    def load_bangla_english(self, csv_path):
    
        df = pd.read_csv(csv_path)
        
        # Extract features (TF-IDF on lyrics column)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(df['lyrics']).toarray()
        
        # Create language labels
        y = np.array([0 if lang=='english' else 1 for lang in df['language']])
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # One-hot encode
        from tensorflow import keras
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)
        
        return (X_train, y_train), (X_test, y_test)
    


    def load_spotify(self, csv_file=None):


        # Try to locate CSV
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            # Default location
            default_path = os.path.join(self.data_dir, 'spotify', 'spotify_features.csv')
            if os.path.exists(default_path):
                df = pd.read_csv(default_path)
            else:
                msg = f"Spotify CSV not found at {default_path}"
                if not self.allow_fallback:
                    raise FileNotFoundError(msg)
                print(msg)
                print("Generating sample data...")
                return self._generate_sample_music_data(n_samples=1000, n_features=9, n_classes=10)

        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # 1. Keep only relevant audio features
        audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness',
                        'valence', 'tempo']
        
        # Keep only available features
        available_features = [f for f in audio_features if f in df.columns]
        print(f"Available audio features: {available_features}")

        # 2. Filter by genre if 'genre' column exists
        if 'genre' in df.columns:
            print(f"Original genre count: {df['genre'].nunique()}")
            
            # Keep only tracks with genre labels (non-null)
            df = df.dropna(subset=['genre'])
            
            # Keep only top N most frequent genres (e.g., top 10)
            top_n = 10
            top_genres = df['genre'].value_counts().head(top_n).index
            df = df[df['genre'].isin(top_genres)]
            
            print(f"Keeping top {top_n} genres: {list(top_genres)}")
            print(f"Rows after genre filtering: {len(df)}")
        else:
            print("No 'genre' column found. Will create synthetic genres.")

        # 3. Sample to manageable size (max 1500 tracks)
        max_samples = 1500
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            print(f"Sampled to {max_samples} tracks")

        # 4. Prepare features and labels
        X = df[available_features].values
        
        if 'genre' in df.columns:
            # Encode genres to integers
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(df['genre'].values)
            print(f"Unique genres after encoding: {len(np.unique(y))}")
        else:
            # Create synthetic genres using clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=10, random_state=42)
            y = kmeans.fit_predict(X)
            print("Created 10 synthetic genres using K-Means")

        # 5. Split and normalize
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 6. One-hot encode labels
        from tensorflow import keras
        n_classes = len(np.unique(y))
        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)

        print(f"\n✅ Spotify dataset loaded and filtered:")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {n_classes}")

        return (X_train, y_train), (X_test, y_test)
    

    def load_jamendo(self, csv_file=None):
        if csv_file is None:
            csv_file = os.path.join(self.data_dir, 'jamendo', 'metadata.csv')
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Jamendo metadata not found at {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # DEBUG: Check the CSV content
        print(f"Jamendo CSV columns: {df.columns.tolist()}")
        if 'genre' in df.columns:
            print(f"'genre' column unique values: {df['genre'].unique()}")
            print(f"'genre' column distribution:\n{df['genre'].value_counts()}")
        
        # Extract audio features from audio files
        audio_features = []
        for idx, row in df.iterrows():
            audio_path = os.path.join(self.data_dir, 'jamendo', 'audio', f"{row['track_id']}.mp3")
            feats = self.extract_audio_features(audio_path) if os.path.exists(audio_path) else np.zeros(40)
            audio_features.append(feats)
        
        X_audio = np.array(audio_features)
        
        # Use 'genre' or 'mood' as labels
        if 'genre' in df.columns:
            labels = df['genre'].values  # This is the 'labels' variable
            print(f"Raw genre values: {labels[:20]}")  # First 20 genres
            print(f"Number of unique genres: {len(np.unique(labels))}")
            print(f"Genre distribution:\n{pd.Series(labels).value_counts()}")
            
            le = LabelEncoder()
            y = le.fit_transform(labels)
            
            print(f"Encoded y values: {y[:20]}")
            print(f"Unique encoded classes: {np.unique(y)}")
            print(f"Number of classes: {len(np.unique(y))}")
            
        elif 'genre_top' in df.columns:
            le = LabelEncoder()
            y = le.fit_transform(df['genre_top'])
        else:
            y = np.zeros(len(X_audio))
            print("WARNING: No genre column found. Using all zeros as labels.")
        
        # Split and return
        X_train, X_test, y_train, y_test = train_test_split(
            X_audio, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        from tensorflow import keras
        n_classes = len(np.unique(y))
        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)
        
        print(f"\n✅ Jamendo dataset loaded:")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {n_classes}")
        
        return (X_train, y_train), (X_test, y_test)













def download_dataset_instructions():
    """Print instructions for downloading datasets."""
    print("\n" + "="*70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    print("\n1. GTZAN Genre Collection")
    print("   URL: http://marsyas.info/downloads/datasets.html")
    print("   Description: 1000 audio tracks, 10 genres, 30 seconds each")
    print("   Extract to: ./music_data/gtzan/")
    
    print("\n2. Million Song Dataset (MSD)")
    print("   URL: http://millionsongdataset.com/")
    print("   Description: Large-scale dataset with audio features and metadata")
    print("   Subset: http://millionsongdataset.com/pages/getting-dataset/")
    
    print("\n3. Jamendo Dataset")
    print("   URL: https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-dataset")
    print("   Description: Songs with audio previews, metadata, and lyrics")
    
    print("\n4. MIR-1K Dataset")
    print("   URL: https://sites.google.com/site/unvoicedsoundseparation/mir-1k")
    print("   Description: 1000 song clips in Mandarin and English")
    
    print("\n5. Lakh MIDI Dataset")
    print("   URL: https://colinraffel.com/projects/lmd/")
    print("   Description: MIDI dataset for music modeling")
    
    print("\n6. Kaggle Lyrics Datasets")
    print("   URL: https://www.kaggle.com/datasets?search=lyrics")
    print("   Description: Multiple datasets with song lyrics in various languages")
    
    print("\n" + "="*70)
    print("NOTE: If datasets are not available, the code will generate")
    print("      sample data for demonstration purposes.")
    print("="*70 + "\n")


if __name__ == "__main__":
    download_dataset_instructions()
    
    # Test with sample data
    print("\nTesting audio data loader with sample data...")
    loader = AudioDataLoader('gtzan')
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    print(f"\nLoaded shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
