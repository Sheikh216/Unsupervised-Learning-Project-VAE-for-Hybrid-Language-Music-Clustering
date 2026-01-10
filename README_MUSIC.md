# Music Neural Network Project - Quick Start

## 🎵 Audio/Music Dataset Implementation

This project now supports training neural networks on **music and audio datasets** including:
- **GTZAN Genre Collection** (10 music genres)
- **Million Song Dataset (MSD)**
- **Jamendo Dataset**
- **MIR-1K Dataset**
- **Lakh MIDI Dataset**
- **Kaggle Lyrics Datasets**

---

## 📦 Installation

### Step 1: Install Dependencies (Including Audio Libraries)

```bash
pip install -r requirements.txt
```

This will install:
- `librosa` - Audio feature extraction
- `soundfile` - Audio file I/O
- `audioread` - Audio decoding
- Plus all previous dependencies (numpy, matplotlib, etc.)

### Step 2: Download Datasets (Optional)

**The code works WITHOUT downloading datasets!** It will generate sample data for demonstration.

To use real datasets, download from:

#### GTZAN Genre Collection (Recommended for Beginners)
```
URL: http://marsyas.info/downloads/datasets.html
Size: ~1.2 GB
Genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock
Extract to: ./music_data/gtzan/
```

#### Million Song Dataset
```
URL: http://millionsongdataset.com/
Description: Large-scale dataset with audio features
Use: Pre-computed features CSV
```

#### Jamendo Dataset
```
URL: https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-dataset
Description: Songs with metadata and lyrics
```

---

## 🚀 Quick Start Commands

### Option 1: Train with Sample Data (NO DOWNLOAD NEEDED)

This is the **easiest way to start** - uses automatically generated sample data:

```bash
python main_music.py --dataset gtzan
```

The code will:
- Detect that GTZAN is not downloaded
- Generate realistic sample music features
- Train a neural network on the sample data
- Show you all the visualizations

### Option 2: Train with Real GTZAN Dataset

If you downloaded GTZAN:

```bash
python main_music.py --dataset gtzan --data-dir ./music_data
```

### Option 3: Use Pre-computed Features

If you have a pre-computed features file:

```bash
python main_music.py --dataset gtzan --features-file ./gtzan_features.pkl
```

### Option 4: Use CSV with Features (MSD, Jamendo)

```bash
python main_music.py --dataset msd --csv-file ./msd_features.csv
```

---

## 🎯 Common Training Commands

### Quick Test (Fast, 5 minutes)
```bash
python main_music.py --dataset gtzan --epochs 10 --batch-size 64
```

### High Accuracy (Better results, 15 minutes)
```bash
python main_music.py --dataset gtzan --architecture 256 128 64 --epochs 100 --learning-rate 0.01 --save-model
```

### With Pre-computed Features
```bash
python main_music.py --dataset gtzan --features-file gtzan_features.pkl --architecture 256 128 --epochs 50
```

### Custom Architecture for Music
```bash
python main_music.py --dataset gtzan --architecture 512 256 128 64 --optimizer momentum --learning-rate 0.05
```

---

## 📊 What Audio Features Are Extracted?

When processing real audio files, the system extracts:

1. **MFCCs (Mel-Frequency Cepstral Coefficients)** - 13 coefficients + std (26 features)
2. **Spectral Features**:
   - Spectral Centroid
   - Spectral Rolloff
   - Spectral Bandwidth
   - Zero Crossing Rate
3. **Rhythm Features**:
   - Tempo (BPM)
4. **Chroma Features** - 12 features

**Total: 43 audio features per song**

---

## 🎼 Dataset Information

### GTZAN Genre Collection
- **Tracks**: 1,000 (100 per genre)
- **Duration**: 30 seconds each
- **Genres**: 10 (Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock)
- **Format**: WAV files
- **Expected Accuracy**: 70-85%

### Million Song Dataset
- **Tracks**: 1,000,000 (subset available)
- **Features**: Pre-computed audio features
- **Metadata**: Artist, title, year, etc.
- **Format**: HDF5 or CSV

### Jamendo Dataset
- **Tracks**: Thousands
- **Features**: Audio + Metadata + Lyrics
- **Format**: CSV with audio features

---

## 💻 Python API Usage

```python
from audio_data_loader import AudioDataLoader
from neural_network import NeuralNetwork
from trainer import Trainer

# Load music data (uses sample data if GTZAN not found)
loader = AudioDataLoader('gtzan')
(X_train, y_train), (X_test, y_test) = loader.load_data()

print(f"Features shape: {X_train.shape}")  # (n_samples, 43)
print(f"Labels shape: {y_train.shape}")    # (n_samples, 10)

# Create neural network for music classification
model = NeuralNetwork(
    layer_sizes=[43, 128, 64, 10],  # 43 audio features -> 10 genres
    activations=['relu', 'relu', 'softmax'],
    loss='categorical_crossentropy'
)

# Train
trainer = Trainer(model, learning_rate=0.01, epochs=50)
history = trainer.train(X_train, y_train)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Music Genre Classification Accuracy: {test_acc:.2%}")
```

---

## 📈 Expected Results

### With Sample Data (No Download)
- **Accuracy**: 40-60% (random sample data)
- **Training Time**: 5-10 minutes
- **Purpose**: Testing and demonstration

### With Real GTZAN Dataset
- **Accuracy**: 70-85%
- **Training Time**: 15-30 minutes (first time extracts features)
- **Purpose**: Actual music genre classification

---

## 🔧 Advanced Options

### Extract Features from Your Own Audio Files

```python
from audio_data_loader import AudioDataLoader

loader = AudioDataLoader('gtzan')

# Extract features from a single audio file
features = loader.extract_audio_features('path/to/song.wav')
print(f"Extracted features: {features.shape}")  # (43,)
```

### Use Pre-computed Features

To avoid re-extracting features every time:

1. First run extracts features and saves to `gtzan_features.pkl`
2. Subsequent runs: Use `--features-file gtzan_features.pkl`

```bash
# First run (slow - extracts features)
python main_music.py --dataset gtzan

# Second run (fast - loads pre-computed features)
python main_music.py --dataset gtzan --features-file music_data/gtzan_features.pkl
```

---

## 🎨 Visualizations

The training generates:

1. **Training History** - Loss and accuracy curves
2. **Confusion Matrix** - Shows which genres are confused
3. **Genre Classification Report** - Precision, recall, F1 per genre
4. **Weight Distributions** - Network weight analysis

All saved to `./music_results/`

---

## 🆘 Troubleshooting

### Issue: "librosa not found"
```bash
pip install librosa soundfile audioread
```

### Issue: "GTZAN dataset not found"
**Solution**: The code automatically generates sample data! Just run:
```bash
python main_music.py --dataset gtzan
```

### Issue: Feature extraction is slow
**Solutions**:
1. Use pre-computed features: `--features-file gtzan_features.pkl`
2. Process fewer files (code limits to 100 per genre)
3. Use sample data mode (no real audio processing)

### Issue: "ffmpeg not found" (Windows)
```bash
# Download ffmpeg from: https://ffmpeg.org/download.html
# Add to system PATH
```

---

## 📚 Dataset Download Links

Run this command to see all download links:
```bash
python main_music.py --show-download-info
```

Or check these URLs directly:

1. **GTZAN**: http://marsyas.info/downloads/datasets.html
2. **Million Song Dataset**: http://millionsongdataset.com/
3. **Jamendo**: https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-dataset
4. **MIR-1K**: https://sites.google.com/site/unvoicedsoundseparation/mir-1k
5. **Lakh MIDI**: https://colinraffel.com/projects/lmd/
6. **Lyrics**: https://www.kaggle.com/datasets?search=lyrics

---

## 🎵 Example Workflow

### Complete Workflow (No Dataset Download Required)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train with sample data (works immediately!)
python main_music.py --dataset gtzan --epochs 30

# 3. Check results in ./music_results/
```

### With Real GTZAN Dataset

```bash
# 1. Download GTZAN from http://marsyas.info/downloads/datasets.html
# 2. Extract to ./music_data/gtzan/
# 3. Train (first time extracts features - takes longer)
python main_music.py --dataset gtzan --epochs 50 --architecture 256 128

# 4. Future runs use cached features (much faster)
python main_music.py --dataset gtzan --features-file music_data/gtzan_features.pkl
```

---

## ✨ Key Differences from Image Version

| Aspect | Image (MNIST) | Music (GTZAN) |
|--------|--------------|---------------|
| **Input** | 784 pixels | 43 audio features |
| **Data** | Built-in download | Manual download OR sample data |
| **Features** | Raw pixels | MFCCs, Spectral, Rhythm |
| **Classes** | 10 digits | 10 genres |
| **Accuracy** | ~97% | ~70-85% |
| **Preprocessing** | Normalization | Feature extraction + scaling |

---

## 🎯 Next Steps

1. **Start Simple**: Run with sample data first
2. **Try Real Data**: Download GTZAN if interested
3. **Experiment**: Try different architectures for music
4. **Extend**: Add lyrics processing, combine audio+text features

---

## 📖 Documentation Files

- `README_MUSIC.md` - Detailed music project documentation
- `audio_data_loader.py` - Audio dataset loader with feature extraction
- `main_music.py` - Music training script
- Original `main.py` - Still works for image datasets (MNIST, etc.)

---

**You can use BOTH versions:**
- `main.py` for image classification (MNIST, Fashion-MNIST, CIFAR-10)
- `main_music.py` for music classification (GTZAN, MSD, Jamendo)

Happy music classification! 🎵🎸🎹
