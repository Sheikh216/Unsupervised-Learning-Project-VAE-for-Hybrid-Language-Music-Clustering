```markdown
# Start Hybrid Music Clustering Project

## Quick Start

1️⃣ **Install dependencies**:
```bash
pip install -r requirements.txt
```

2️⃣ **Download datasets**:
```bash
# Download GTZAN audio dataset
python download_gtzan.py

# Download lyrics dataset (creates synthetic if not available)
python download_lyrics.py

# Optional: Download Spotify dataset
python download_spotify.py ( tried but didn't able to work ) 

# upload bangla.csv 
```

3️⃣ **Run experiments** (choose one based on task):

### 🟢 **Easy Task** - Audio-only clustering
```bash
python run_experiments.py \
  --use-audio \
  --clusters 10 \
  --epochs 50 \
  --output ./results/easy_task_results.csv
```

### 🟡 **Medium Task** - Audio + Lyrics hybrid clustering
```bash
python run_experiments.py \
  --use-audio \
  --use-lyrics \
  --clusters 10 \
  --epochs 50 \
  --output ./results/medium_task_results.csv
```

### 🔴 **Hard Task** - Advanced VAEs with label conditioning
```bash
# Beta-VAE
python run_experiments.py \
  --use-audio \
  --use-lyrics \
  --clusters 10 \
  --beta 1.5 \
  --epochs 50 \
  --output ./results/hard_betavae_results.csv

# CVAE (if language labels exist)
python run_experiments.py \
  --use-audio \
  --use-lyrics \
  --clusters 10 \
  --beta 1.0 \
  --epochs 50 \
  --output ./results/hard_cvae_results.csv
```

# run with bangla.csv

python3 run_experiments.py \
    --bangla_csv ./music_data/bangla.csv \
    --epochs 50 \ 
    --latent-dim 16 \
    --beta 1.0 \
    --clusters 10 \
    --tfidf-features 5000

4️⃣ **View results**:
- Check `./results/clustering_metrics.csv` for all metrics
- View visualizations in `./results/latent_visualization/`:
  - `pca_kmeans_tsne.png` - PCA baseline
  - `ae_kmeans_tsne.png` - Autoencoder clusters
  - `vae_beta1.0_kmeans_tsne.png` - Beta-VAE clusters
  - `cvae_kmeans_tsne.png` - Conditional VAE clusters
  - Plus UMAP versions of each

5️⃣ **Run all tasks sequentially** (recommended):
```bash
# Create a run_all.sh file with:
#!/bin/bash
echo "=== EASY TASK ==="
python run_experiments.py --use-audio --clusters 10 --epochs 30 --output ./results/easy.csv

echo "=== MEDIUM TASK ==="
python run_experiments.py --use-audio --use-lyrics --clusters 10 --epochs 30 --output ./results/medium.csv

echo "=== HARD TASK ==="
python run_experiments.py --use-audio --use-lyrics --clusters 10 --beta 1.5 --epochs 30 --output ./results/hard.csv

# Make executable and run:
chmod +x run_all.sh
./run_all.sh
```

## Project Structure
```
music-clustering-vae/
├── run_experiments.py          # MAIN SCRIPT - runs all tasks
├── download_*.py               # Dataset downloaders
├── src/                        # Core modules
│   ├── vae.py                  # VAE implementations
│   ├── dataset.py              # Data loading
│   ├── clustering.py           # Clustering algorithms
│   ├── evaluation.py           # Metrics (Silhouette, NMI, etc.)
│   ├── unsupervised_viz.py     # t-SNE/UMAP visualizations
│   └── audio_data_loader.py    # Audio feature extraction
├── music_data/                 # Datasets (auto-created)
├── results/                    # Outputs (auto-created)
└── requirements.txt            # Python dependencies
```

## Output Files
After running, you'll get:
- **CSV files**: Clustering metrics (Silhouette, Calinski-Harabasz, NMI, ARI, Purity)
- **PNG images**: t-SNE and UMAP visualizations of latent space


## For Multilingual Experiments (Bangla/Hindi)
1. Place Bangla lyrics CSV at `music_data/bangla_lyrics.csv`
2. Modify `dataset.py` to load both English and Bangla
3. Run with language conditioning:
```bash
python run_experiments.py --use-lyrics --language-mode multilingual --clusters 10 --beta 1.5
```

That's it! Your hybrid music clustering results are ready for analysis and reporting.
```
