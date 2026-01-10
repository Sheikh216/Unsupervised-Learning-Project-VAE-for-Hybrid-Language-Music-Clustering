# Unsupervised Learning Project: VAE for Hybrid Language Music Clustering

Course: Neural Networks  
Prepared By: Moin Mostakim  
Submission Due: January 10th, 2026

---

## 1. Abstract
This project implements an end-to-end unsupervised learning pipeline for hybrid (audio + lyrics) music clustering using Variational Autoencoders (VAEs). We extract 40-dimensional MFCC-based audio features from the GTZAN Genre Collection (1000 tracks, 10 genres) and 500-dimensional TF‑IDF features from a real lyrics dataset (1000 samples), concatenating them into a 540-dimensional representation. We compare PCA+KMeans and Autoencoder baselines against VAE, Beta‑VAE, and Conditional VAE (CVAE), evaluating cluster quality via Silhouette, Calinski‑Harabasz (CH), Davies‑Bouldin (DB), and supervised metrics (ARI, NMI, Purity) using genre labels. The best unsupervised separation is achieved by Autoencoder + Agglomerative (Silhouette = 0.314, CH = 163.7). Supervised alignment is modest (best NMI = 0.051 for PCA+KMeans; best CVAE NMI = 0.023), reflecting expected differences between unsupervised structure and genre annotations. We provide UMAP and t‑SNE visualizations and a reproducible pipeline.

---

## 2. Introduction
Music can be characterized by multiple modalities—acoustics (audio) and semantics (lyrics). Unsupervised learning aims to discover latent structures without labels, which is well suited for analyzing large music corpora where annotations are sparse or subjective. VAEs offer a flexible framework to learn low-dimensional latent variables capturing generative factors of variation. In this work, we:
- Build hybrid audio+lyrics features to reflect both signal and semantic cues.
- Learn latent representations with VAE-family models.
- Cluster the latent space and assess quality with established metrics.
- Compare against PCA and Autoencoder baselines.

Our goals are to understand: (1) how VAE representations structure hybrid music data, (2) whether they improve clustering over baselines, and (3) how unsupervised clusters relate to genre labels.

---

## 3. Related Work
- Variational Autoencoders (Kingma & Welling, 2014) learn probabilistic latent variables with a reconstruction + KL objective; Beta‑VAE (Higgins et al., 2017) balances disentanglement via a β factor; CVAE conditions the latent on auxiliary labels.
- Clustering pipelines commonly use PCA or deep Autoencoders for dimensionality reduction, followed by KMeans/Agglomerative/DBSCAN.
- Music representation: MFCCs and spectrograms for audio; bag-of-words / TF‑IDF for lyrics; multimodal fusion by concatenation or joint models.

Key references are listed in Section 9.

---

## 4. Method
### 4.1 Architectures
- Autoencoder (AE): deterministic encoder/decoder minimizing MSE reconstruction.
- VAE: encoder outputs μ, logσ²; reparameterization z = μ + σ ⊙ ε; loss = reconstruction + β·KL(q(z|x) || p(z)).
- Beta‑VAE: β > 1 to encourage disentanglement.
- CVAE: conditions encoder/decoder on labels (here, language/genre if available).

Latent dimension: 16 for all deep models.

### 4.2 Feature Extraction
- Audio (GTZAN):
  - Loading: 10 s @ 22,050 Hz, mono.
  - Features: MFCC (n_mfcc=20) mean/std → 40 dims.
  - Implementation: `audio_data_loader.py` → cached at [music_data/gtzan_features.pkl](../music_data/gtzan_features.pkl).
- Lyrics:
  - Source: Kaggle lyrics dataset (1000 samples with textual content and genres).
  - Vectorization: TF‑IDF (max_features = 500) using scikit-learn.
- Hybrid Fusion: Concatenate audio (40) + lyrics (500) → 540 dims; per-modality `StandardScaler` before concat in `src/dataset.py`.

### 4.3 Clustering Methods
- KMeans, Agglomerative Clustering, DBSCAN (with noise handling).
- PCA+KMeans baseline and AE+clustering baseline for comparison.

### 4.4 Training
- Optimizer: Adam (default learning rates per Keras).
- Epochs: 50; Batch size: 64; Latent dim: 16; β=1.0.
- Hardware: CPU (oneDNN enabled by TensorFlow).

### 4.5 Evaluation Metrics
Given embeddings Z and predicted clusters ŷ, and optional ground truth labels y (genre):
- Silhouette: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$.
- Calinski‑Harabasz: $\text{CH} = \frac{\operatorname{tr}(B_k)/(k-1)}{\operatorname{tr}(W_k)/(n-k)}$.
- Davies‑Bouldin: $\text{DB} = \frac{1}{k} \sum_i \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d_{ij}}$.
- ARI: Adjusted Rand Index (chance-adjusted pair agreement).
- NMI: $\text{NMI}(U,V) = \frac{2 I(U;V)}{H(U)+H(V)}$.
- Purity: $\text{Purity} = \frac{1}{n} \sum_k \max_j |c_k \cap t_j|$.

Implementation: `src/evaluation.py` with DBSCAN noise filtering for silhouette/CH/DB.

---

## 5. Experiments
### 5.1 Datasets
- GTZAN (Audio): 1000 tracks, 10 genres (100 each). Structure: [music_data/gtzan/genres](../music_data/gtzan/genres).
- Lyrics (Text): 1000 samples; saved at [music_data/lyrics.csv](../music_data/lyrics.csv).

### 5.2 Preprocessing Steps
1. Download GTZAN via `download_gtzan.py` (KaggleHub), copy to `music_data/gtzan/genres`.
2. Extract 40-dim MFCC features (mean/std) from 10-second excerpts; cache features.
3. Load lyrics CSV; TF‑IDF (500 dims); align by sampling to 1000 rows.
4. Standardize audio and lyrics separately; concatenate into 540-dim features.

### 5.3 Hyperparameters
- Latent dim = 16; Epochs = 50; β = 1.0; KMeans clusters = 10.

### 5.4 Training & Scripts
- Main: `run_experiments.py` orchestrates baselines and VAEs, saves metrics/plots.
- Visualizations: t‑SNE/UMAP in `src/unsupervised_viz.py` saved under [results/latent_visualization](../results/latent_visualization).
- Additional plots: `scripts/plot_metrics.py` creates metric bar charts in [results/analysis_plots](../results/analysis_plots).

### 5.5 Reproduce
```bash
# Windows PowerShell
C:/Users/USERAS/Desktop/715_Project/.venv/Scripts/python.exe download_gtzan.py
C:/Users/USERAS/Desktop/715_Project/.venv/Scripts/python.exe download_lyrics.py
C:/Users/USERAS/Desktop/715_Project/.venv/Scripts/python.exe run_experiments.py --data-dir ./music_data --lyrics-csv ./music_data/lyrics.csv --use-lyrics --use-audio --epochs 50 --latent-dim 16 --clusters 10
C:/Users/USERAS/Desktop/715_Project/.venv/Scripts/python.exe scripts/plot_metrics.py
```

---

## 6. Results
All raw metrics are logged in [results/clustering_metrics.csv](../results/clustering_metrics.csv). Summary (hybrid features, 540 dims):

| Method | Silhouette ↑ | CH ↑ | DB ↓ | ARI ↑ | NMI ↑ | Purity ↑ |
|--------|--------------:|-----:|-----:|------:|------:|---------:|
| PCA + KMeans | 0.239 | 92.312 | 1.763 | 0.018 | 0.051 | 0.174 |
| AE + KMeans | 0.309 | 157.390 | 1.351 | -0.000 | 0.018 | 0.148 |
| AE + Agglomerative | 0.314 | 163.680 | 1.327 | 0.001 | 0.020 | 0.151 |
| AE + DBSCAN | — | — | — | 0.000 | 0.000 | 0.100 |
| VAE (β=1.0) + KMeans | 0.182 | 75.113 | 1.817 | 0.001 | 0.022 | 0.155 |
| VAE + Agglomerative | 0.177 | 72.716 | 1.928 | 0.000 | 0.022 | 0.152 |
| VAE + DBSCAN | — | — | — | 0.000 | 0.000 | 0.100 |
| CVAE + KMeans | 0.194 | 77.350 | 1.762 | 0.002 | 0.023 | 0.161 |
| CVAE + Agglomerative | 0.191 | 74.694 | 1.893 | 0.001 | 0.022 | 0.155 |
| CVAE + DBSCAN | — | — | — | 0.000 | 0.000 | 0.100 |

Note: Rounded for readability; see [results/clustering_metrics.csv](../results/clustering_metrics.csv) for full precision.

### 6.1 Latent Space Visualizations
- PCA + KMeans: [results/latent_visualization/pca_kmeans_tsne.png](../results/latent_visualization/pca_kmeans_tsne.png), [results/latent_visualization/pca_kmeans_umap.png](../results/latent_visualization/pca_kmeans_umap.png)
- AE + KMeans: [results/latent_visualization/ae_kmeans_tsne.png](../results/latent_visualization/ae_kmeans_tsne.png), [results/latent_visualization/ae_kmeans_umap.png](../results/latent_visualization/ae_kmeans_umap.png)
- VAE (β=1.0) + KMeans: [results/latent_visualization/vae_beta1.0_kmeans_tsne.png](../results/latent_visualization/vae_beta1.0_kmeans_tsne.png), [results/latent_visualization/vae_beta1.0_kmeans_umap.png](../results/latent_visualization/vae_beta1.0_kmeans_umap.png)
- CVAE + KMeans: [results/latent_visualization/cvae_kmeans_tsne.png](../results/latent_visualization/cvae_kmeans_tsne.png), [results/latent_visualization/cvae_kmeans_umap.png](../results/latent_visualization/cvae_kmeans_umap.png)

### 6.2 Metric Diagrams (Generated)
- Silhouette: [results/analysis_plots/silhouette.png](../results/analysis_plots/silhouette.png)
- Calinski‑Harabasz: [results/analysis_plots/calinski_harabasz.png](../results/analysis_plots/calinski_harabasz.png)
- Davies‑Bouldin (inverted): [results/analysis_plots/davies_bouldin.png](../results/analysis_plots/davies_bouldin.png)
- ARI: [results/analysis_plots/ari.png](../results/analysis_plots/ari.png)
- NMI: [results/analysis_plots/nmi.png](../results/analysis_plots/nmi.png)
- Purity: [results/analysis_plots/purity.png](../results/analysis_plots/purity.png)

---

## 7. Discussion
- Unsupervised separation: AE-based embeddings yield the best Silhouette and CH scores, indicating compact, well-separated clusters in latent space.
- Genre alignment: ARI/NMI/Purity are low across the board. This is expected in unsupervised settings—clusters follow continuous acoustic/semantic manifolds rather than discrete human genre boundaries.
- CVAE insights: Conditioning provides a slight boost in NMI/Purity, suggesting side information can help align latent structure with labels.
- VAE vs AE: With β=1.0 and simple MFCC+TF‑IDF fusion, AE sometimes outperforms VAE on unsupervised metrics, likely due to the KL regularizer trading off reconstruction capacity. Tuning β, deeper encoders, or spectrogram-based ConvVAE may improve results.
- DBSCAN: Often marks many points as noise in high-dimensional hybrids; better with tuned eps/min_samples or after more aggressive dimensionality reduction.

Limitations:
- Lyrics dataset alignment with audio is approximate (sampled to 1000 rows; not one-to-one with GTZAN tracks). True pairing may improve multimodal performance.
- Metrics reflect unsupervised structure; supervised metrics are diagnostic rather than optimization targets.

---

## 8. Conclusion
We delivered a complete, reproducible hybrid (audio + lyrics) clustering pipeline with VAE-family models. AE + Agglomerative achieved the best unsupervised cluster quality (Silhouette 0.314), while supervised alignment with genre labels remained modest (NMI ≤ 0.051). The codebase includes dataset downloaders, feature extractors, deep models, clustering baselines, metrics, and visualizations. Future work: improved lyrics–audio pairing, ConvVAE on spectrograms, β/architecture tuning, and contrastive or self-supervised pretraining to better align with genre semantics.

---

## 9. References
- D. P. Kingma and M. Welling. Auto-Encoding Variational Bayes. ICLR, 2014.
- I. Higgins et al. beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR, 2017.
- S. Kullback and R. A. Leibler. On Information and Sufficiency. Ann. Math. Statist., 1951.
- P. Vincent et al. Stacked Denoising Autoencoders. JMLR, 2010.
- A. McInnes et al. UMAP: Uniform Manifold Approximation and Projection. arXiv:1802.03426, 2018.
- scikit‑learn: Machine Learning in Python. Pedregosa et al., JMLR 12, 2011.
- Keras / TensorFlow documentation for VAEs and Autoencoders.

---

### Appendix A. Reproducibility
- Environment: Python, TensorFlow/Keras, scikit‑learn, librosa, umap‑learn, matplotlib.
- One‑command sequence (Windows PowerShell):
```bash
C:/Users/USERAS/Desktop/715_Project/.venv/Scripts/python.exe download_gtzan.py
C:/Users/USERAS/Desktop/715_Project/.venv/Scripts/python.exe download_lyrics.py
C:/Users/USERAS/Desktop/715_Project/.venv/Scripts/python.exe run_experiments.py --data-dir ./music_data --lyrics-csv ./music_data/lyrics.csv --use-lyrics --use-audio --epochs 50 --latent-dim 16 --clusters 10
C:/Users/USERAS/Desktop/715_Project/.venv/Scripts/python.exe scripts/plot_metrics.py
```

Overleaf export tip: If you convert this report to LaTeX, place figures under a `figures/` folder and reference them with `\includegraphics{figures/<name>.png}`. Use `\url{}` or `\href{}` to cite workspace-relative paths such as `results/analysis_plots/*.png`.

### Appendix B. File Map
- Datasets: [music_data/gtzan/genres](../music_data/gtzan/genres), [music_data/lyrics.csv](../music_data/lyrics.csv)
- Metrics CSV: [results/clustering_metrics.csv](../results/clustering_metrics.csv)
- Latent plots: [results/latent_visualization](../results/latent_visualization)
- Metric plots: [results/analysis_plots](../results/analysis_plots)
- Models/Code: [src/vae.py](../src/vae.py), [src/dataset.py](../src/dataset.py), [src/clustering.py](../src/clustering.py), [src/evaluation.py](../src/evaluation.py), [src/unsupervised_viz.py](../src/unsupervised_viz.py)
