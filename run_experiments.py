"""
Experiment runner for VAE-based hybrid language music clustering.
Runs: VAE (beta variants), CVAE (if labels), ConvVAE (optional), PCA+KMeans baseline, AE+KMeans baseline.
Outputs metrics CSV and latent visualizations.
"""
from __future__ import annotations
import os
import json
import argparse
from glob import glob
import numpy as np
import pandas as pd
import tensorflow as tf


from src.dataset import load_bangla_lyrics_dataset


from src.dataset import load_hybrid_dataset, build_audio_spectrogram_dataset
from src.vae import VAE, CVAE, Autoencoder, ConvVAE
from src.clustering import run_kmeans, run_agglomerative, run_dbscan, pca_kmeans, evaluate_clustering
from src.unsupervised_viz import plot_tsne, plot_umap, plot_cluster_distribution


RESULTS_DIR = "./results"
LV_DIR = os.path.join(RESULTS_DIR, "latent_visualization")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LV_DIR, exist_ok=True)


def train_vae(X, latent_dim=16, beta=1.0, epochs=30, batch_size=128, val_split=0.1, seed=42):
    tf.keras.utils.set_random_seed(seed)
    vae = VAE(input_dim=X.shape[1], latent_dim=latent_dim, hidden_dims=(256,128), beta=beta)
    # Provide a dummy loss and train on (X, X) to satisfy Keras compile requirements
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=lambda y_true, y_pred: 0.0)
    vae.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=1)
    Z = vae.encode(X)
    return vae, Z


def train_cvae(X, C, latent_dim=16, beta=1.0, epochs=30, batch_size=128, val_split=0.1, seed=42):
    tf.keras.utils.set_random_seed(seed)
    cvae = CVAE(input_dim=X.shape[1], cond_dim=C.shape[1], latent_dim=latent_dim, hidden_dims=(256,128), beta=beta)
    # Provide a dummy loss and train on targets X to satisfy Keras compile requirements
    cvae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=lambda y_true, y_pred: 0.0)
    cvae.fit((X, C), X, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=1)
    Z = cvae.encode(X, C)
    return cvae, Z


def train_conv_vae(X, latent_dim=16, beta=1.0, epochs=30, batch_size=32, val_split=0.1, seed=42):
    tf.keras.utils.set_random_seed(seed)
    conv_vae = ConvVAE(input_shape=X.shape[1:], latent_dim=latent_dim, beta=beta)
    conv_vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=lambda y_true, y_pred: 0.0)
    conv_vae.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=1)
    Z = conv_vae.encode(X)
    return conv_vae, Z


def train_ae(X, latent_dim=16, epochs=30, batch_size=128, val_split=0.1, seed=42):
    tf.keras.utils.set_random_seed(seed)
    ae = Autoencoder(input_dim=X.shape[1], latent_dim=latent_dim, hidden_dims=(256,128))
    hist = ae.fit_ae(X, batch_size=batch_size, epochs=epochs, validation_data=None)
    Z = ae.encode(X)
    return ae, Z


def cluster_and_evaluate(Z, n_clusters, y_true=None, tag="vae"):
    metrics_rows = []
    # KMeans
    km_labels = run_kmeans(Z, n_clusters)
    km_metrics = evaluate_clustering(Z, km_labels, y_true)
    km_metrics.update({"method": f"{tag}+kmeans"})
    metrics_rows.append(km_metrics)
    # Agglomerative
    agg_labels = run_agglomerative(Z, n_clusters)
    agg_metrics = evaluate_clustering(Z, agg_labels, y_true)
    agg_metrics.update({"method": f"{tag}+agglomerative"})
    metrics_rows.append(agg_metrics)
    # DBSCAN
    db_labels = run_dbscan(Z, eps=0.8, min_samples=5)
    db_metrics = evaluate_clustering(Z, db_labels, y_true)
    db_metrics.update({"method": f"{tag}+dbscan"})
    metrics_rows.append(db_metrics)
    return metrics_rows, {"kmeans": km_labels, "agglomerative": agg_labels, "dbscan": db_labels}


def save_metrics(metrics_rows, out_csv):
    df = pd.DataFrame(metrics_rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run VAE hybrid music clustering experiments")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters")
    parser.add_argument("--use-lyrics", action="store_true", default=False)
    parser.add_argument("--use-audio", action="store_true", default=True)
    parser.add_argument("--data-dir", type=str, default="./music_data")
    parser.add_argument("--lyrics-csv", type=str, default=None)
    parser.add_argument("--allow-fallback", action="store_true", default=False, help="Allow synthetic fallback data (default: False)")
    parser.add_argument("--spectrogram-dir", type=str, default=None, help="Directory of audio files to build spectrograms for ConvVAE")
    parser.add_argument("--spectrogram-ext", type=str, default=".wav,.mp3,.flac", help="Comma-separated audio extensions for spectrogram loading")
    parser.add_argument("--conv-latent-dim", type=int, default=16)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "clustering_metrics.csv"))
    parser.add_argument(
    "--bangla_csv",
    type=str,
    default=None,
    help="Path to Bangla lyrics CSV"
    )

    args = parser.parse_args()

    ensure_dirs()


 

    # --------- INSERT HERE ---------
    if args.bangla_csv is not None:
        print("Running HARD task using Bangla lyrics dataset")

        X, y_lang, label_encoder = load_bangla_lyrics_dataset(
            args.bangla_csv,
            max_features=args.tfidf_features
        )

        y_true = y_lang   # ground truth for evaluation
    # --------------------------------


    # Load hybrid dataset (no synthetic fallback unless explicitly allowed)
    try:
        data = load_hybrid_dataset(
            use_audio=args.use_audio,
            use_lyrics=args.use_lyrics,
            data_dir=args.data_dir,
            lyrics_csv=args.lyrics_csv,
            allow_fallback=args.allow_fallback,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"Dataset missing: {e}")
    X = data["X_combined"]
    y_lang = data.get("y_language", None)
    y_genre = data.get("y_genre", None)
    
    # Use genre labels as ground truth for supervised metrics (ARI, NMI, Purity)
    y_true = y_genre if y_genre is not None else y_lang
    print(f"Using ground truth labels: {'genre' if y_genre is not None else 'language' if y_lang is not None else 'none'}")
    if y_true is not None:
        print(f"Number of unique labels: {len(np.unique(y_true))}")

    metrics_all = []

    # 1) Baseline: PCA + KMeans
    Z_pca, labels_pca = pca_kmeans(X, n_components=args.latent_dim, n_clusters=args.clusters)
    metrics_pca = evaluate_clustering(Z_pca, labels_pca, y_true=y_true)
    metrics_pca.update({"method": "pca+kmeans"})
    metrics_all.append(metrics_pca)

    plot_tsne(Z_pca, labels_pca, title="PCA+KMeans t-SNE", save_path=os.path.join(LV_DIR, "pca_kmeans_tsne.png"))
    plot_umap(Z_pca, labels_pca, title="PCA+KMeans UMAP", save_path=os.path.join(LV_DIR, "pca_kmeans_umap.png"))

    # 2) Autoencoder + KMeans (baseline)
    ae, Z_ae = train_ae(X, latent_dim=args.latent_dim, epochs=args.epochs)
    m_rows, lab = cluster_and_evaluate(Z_ae, n_clusters=args.clusters, y_true=y_true, tag="ae")
    metrics_all.extend(m_rows)
    plot_tsne(Z_ae, lab["kmeans"], title="AE+KMeans t-SNE", save_path=os.path.join(LV_DIR, "ae_kmeans_tsne.png"))
    plot_umap(Z_ae, lab["kmeans"], title="AE+KMeans UMAP", save_path=os.path.join(LV_DIR, "ae_kmeans_umap.png"))

    # 3) VAE (beta)
    vae, Z = train_vae(X, latent_dim=args.latent_dim, beta=args.beta, epochs=args.epochs)
    m_rows, lab = cluster_and_evaluate(Z, n_clusters=args.clusters, y_true=y_true, tag=f"vae_beta{args.beta}")
    metrics_all.extend(m_rows)
    plot_tsne(Z, lab["kmeans"], title=f"VAE(beta={args.beta})+KMeans t-SNE", save_path=os.path.join(LV_DIR, f"vae_beta{args.beta}_kmeans_tsne.png"))
    plot_umap(Z, lab["kmeans"], title=f"VAE(beta={args.beta})+KMeans UMAP", save_path=os.path.join(LV_DIR, f"vae_beta{args.beta}_kmeans_umap.png"))

    # 4) CVAE (if language labels exist)
    if y_lang is not None:
        # One-hot condition
        n_cond = int(np.max(y_lang)) + 1
        C = np.eye(n_cond)[y_lang]
        cvae, Zc = train_cvae(X, C, latent_dim=args.latent_dim, beta=args.beta, epochs=args.epochs)
        m_rows, lab = cluster_and_evaluate(Zc, n_clusters=args.clusters, y_true=y_true, tag="cvae")
        metrics_all.extend(m_rows)
        plot_tsne(Zc, lab["kmeans"], title="CVAE+KMeans t-SNE", save_path=os.path.join(LV_DIR, "cvae_kmeans_tsne.png"))
        plot_umap(Zc, lab["kmeans"], title="CVAE+KMeans UMAP", save_path=os.path.join(LV_DIR, "cvae_kmeans_umap.png"))

    # 5) ConvVAE on spectrograms (optional, requires spectrogram_dir)
    if args.spectrogram_dir:
        exts = [e.strip().lower() for e in args.spectrogram_ext.split(',') if e.strip()]
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(args.spectrogram_dir, f"**/*{ext}"), recursive=True))
        if not files:
            raise SystemExit(f"No audio files found in {args.spectrogram_dir} with extensions {exts}")
        X_spec = build_audio_spectrogram_dataset(files)
        if X_spec.size == 0:
            raise SystemExit("Spectrogram extraction produced no samples.")
        conv_vae, Z_conv = train_conv_vae(X_spec, latent_dim=args.conv_latent_dim, beta=args.beta, epochs=args.epochs, batch_size=16)
        m_rows, lab = cluster_and_evaluate(Z_conv, n_clusters=args.clusters, y_true=None, tag="conv_vae")
        metrics_all.extend(m_rows)
        plot_tsne(Z_conv, lab["kmeans"], title="ConvVAE+KMeans t-SNE", save_path=os.path.join(LV_DIR, "convvae_kmeans_tsne.png"))
        plot_umap(Z_conv, lab["kmeans"], title="ConvVAE+KMeans UMAP", save_path=os.path.join(LV_DIR, "convvae_kmeans_umap.png"))

    # Save metrics
    save_metrics(metrics_all, args.output)

    # Also save a JSON summary
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)
    print("All experiments completed.")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
