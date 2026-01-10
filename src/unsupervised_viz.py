"""
Visualization utilities for unsupervised learning: t-SNE, UMAP, distributions, reconstructions.
"""
from __future__ import annotations
from typing import Optional, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

try:
    import umap
except Exception:
    umap = None


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_tsne(Z: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "t-SNE", save_path: Optional[str] = None):
    tsne = TSNE(n_components=2, random_state=42, init='random', perplexity=30)
    Z2 = tsne.fit_transform(Z)
    plt.figure(figsize=(8,6))
    if labels is not None:
        scatter = plt.scatter(Z2[:,0], Z2[:,1], c=labels, cmap='tab10', s=12, alpha=0.8)
        plt.legend(*scatter.legend_elements(num=np.unique(labels).size), title="Cluster", loc='best', fontsize=8)
    else:
        plt.scatter(Z2[:,0], Z2[:,1], s=12, alpha=0.8)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_umap(Z: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "UMAP", save_path: Optional[str] = None):
    if umap is None:
        return
    reducer = umap.UMAP(random_state=42)
    Z2 = reducer.fit_transform(Z)
    plt.figure(figsize=(8,6))
    if labels is not None:
        scatter = plt.scatter(Z2[:,0], Z2[:,1], c=labels, cmap='tab10', s=12, alpha=0.8)
        plt.legend(*scatter.legend_elements(num=np.unique(labels).size), title="Cluster", loc='best', fontsize=8)
    else:
        plt.scatter(Z2[:,0], Z2[:,1], s=12, alpha=0.8)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cluster_distribution(labels: np.ndarray, title: str = "Cluster Distribution", save_path: Optional[str] = None):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8,4))
    sns.barplot(x=unique, y=counts)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_reconstructions(x: np.ndarray, x_rec: np.ndarray, n: int = 8, title: str = "Reconstruction", save_path: Optional[str] = None):
    n = min(n, x.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0, i].plot(x[i])
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        axes[1, i].plot(x_rec[i])
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
    axes[0,0].set_ylabel("Input")
    axes[1,0].set_ylabel("Recon")
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
