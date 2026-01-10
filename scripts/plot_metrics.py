import os
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = os.path.join('results', 'clustering_metrics.csv')
OUT_DIR = os.path.join('results', 'analysis_plots')

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure method exists and coerce numerics
    for col in ['silhouette','calinski_harabasz','davies_bouldin','ari','nmi','purity']:
        df[col] = df[col].apply(safe_float)
    return df


def barplot(df: pd.DataFrame, metric: str, title: str, outfile: str, invert: bool=False):
    d = df[['method', metric]].dropna()
    if d.empty:
        return
    # For DB, lower is better -> invert to make bars intuitive
    vals = d[metric].values
    if invert:
        vals = max(vals) - vals
        title = f"{title} (inverted for visualization)"
    order = d.assign(val=vals).sort_values('val', ascending=False)['method']
    d = d.set_index('method').loc[order]

    plt.figure(figsize=(10,5))
    bars = plt.bar(range(len(d)), d[metric].values, color='#4C78A8')
    plt.xticks(range(len(d)), d.index, rotation=45, ha='right')
    plt.title(title)
    plt.ylabel(metric.replace('_',' ').title())
    plt.tight_layout()
    for i, b in enumerate(bars):
        v = d[metric].values[i]
        plt.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    plt.savefig(outfile, dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_metrics(IN_CSV)
    barplot(df, 'silhouette', 'Silhouette Score by Method', os.path.join(OUT_DIR, 'silhouette.png'))
    barplot(df, 'calinski_harabasz', 'Calinski-Harabasz by Method', os.path.join(OUT_DIR, 'calinski_harabasz.png'))
    barplot(df, 'davies_bouldin', 'Davies-Bouldin by Method', os.path.join(OUT_DIR, 'davies_bouldin.png'), invert=True)
    barplot(df, 'ari', 'Adjusted Rand Index (ARI) by Method', os.path.join(OUT_DIR, 'ari.png'))
    barplot(df, 'nmi', 'Normalized Mutual Information (NMI) by Method', os.path.join(OUT_DIR, 'nmi.png'))
    barplot(df, 'purity', 'Cluster Purity by Method', os.path.join(OUT_DIR, 'purity.png'))
    print(f"Saved plots to {OUT_DIR}")

if __name__ == '__main__':
    main()
