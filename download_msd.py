"""
Download Million Song Dataset (MSD) features from Kaggle.
"""
import os
import pandas as pd
import kagglehub
import shutil
import glob

def download_msd_dataset():
    """Download Million Song Dataset features from Kaggle."""
    
    print("Downloading Million Song Dataset (MSD)...")
    
    # Try different MSD dataset versions
    msd_datasets = [
        "rodolfofigueroa/gtzan-dataset-music-genre-classification",  # MSD features
        "notshrirang/spotify-million-song-dataset",
        "tomigelo/spotify-audio-features",
        "eliasdabbas/audio-features-of-spotify-songs"
    ]
    
    path = None
    for dataset in msd_datasets:
        try:
            print(f"Trying: {dataset}")
            path = kagglehub.dataset_download(dataset)
            print(f"✅ Downloaded: {dataset}")
            break
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    if path is None:
        print("❌ Could not download any MSD dataset")
        return False
    
    print(f"Downloaded to: {path}")
    
    # Find CSV files
    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
    
    if not csv_files:
        print("❌ No CSV files found")
        return False
    
    # Use the largest CSV (likely main dataset)
    csv_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    csv_path = csv_files[0]
    
    print(f"Using main CSV: {csv_path}")
    print(f"File size: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare output directory
    output_dir = "music_data/msd"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full dataset
    full_output = os.path.join(output_dir, "msd_full.csv")
    df.to_csv(full_output, index=False)
    print(f"✅ Full dataset saved to: {full_output}")
    
    # Check for audio features columns
    audio_feature_keywords = ['danceability', 'energy', 'loudness', 'tempo', 
                              'acousticness', 'instrumentalness', 'valence',
                              'speechiness', 'liveness', 'key', 'mode']
    
    audio_cols = [col for col in df.columns if any(kw in col.lower() for kw in audio_feature_keywords)]
    print(f"\n🎵 Audio feature columns found: {audio_cols}")
    
    # Check for genre/tag columns
    genre_keywords = ['genre', 'tag', 'category', 'style', 'type']
    genre_cols = [col for col in df.columns if any(kw in col.lower() for kw in genre_keywords)]
    
    if genre_cols:
        print(f"🎯 Genre columns found: {genre_cols}")
        print(f"Genre distribution:")
        for col in genre_cols[:2]:  # Show first 2 genre columns
            print(f"\n{col}:")
            print(df[col].value_counts().head(10))
    else:
        print("⚠️ No genre columns found")
    
    # Create processed version with key features
    processed_cols = audio_cols[:]
    if genre_cols:
        processed_cols.append(genre_cols[0])  # Add first genre column
    
    if len(processed_cols) > 0:
        processed_df = df[processed_cols].copy()
        processed_output = os.path.join(output_dir, "msd_features.csv")
        processed_df.to_csv(processed_output, index=False)
        print(f"\n✅ Processed features saved to: {processed_output}")
        print(f"   Features: {len(processed_cols)} columns")
        print(f"   Samples: {len(processed_df)}")
    
    # Also copy any additional files
    for file in csv_files[:5]:  # Copy first 5 CSV files
        if file != csv_path:
            shutil.copy2(file, os.path.join(output_dir, os.path.basename(file)))
    
    print(f"\n📊 MSD dataset ready in: {output_dir}")
    return True


if __name__ == "__main__":
    download_msd_dataset()