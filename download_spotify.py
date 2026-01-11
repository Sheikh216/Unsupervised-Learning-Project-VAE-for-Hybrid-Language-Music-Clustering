import os
import pandas as pd
import kagglehub

def download_spotify_dataset():
    """Download Spotify Tracks Dataset from Kaggle."""
    
    print("Downloading Spotify Tracks Dataset...")
    
    # Download dataset
    dataset = "maharshipandya/-spotify-tracks-dataset"
    path = kagglehub.dataset_download(dataset)
    print(f"Downloaded to: {path}")
    
    # Find CSV file
    import glob
    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
    
    if not csv_files:
        print("❌ No CSV file found in downloaded dataset")
        return False
    
    # Use the first CSV found
    csv_path = csv_files[0]
    print(f"Found CSV: {csv_path}")
    
    # Load and prepare data
    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Select important columns
    # Spotify dataset has these columns: track_id,artists,album_name,track_name,popularity,
    # duration_ms,explicit,danceability,energy,key,loudness,mode,speechiness,acousticness,
    # instrumentalness,liveness,valence,tempo,time_signature,genre
    
    required_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 
                    'valence', 'tempo', 'genre']
    
    # Check which columns exist
    available_cols = [col for col in required_cols if col in df.columns]
    print(f"Available audio features: {available_cols}")
    
    # Create processed dataset
    processed_df = df[available_cols].copy()
    
    # Handle missing genre - create synthetic if needed
    if 'genre' not in processed_df.columns:
        print("Creating synthetic genres based on audio features...")
        from sklearn.cluster import KMeans
        features = processed_df.drop(columns=['genre'] if 'genre' in processed_df.columns else [])
        kmeans = KMeans(n_clusters=10, random_state=42)
        processed_df['genre'] = kmeans.fit_predict(features)
    
    # Save processed version
    output_dir = "music_data/spotify"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "spotify_features.csv")
    processed_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Spotify dataset processed and saved to: {output_path}")
    print(f"Processed shape: {processed_df.shape}")
    print(f"Genres: {processed_df['genre'].nunique()} unique genres")
    
    # Also save full dataset for reference
    full_output = os.path.join(output_dir, "spotify_full.csv")
    df.to_csv(full_output, index=False)
    
    print(f"Full dataset saved to: {full_output}")
    
    return True


if __name__ == "__main__":
    download_spotify_dataset()