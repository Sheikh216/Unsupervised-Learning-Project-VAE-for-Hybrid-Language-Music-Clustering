"""
Hybrid dataset utilities for audio (MFCC/spectrogram) and lyrics (TF-IDF) features.
Generates realistic sample data if real datasets are not present.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from audio_data_loader import AudioDataLoader

try:
    import librosa
except Exception:
    librosa = None


def build_lyrics_features(texts: list[str], max_features: int = 500) -> Tuple[np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(max_features=max_features, analyzer="word")
    X = vectorizer.fit_transform(texts).toarray()
    return X.astype(np.float32), vectorizer


def build_audio_mfcc_features(n_samples: int = 1000, data_dir: str = "./music_data", allow_fallback: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    loader = AudioDataLoader('gtzan', data_dir=data_dir, allow_fallback=allow_fallback)
    (X_train, y_train), (X_test, y_test) = loader.load_gtzan()
    X = np.vstack([X_train, X_test])
    y = np.vstack([y_train, y_test])
    return X, y


def build_audio_spectrogram_dataset(file_paths: list[str], sr: int = 22050, n_mels: int = 128, frames: int = 128) -> np.ndarray:
    if librosa is None:
        raise RuntimeError("librosa is required for spectrogram extraction")
    X = []
    for p in file_paths:
        try:
            y, _ = librosa.load(p, sr=sr, duration=30.0)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)
            # Pad/trim to fixed number of frames
            if S_db.shape[1] < frames:
                pad = frames - S_db.shape[1]
                S_db = np.pad(S_db, ((0,0),(0,pad)), mode='constant')
            else:
                S_db = S_db[:, :frames]
            X.append(S_db[..., None])
        except Exception:
            continue
    return np.array(X, dtype=np.float32)


def load_hybrid_dataset(
    use_audio: bool = True,
    use_lyrics: bool = True,
    data_dir: str = "./music_data",
    lyrics_csv: Optional[str] = None,
    lyrics_text_col: str = "lyrics",
    language_col: str = "language",
    audio_feature_dim: int = 43,
    lyrics_max_features: int = 500,
    allow_fallback: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Returns a dict with possible keys: X_audio, X_lyrics, X_combined, y_language, y_genre
    - If no real data present, generates sample audio features and sample lyrics.
    """
    out: Dict[str, np.ndarray] = {}

    # Audio features (MFCC summary from AudioDataLoader)
    if use_audio:
        X_audio, y_genre_onehot = build_audio_mfcc_features(data_dir=data_dir, allow_fallback=allow_fallback)
        out["X_audio"] = X_audio.astype(np.float32)
        # genres if present
        if y_genre_onehot is not None and y_genre_onehot.ndim == 2:
            out["y_genre"] = np.argmax(y_genre_onehot, axis=1)

    # Lyrics features
    if use_lyrics:
        if lyrics_csv and os.path.exists(lyrics_csv):
            df = pd.read_csv(lyrics_csv)
            texts = df[lyrics_text_col].fillna("").astype(str).tolist()
            X_lyrics, vec = build_lyrics_features(texts, max_features=lyrics_max_features)
            out["X_lyrics"] = X_lyrics
            if language_col in df.columns:
                langs = df[language_col].astype(str).str.lower()
                # map common labels
                mapping = {"english": 0, "bangla": 1, "bengali": 1, "en":0, "bn":1}
                y_lang = np.array([mapping.get(v, 0) for v in langs])
                out["y_language"] = y_lang
        else:
            raise FileNotFoundError("Lyrics CSV is required when use_lyrics=True and no fallback is allowed.")

    # Combine modalities (concatenate after scaling)
    if use_audio and use_lyrics:
        scaler_a = StandardScaler()
        scaler_l = StandardScaler()
        Xa = scaler_a.fit_transform(out["X_audio"]) if "X_audio" in out else None
        Xl = scaler_l.fit_transform(out["X_lyrics"]) if "X_lyrics" in out else None
        Xc = np.concatenate([Xa, Xl], axis=1).astype(np.float32)
        out["X_combined"] = Xc
    elif use_audio:
        out["X_combined"] = out["X_audio"].astype(np.float32)
    elif use_lyrics:
        out["X_combined"] = out["X_lyrics"].astype(np.float32)

    return out
