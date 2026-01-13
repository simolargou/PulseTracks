import io
import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from matplotlib import pyplot as plt
from pydantic import BaseModel
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.getenv("TRACKS_CSV", "backend/data/tracks.csv")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "backend/data/artifacts")
MAX_ROWS = int(os.getenv("MAX_ROWS", "0"))
DEFAULT_FEATURES = [
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

os.makedirs(ARTIFACT_DIR, exist_ok=True)

app = FastAPI(title="Spotify-like Recs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    genre: Optional[str] = None
    limit: int = 10
    seed_track_id: Optional[str] = None
    target_features: Dict[str, float] = {}
    weights: Dict[str, float] = {}


class ClusterRequest(BaseModel):
    algorithm: str
    params: Dict[str, Any] = {}
    features: List[str] = DEFAULT_FEATURES
    weights: Dict[str, float] = {}


@lru_cache(maxsize=1)
def load_tracks() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail=f"Missing dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["explicit"] = df["explicit"].fillna(0).astype(int)
    numeric_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))
    df["artists"] = df["artists"].fillna("")
    df["track_name"] = df["track_name"].fillna("")
    df["album_name"] = df["album_name"].fillna("")
    df["track_genre"] = df["track_genre"].fillna("unknown").astype(str)
    df["track_genre_norm"] = df["track_genre"].str.strip().str.lower()
    return df


def _normalize_genre(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def _artifact_path(name: str) -> str:
    return os.path.join(ARTIFACT_DIR, name)


def _select_subset(df: pd.DataFrame) -> pd.DataFrame:
    if MAX_ROWS > 0 and len(df) > MAX_ROWS:
        return df.sample(n=MAX_ROWS, random_state=42)
    return df


def _load_artifacts() -> Dict[str, Any]:
    artifacts_path = _artifact_path("artifacts.joblib")
    if os.path.exists(artifacts_path):
        try:
            return joblib.load(artifacts_path)
        except Exception:
            os.remove(artifacts_path)
    df = load_tracks()
    df_subset = _select_subset(df)
    numeric_cols = [c for c in DEFAULT_FEATURES if c in df_subset.columns]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_subset[numeric_cols]).astype(np.float32)
    pca2 = PCA(n_components=2, random_state=42, svd_solver="randomized")
    embed2 = pca2.fit_transform(scaled).astype(np.float32)
    pca20 = PCA(n_components=min(20, scaled.shape[1]), random_state=42, svd_solver="randomized")
    embed20 = pca20.fit_transform(scaled).astype(np.float32)
    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(embed20)
    kmeans = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=2048, n_init=5)
    labels = kmeans.fit_predict(scaled)
    index_map = {int(idx): pos for pos, idx in enumerate(df_subset.index)}
    artifacts = {
        "features": numeric_cols,
        "indices": df_subset.index.tolist(),
        "index_map": index_map,
        "scaler": scaler,
        "scaled": scaled,
        "pca2": pca2,
        "embed2": embed2,
        "pca20": pca20,
        "embed20": embed20,
        "nn": nn,
        "cluster_model": kmeans,
        "cluster_labels": labels,
        "cluster_algo": "kmeans",
        "cluster_params": {"k": 8},
        "weights": {c: 1.0 for c in numeric_cols},
    }
    joblib.dump(artifacts, artifacts_path)
    return artifacts


def _save_artifacts(artifacts: Dict[str, Any]) -> None:
    joblib.dump(artifacts, _artifact_path("artifacts.joblib"))


def _apply_weights(data: np.ndarray, features: List[str], weights: Dict[str, float]) -> np.ndarray:
    applied = data.copy()
    for idx, feat in enumerate(features):
        weight = float(weights.get(feat, 1.0))
        applied[:, idx] *= weight
    return applied


def _row_to_scaled(df: pd.DataFrame, idx: int, artifacts: Dict[str, Any]) -> np.ndarray:
    features = artifacts["features"]
    row = df.loc[idx, features].to_frame().T
    scaled = artifacts["scaler"].transform(row).astype(np.float32)
    return _apply_weights(scaled, features, artifacts.get("weights", {}))[0]


def _track_row(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    row = df.loc[idx]
    return {
        "track_id": row.get("track_id"),
        "track_name": row.get("track_name"),
        "album_name": row.get("album_name"),
        "artists": row.get("artists"),
        "popularity": float(row.get("popularity")),
        "track_genre": row.get("track_genre"),
    }


def _explain_similarity(features: List[str], a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    deltas = {}
    for idx, feat in enumerate(features):
        deltas[feat] = float(b[idx] - a[idx])
    return deltas


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/genres")
def genres() -> Dict[str, Any]:
    df = load_tracks()
    counts = df["track_genre_norm"].value_counts().to_dict()
    return {"genres": sorted(counts.keys()), "counts": counts}


@app.get("/tracks/search")
def tracks_search(
    query: str = Query(""),
    genre: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    df = load_tracks()
    subset = df
    genre_norm = _normalize_genre(genre)
    if genre_norm:
        subset = subset[subset["track_genre_norm"] == genre_norm]
    if query:
        q = query.lower()
        subset = subset[
            subset["track_name"].str.lower().str.contains(q)
            | subset["artists"].str.lower().str.contains(q)
            | subset["album_name"].str.lower().str.contains(q)
        ]
    results = [_track_row(subset, idx) for idx in subset.head(limit).index]
    return {"items": results, "total": int(len(subset))}


@app.get("/tracks/{track_id}")
def track_detail(track_id: str) -> Dict[str, Any]:
    df = load_tracks()
    match = df[df["track_id"] == track_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Track not found")
    row = match.iloc[0].to_dict()
    return {"track": row}


@app.get("/tracks/{track_id}/similar")
def similar_tracks(track_id: str, limit: int = Query(10, ge=1, le=50), genre: Optional[str] = Query(None)) -> Dict[str, Any]:
    df = load_tracks()
    artifacts = _load_artifacts()
    if track_id not in df["track_id"].values:
        raise HTTPException(status_code=404, detail="Track not found")
    nn = artifacts["nn"]
    subset_indices = artifacts["indices"]
    idx = int(df.index[df["track_id"] == track_id][0])
    query_scaled = _row_to_scaled(df, idx, artifacts)
    query_embed = artifacts["pca20"].transform([query_scaled])[0]
    distances, indices = nn.kneighbors([query_embed], n_neighbors=min(limit + 1, len(subset_indices)))
    items = []
    genre_norm = _normalize_genre(genre)
    for dist, i in zip(distances[0], indices[0]):
        global_idx = subset_indices[int(i)]
        if global_idx == idx:
            continue
        if genre_norm and df.loc[global_idx]["track_genre_norm"] != genre_norm:
            continue
        items.append({
            "track": _track_row(df, global_idx),
            "distance": float(dist),
            "cluster": int(artifacts["cluster_labels"][i]),
        })
        if len(items) >= limit:
            break
    return {"items": items}


@app.post("/recommend")
def recommend(payload: RecommendRequest = Body(...)) -> Dict[str, Any]:
    df = load_tracks()
    artifacts = _load_artifacts()
    features = artifacts["features"]
    scaled = artifacts["scaled"]
    weights = {**artifacts.get("weights", {}), **payload.weights}
    weighted = _apply_weights(scaled, features, weights)

    subset_idx = artifacts["indices"]
    genre_norm = _normalize_genre(payload.genre)
    if genre_norm:
        subset_idx = [i for i in subset_idx if df.loc[i, "track_genre_norm"] == genre_norm]
    if not subset_idx:
        raise HTTPException(status_code=404, detail="No tracks for genre")

    target_vec = None
    if payload.seed_track_id:
        match_idx = df.index[df["track_id"] == payload.seed_track_id]
        if len(match_idx) == 0:
            raise HTTPException(status_code=404, detail="Seed track not found")
        target_vec = _row_to_scaled(df, int(match_idx[0]), {**artifacts, "weights": weights})
    else:
        base = df[features].median(numeric_only=True).to_dict()
        for key, value in payload.target_features.items():
            if key in base:
                base[key] = value
        raw_vec = np.array([base[f] for f in features], dtype=float).reshape(1, -1)
        target_vec = _apply_weights(artifacts["scaler"].transform(raw_vec), features, weights)[0]

    index_map = artifacts["index_map"]
    subset_positions = [index_map[int(i)] for i in subset_idx]
    subset_vectors = weighted[subset_positions]
    distances = pairwise_distances([target_vec], subset_vectors, metric="cosine")[0]
    order = np.argsort(distances)[: payload.limit]

    items = []
    for rank in order:
        i = subset_idx[rank]
        pos = subset_positions[rank]
        items.append({
            "track": _track_row(df, i),
            "distance": float(distances[rank]),
            "cluster": int(artifacts["cluster_labels"][pos]),
            "feature_deltas": _explain_similarity(features, target_vec, weighted[pos]),
        })

    return {"items": items, "target_features": {f: float(target_vec[idx]) for idx, f in enumerate(features)}}


@app.post("/cluster/recompute")
def recompute_cluster(payload: ClusterRequest = Body(...)) -> Dict[str, Any]:
    df = load_tracks()
    features = [f for f in payload.features if f in df.columns]
    if not features:
        raise HTTPException(status_code=400, detail="No valid features provided")
    df_subset = _select_subset(df)
    data = df_subset[features].fillna(df_subset[features].median(numeric_only=True))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data).astype(np.float32)
    weights = payload.weights
    weighted = _apply_weights(scaled, features, weights)

    algo = payload.algorithm.lower()
    if algo == "kmeans":
        k = int(payload.params.get("k", 8))
        model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=5)
        labels = model.fit_predict(weighted)
    elif algo == "agglomerative":
        k = int(payload.params.get("k", 8))
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(weighted)
    elif algo == "dbscan":
        eps = float(payload.params.get("eps", 0.5))
        min_samples = int(payload.params.get("min_samples", 5))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(weighted)
    else:
        raise HTTPException(status_code=400, detail="Unsupported algorithm")

    pca2 = PCA(n_components=2, random_state=42, svd_solver="randomized")
    embed2 = pca2.fit_transform(weighted).astype(np.float32)
    pca20 = PCA(n_components=min(20, weighted.shape[1]), random_state=42, svd_solver="randomized")
    embed20 = pca20.fit_transform(weighted).astype(np.float32)
    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(embed20)

    index_map = {int(idx): pos for pos, idx in enumerate(df_subset.index)}
    artifacts = {
        "features": features,
        "indices": df_subset.index.tolist(),
        "index_map": index_map,
        "scaler": scaler,
        "scaled": weighted,
        "pca2": pca2,
        "embed2": embed2,
        "pca20": pca20,
        "embed20": embed20,
        "nn": nn,
        "cluster_model": model,
        "cluster_labels": labels,
        "cluster_algo": algo,
        "cluster_params": payload.params,
        "weights": weights,
    }
    _save_artifacts(artifacts)
    return {"status": "ok", "clusters": int(len(set(labels)))}


@app.get("/cluster/summary")
def cluster_summary(genre: Optional[str] = Query(None)) -> Dict[str, Any]:
    df = load_tracks()
    artifacts = _load_artifacts()
    labels = artifacts["cluster_labels"]
    embed = artifacts["embed20"]
    base_df = df.loc[artifacts["indices"]]
    subset = base_df
    subset_idx = base_df.index
    genre_norm = _normalize_genre(genre)
    if genre_norm:
        subset = base_df[base_df["track_genre_norm"] == genre_norm]
        subset_idx = subset.index

    if subset.empty:
        raise HTTPException(status_code=404, detail="No tracks for genre")

    popular = subset.sort_values("popularity")
    best = _track_row(subset, popular.index[-1])
    worst = _track_row(subset, popular.index[0])

    subset_positions = [artifacts["index_map"][int(i)] for i in subset_idx]
    sub_embed = embed[subset_positions]
    nn_local = NearestNeighbors(metric="cosine", algorithm="auto")
    nn_local.fit(sub_embed)
    distances, indices = nn_local.kneighbors(sub_embed, n_neighbors=2)
    nearest_row = int(np.argmin(distances[:, 1]))
    nearest_pair = (nearest_row, int(indices[nearest_row, 1]))

    sample_size = min(2000, len(subset_idx))
    sample_idx = np.random.choice(len(subset_idx), size=sample_size, replace=False)
    sample_embed = sub_embed[sample_idx]
    sample_dist = pairwise_distances(sample_embed, metric="cosine")
    farthest_pair = np.unravel_index(np.argmax(sample_dist), sample_dist.shape)
    farthest_pair = (int(sample_idx[farthest_pair[0]]), int(sample_idx[farthest_pair[1]]))

    def pair_info(pair):
        a_idx = subset_idx[pair[0]]
        b_idx = subset_idx[pair[1]]
        distance = float(pairwise_distances([sub_embed[pair[0]]], [sub_embed[pair[1]]], metric="cosine")[0][0])
        return {
            "a": _track_row(df, a_idx),
            "b": _track_row(df, b_idx),
            "distance": distance,
        }

    summaries = []
    for cluster_id in sorted(set(labels)):
        cluster_mask = labels == cluster_id
        cluster_indices = base_df.index[cluster_mask]
        if genre:
            cluster_indices = [i for i in cluster_indices if i in subset_idx]
        if not cluster_indices:
            continue
        cluster_positions = [artifacts["index_map"][int(i)] for i in cluster_indices]
        cluster_vectors = artifacts["scaled"][cluster_positions]
        centroid = cluster_vectors.mean(axis=0)
        cluster_df = df.loc[cluster_indices]
        top = _track_row(cluster_df, cluster_df["popularity"].idxmax())
        bottom = _track_row(cluster_df, cluster_df["popularity"].idxmin())
        summaries.append({
            "cluster": int(cluster_id),
            "centroid": {f: float(centroid[idx]) for idx, f in enumerate(artifacts["features"])},
            "top_track": top,
            "bottom_track": bottom,
            "size": int(len(cluster_indices)),
        })

    return {
        "best_track": best,
        "worst_track": worst,
        "nearest_pair": pair_info(nearest_pair),
        "farthest_pair": pair_info(farthest_pair),
        "clusters": summaries,
    }


@app.get("/analytics/overview")
def analytics_overview(genre: Optional[str] = Query(None)) -> Dict[str, Any]:
    df = load_tracks()
    artifacts = _load_artifacts()
    genre_norm = _normalize_genre(genre)
    base_df = df.loc[artifacts["indices"]]
    subset = base_df if not genre_norm else base_df[base_df["track_genre_norm"] == genre_norm]
    if subset.empty:
        raise HTTPException(status_code=404, detail="No tracks for genre")
    genre_counts = subset["track_genre"].value_counts().to_dict()
    popularity_hist = np.histogram(subset["popularity"], bins=10, range=(0, 100))
    labels = artifacts["cluster_labels"]
    cluster_sizes = pd.Series(labels).value_counts().to_dict()
    means = subset[artifacts["features"]].mean().to_dict()
    return {
        "genre_distribution": genre_counts,
        "popularity_histogram": {
            "bins": popularity_hist[1].tolist(),
            "counts": popularity_hist[0].tolist(),
        },
        "cluster_sizes": cluster_sizes,
        "feature_means": {k: float(v) for k, v in means.items()},
    }


def _plot_response(fig) -> Response:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/plots/cluster_scatter.png")
def plot_cluster_scatter(genre: Optional[str] = Query(None)) -> Response:
    df = load_tracks()
    artifacts = _load_artifacts()
    embed = artifacts["embed2"]
    labels = artifacts["cluster_labels"]
    subset_idx = artifacts["indices"]
    if genre:
        genre_norm = _normalize_genre(genre)
        subset_idx = [i for i in subset_idx if df.loc[i, "track_genre_norm"] == genre_norm]
    plot_df = pd.DataFrame({
        "pca1": embed[[artifacts["index_map"][int(i)] for i in subset_idx], 0],
        "pca2": embed[[artifacts["index_map"][int(i)] for i in subset_idx], 1],
        "cluster": labels[[artifacts["index_map"][int(i)] for i in subset_idx]],
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=plot_df, x="pca1", y="pca2", hue="cluster", palette="tab10", ax=ax, legend=False)
    ax.set_title("Cluster scatter")
    return _plot_response(fig)


@app.get("/plots/cluster_popularity.png")
def plot_cluster_popularity(genre: Optional[str] = Query(None)) -> Response:
    df = load_tracks()
    artifacts = _load_artifacts()
    labels = artifacts["cluster_labels"]
    subset = df.loc[artifacts["indices"]]
    if genre:
        genre_norm = _normalize_genre(genre)
        subset = subset[subset["track_genre_norm"] == genre_norm]
    if subset.empty:
        raise HTTPException(status_code=404, detail="No tracks for genre")
    plot_df = subset.copy()
    plot_df["cluster"] = [labels[artifacts["index_map"][int(i)]] for i in subset.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=plot_df, x="cluster", y="popularity", ax=ax)
    ax.set_title("Popularity by cluster")
    return _plot_response(fig)


@app.get("/plots/cluster_heatmap.png")
def plot_cluster_heatmap(genre: Optional[str] = Query(None)) -> Response:
    df = load_tracks()
    artifacts = _load_artifacts()
    labels = artifacts["cluster_labels"]
    subset = df.loc[artifacts["indices"]]
    if genre:
        genre_norm = _normalize_genre(genre)
        subset = subset[subset["track_genre_norm"] == genre_norm]
    if subset.empty:
        raise HTTPException(status_code=404, detail="No tracks for genre")
    plot_df = subset.copy()
    plot_df["cluster"] = [labels[artifacts["index_map"][int(i)]] for i in subset.index]
    means = plot_df.groupby("cluster")[artifacts["features"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(means, cmap="YlGnBu", ax=ax)
    ax.set_title("Feature means per cluster")
    return _plot_response(fig)
