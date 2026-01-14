# PulseTracks 🎧

PulseTracks is a music analytics and discovery tool—think Spotify, but built for curious listeners and playlist creators. It’s a full-stack web app running in Docker, made with **FastAPI (Python)** and **React (Vite + Tailwind)**.

Search for tracks, explore audio features, find similar songs, generate vibe-based recommendations, and recompute clusters by “mood” in real time.

## Features

### Search & Explore
- Search by **track name**, **artist**, or **album**
- Filter by **genre**
- View track metadata + audio features (danceability, energy, valence, tempo, etc.)

### Similar Songs
- Find similar tracks instantly using **PCA embeddings + cosine similarity**
- Optional **genre filter** for tighter results

### Recommendations (Unsupervised ML)
- Vibe-based recommendations using:
  - seed track
  - user-selected target features
  - weighted similarity in scaled feature space

### Clustering Lab (User-Controlled)
Recompute clusters in real time with:
- **KMeans**
- **Agglomerative**
- **DBSCAN**
- Choose feature subsets + feature weights
- Updated cluster labels propagate through the whole app automatically

### Analytics & Visualizations
- Interactive charts in the UI (genre distribution, cluster sizes, etc.)
- Seaborn-generated plots served by the backend as images:
  - cluster scatter plot (PCA)
  - popularity by cluster
  - feature heatmap per cluster

### Fully Dockerized
- One command to run everything
- Clean separation between frontend and backend services
- Cached artifacts for faster subsequent startups

## Tech Stack

### Backend
- FastAPI + Uvicorn
- Pandas / NumPy
- Scikit-learn (StandardScaler, PCA, clustering, nearest neighbors)
- Seaborn + Matplotlib
- Joblib artifact caching

### Frontend
- React + Vite
- TailwindCSS (green/black palette + gradients)
- Recharts
- React Query
- Axios

### DevOps
- Docker + Docker Compose
- Nginx (production static hosting for frontend)

## Quick Start

```bash
docker compose up --build
```
### Dataset
- name it tracks.csv and add it to backend/app/data/
- Expected columns:
- `track_id`, `artists`, `album_name`, `track_name`, `track_genre`
- `popularity`, `duration_ms`, `explicit`
- `danceability`, `energy`, `key`, `loudness`, `mode`
- `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`
- `tempo`, `time_signature`
<img width="1550" height="900" alt="demop" src="https://github.com/user-attachments/assets/d4944e21-8940-4dbd-8707-32da4436f8ab" /><img width="1628" height="891" alt="demop2" src="https://github.com/user-attachments/assets/7ce7efb8-3480-459e-bd22-f11f8329a5ad" />

