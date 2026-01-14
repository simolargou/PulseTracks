import axios from 'axios'

const resolveBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL
  }
  if (typeof window !== 'undefined') {
    return `http://${window.location.hostname}:8000`
  }
  return 'http://localhost:8000'
}

const api = axios.create({
  baseURL: resolveBaseUrl()
})

export const fetchGenres = async () => (await api.get('/genres')).data
export const searchTracks = async ({ query, genre, limit }) =>
  (await api.get('/tracks/search', { params: { query, genre, limit } })).data
export const fetchTrack = async (trackId) => (await api.get(`/tracks/${trackId}`)).data
export const fetchSimilar = async ({ trackId, limit, genre }) =>
  (await api.get(`/tracks/${trackId}/similar`, { params: { limit, genre } })).data
export const recommendTracks = async (payload) => (await api.post('/recommend', payload)).data
export const recomputeCluster = async (payload) => (await api.post('/cluster/recompute', payload)).data
export const fetchClusterSummary = async (genre) =>
  (await api.get('/cluster/summary', { params: { genre } })).data
export const fetchAnalytics = async (genre) =>
  (await api.get('/analytics/overview', { params: { genre } })).data

export const plotUrl = (name, genre) => {
  const base = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
  const url = new URL(`${base}/plots/${name}.png`)
  if (genre) {
    url.searchParams.set('genre', genre)
  }
  return url.toString()
}

export default api
