import { useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import Card from '../components/Card'
import { ErrorState, Loading } from '../components/Status'
import { fetchClusterSummary, recomputeCluster, plotUrl, fetchGenres } from '../api/client'

const featureOptions = [
  'popularity',
  'duration_ms',
  'explicit',
  'danceability',
  'energy',
  'key',
  'loudness',
  'mode',
  'speechiness',
  'acousticness',
  'instrumentalness',
  'liveness',
  'valence',
  'tempo',
  'time_signature'
]

const ClusteringLab = () => {
  const [algorithm, setAlgorithm] = useState('kmeans')
  const [k, setK] = useState(8)
  const [eps, setEps] = useState(0.5)
  const [minSamples, setMinSamples] = useState(5)
  const [features, setFeatures] = useState(featureOptions)
  const [genre, setGenre] = useState('')

  const { data: genresData } = useQuery({ queryKey: ['genres'], queryFn: fetchGenres })
  const { data: summaryData, isLoading, error, refetch } = useQuery({
    queryKey: ['clusterSummary', genre],
    queryFn: () => fetchClusterSummary(genre || undefined)
  })

  const recomputeMutation = useMutation({
    mutationFn: (payload) => recomputeCluster(payload),
    onSuccess: () => refetch()
  })

  const handleFeatureToggle = (feature) => {
    setFeatures((prev) =>
      prev.includes(feature) ? prev.filter((item) => item !== feature) : [...prev, feature]
    )
  }

  const payload = useMemo(() => {
    const params = algorithm === 'dbscan' ? { eps, min_samples: minSamples } : { k }
    return {
      algorithm,
      params,
      features,
      weights: {}
    }
  }, [algorithm, k, eps, minSamples, features])

  return (
    <div className="space-y-6">
      <Card
        title="Clustering controls"
        actions={
          <button
            onClick={() => recomputeMutation.mutate(payload)}
            className="rounded-full bg-moss px-4 py-2 text-xs uppercase tracking-widest text-ink"
          >
            Recompute
          </button>
        }
      >
        <div className="grid gap-4 md:grid-cols-3">
          <label className="text-xs uppercase text-ash">
            Algorithm
            <select
              value={algorithm}
              onChange={(event) => setAlgorithm(event.target.value)}
              className="mt-2 w-full rounded-lg bg-black/40 border border-emerald/40 px-3 py-2 text-sm"
            >
              <option value="kmeans">KMeans</option>
              <option value="agglomerative">Agglomerative</option>
              <option value="dbscan">DBSCAN</option>
            </select>
          </label>
          {(algorithm === 'kmeans' || algorithm === 'agglomerative') && (
            <label className="text-xs uppercase text-ash">
              k clusters
              <input
                type="number"
                min="2"
                max="30"
                value={k}
                onChange={(event) => setK(parseInt(event.target.value, 10))}
                className="mt-2 w-full rounded-lg bg-black/40 border border-emerald/40 px-3 py-2 text-sm"
              />
            </label>
          )}
          {algorithm === 'dbscan' && (
            <>
              <label className="text-xs uppercase text-ash">
                eps
                <input
                  type="number"
                  step="0.1"
                  value={eps}
                  onChange={(event) => setEps(parseFloat(event.target.value))}
                  className="mt-2 w-full rounded-lg bg-black/40 border border-emerald/40 px-3 py-2 text-sm"
                />
              </label>
              <label className="text-xs uppercase text-ash">
                min samples
                <input
                  type="number"
                  min="2"
                  max="20"
                  value={minSamples}
                  onChange={(event) => setMinSamples(parseInt(event.target.value, 10))}
                  className="mt-2 w-full rounded-lg bg-black/40 border border-emerald/40 px-3 py-2 text-sm"
                />
              </label>
            </>
          )}
          <label className="text-xs uppercase text-ash">
            Genre scope
            <select
              value={genre}
              onChange={(event) => setGenre(event.target.value)}
              className="mt-2 w-full rounded-lg bg-black/40 border border-emerald/40 px-3 py-2 text-sm"
            >
              <option value="">All genres</option>
              {genresData?.genres?.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="grid grid-cols-2 gap-3 text-xs uppercase text-ash">
          {featureOptions.map((feature) => (
            <label key={feature} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={features.includes(feature)}
                onChange={() => handleFeatureToggle(feature)}
              />
              {feature}
            </label>
          ))}
        </div>
        {recomputeMutation.isLoading && <Loading label="Recomputing clusters..." />}
        {recomputeMutation.error && <ErrorState message="Cluster recompute failed." />}
      </Card>

      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <Card title="Cluster summary">
          {isLoading && <Loading />}
          {error && <ErrorState message="Summary unavailable." />}
          {summaryData && (
            <div className="space-y-4">
              <div className="grid gap-3 text-sm">
                <div className="rounded-xl bg-black/30 p-3 border border-emerald/20">
                  <p className="text-ash text-xs uppercase">Best track</p>
                  <p>{summaryData.best_track.track_name}</p>
                </div>
                <div className="rounded-xl bg-black/30 p-3 border border-emerald/20">
                  <p className="text-ash text-xs uppercase">Worst track</p>
                  <p>{summaryData.worst_track.track_name}</p>
                </div>
              </div>
              <div className="text-xs text-ash">
                Nearest pair: {summaryData.nearest_pair.a.track_name} · {summaryData.nearest_pair.b.track_name}
              </div>
              <div className="text-xs text-ash">
                Farthest pair: {summaryData.farthest_pair.a.track_name} · {summaryData.farthest_pair.b.track_name}
              </div>
            </div>
          )}
        </Card>

        <Card title="Cluster plots">
          <div className="space-y-4">
            <img
              src={plotUrl('cluster_scatter', genre || undefined)}
              alt="Cluster scatter"
              className="w-full rounded-xl border border-emerald/20"
            />
            <img
              src={plotUrl('cluster_popularity', genre || undefined)}
              alt="Cluster popularity"
              className="w-full rounded-xl border border-emerald/20"
            />
            <img
              src={plotUrl('cluster_heatmap', genre || undefined)}
              alt="Cluster heatmap"
              className="w-full rounded-xl border border-emerald/20"
            />
          </div>
        </Card>
      </div>
    </div>
  )
}

export default ClusteringLab
