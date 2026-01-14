import { useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import Card from '../components/Card'
import { ErrorState, Loading } from '../components/Status'
import { fetchSimilar, fetchTrack, recommendTracks } from '../api/client'

const vibeFeatures = [
  'danceability',
  'energy',
  'valence',
  'acousticness',
  'speechiness',
  'instrumentalness',
  'liveness'
]

const TrackDetail = () => {
  const { trackId } = useParams()
  const [limit, setLimit] = useState(8)
  const [genre, setGenre] = useState('')
  const [target, setTarget] = useState(
    Object.fromEntries(vibeFeatures.map((item) => [item, 0]))
  )
  const trackUrl = (id) => `https://open.spotify.com/track/${id}`

  const { data: trackData, isLoading, error } = useQuery({
    queryKey: ['track', trackId],
    queryFn: () => fetchTrack(trackId)
  })

  const {
    data: similarData,
    isLoading: similarLoading,
    error: similarError
  } = useQuery({
    queryKey: ['similar', trackId, limit, genre],
    queryFn: () => fetchSimilar({ trackId, limit, genre: genre || undefined })
  })

  const recommendMutation = useMutation({
    mutationFn: (payload) => recommendTracks(payload)
  })

  const handleSlider = (feature, value) => {
    setTarget((prev) => ({ ...prev, [feature]: parseFloat(value) }))
  }

  const recommendationPayload = useMemo(
    () => ({
      genre: genre || undefined,
      limit,
      seed_track_id: trackId,
      target_features: target
    }),
    [genre, limit, target, trackId]
  )

  return (
    <div className="space-y-6">
      <Card title="Track detail">
        {isLoading && <Loading />}
        {error && <ErrorState message="Unable to load track." />}
        {trackData && (
          <div className="space-y-3">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
              <div>
                <p className="text-2xl font-display">{trackData.track.track_name}</p>
                <p className="text-ash">{trackData.track.artists}</p>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-sm text-ash">Genre: {trackData.track.track_genre}</div>
                <a
                  href={trackUrl(trackData.track.track_id)}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-full border border-emerald/40 px-4 py-2 text-xs uppercase tracking-widest text-moss transition hover:border-moss"
                >
                  Play
                </a>
              </div>
            </div>
          </div>
        )}
      </Card>

      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <Card title="Similar tracks">
          <div className="flex flex-col md:flex-row gap-3">
            <label className="text-xs uppercase text-ash">
              Limit
              <input
                type="number"
                min="1"
                max="20"
                value={limit}
                onChange={(event) => setLimit(parseInt(event.target.value, 10))}
                className="mt-1 w-full rounded-lg bg-black/40 border border-emerald/40 px-3 py-2 text-sm"
              />
            </label>
            <label className="text-xs uppercase text-ash">
              Genre filter
              <input
                type="text"
                value={genre}
                onChange={(event) => setGenre(event.target.value)}
                placeholder="Optional"
                className="mt-1 w-full rounded-lg bg-black/40 border border-emerald/40 px-3 py-2 text-sm"
              />
            </label>
          </div>
          {similarLoading && <Loading />}
          {similarError && <ErrorState message="Unable to load similar tracks." />}
          <div className="space-y-3">
            {similarData?.items?.map((item) => (
              <div key={item.track.track_id} className="rounded-xl border border-emerald/30 bg-black/30 px-4 py-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-medium">{item.track.track_name}</p>
                    <p className="text-ash text-sm">{item.track.artists}</p>
                  </div>
                  <a
                    href={trackUrl(item.track.track_id)}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-full border border-emerald/30 px-3 py-1 text-xs uppercase tracking-widest text-moss transition hover:border-moss"
                  >
                    Play
                  </a>
                </div>
                <p className="text-xs text-ash">Cluster {item.cluster} · Distance {item.distance.toFixed(3)}</p>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Vibe recommendations" actions={
          <button
            onClick={() => recommendMutation.mutate(recommendationPayload)}
            className="rounded-full bg-moss px-4 py-2 text-xs uppercase tracking-widest text-ink"
          >
            Recommend
          </button>
        }>
          <div className="space-y-4">
            {vibeFeatures.map((feature) => (
              <label key={feature} className="text-xs uppercase text-ash">
                {feature}
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={target[feature]}
                  onChange={(event) => handleSlider(feature, event.target.value)}
                  className="mt-2 w-full"
                />
              </label>
            ))}
          </div>
          {recommendMutation.isLoading && <Loading label="Computing recommendations..." />}
          {recommendMutation.error && <ErrorState message="Recommendation failed." />}
          <div className="space-y-3">
            {recommendMutation.data?.items?.map((item) => (
              <div key={item.track.track_id} className="rounded-xl border border-emerald/30 bg-black/30 px-4 py-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-medium">{item.track.track_name}</p>
                    <p className="text-ash text-sm">{item.track.artists}</p>
                  </div>
                  <a
                    href={trackUrl(item.track.track_id)}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-full border border-emerald/30 px-3 py-1 text-xs uppercase tracking-widest text-moss transition hover:border-moss"
                  >
                    Play
                  </a>
                </div>
                <p className="text-xs text-ash">Cluster {item.cluster} · Distance {item.distance.toFixed(3)}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}

export default TrackDetail
