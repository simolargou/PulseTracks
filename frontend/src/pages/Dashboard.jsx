import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid
} from 'recharts'
import Card from '../components/Card'
import { ErrorState, Loading } from '../components/Status'
import { fetchAnalytics, fetchGenres, searchTracks } from '../api/client'

const Dashboard = () => {
  const [query, setQuery] = useState('')
  const [genre, setGenre] = useState('')

  const { data: genresData } = useQuery({ queryKey: ['genres'], queryFn: fetchGenres })
  const { data: analytics, isLoading: analyticsLoading, error: analyticsError } = useQuery({
    queryKey: ['analytics', genre],
    queryFn: () => fetchAnalytics(genre || undefined)
  })
  const {
    data: searchData,
    isLoading: searchLoading,
    error: searchError
  } = useQuery({
    queryKey: ['search', query, genre],
    queryFn: () => searchTracks({ query, genre: genre || undefined, limit: 12 })
  })

  const histogram = useMemo(() => {
    if (!analytics) return []
    const { bins, counts } = analytics.popularity_histogram
    return counts.map((count, idx) => ({
      bucket: `${Math.round(bins[idx])}-${Math.round(bins[idx + 1])}`,
      count
    }))
  }, [analytics])

  const clusterSizes = useMemo(() => {
    if (!analytics) return []
    return Object.entries(analytics.cluster_sizes).map(([cluster, size]) => ({
      cluster: `C${cluster}`,
      size
    }))
  }, [analytics])

  return (
    <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
      <div className="space-y-6">
        <Card title="Track search">
          <div className="flex flex-col gap-4 md:flex-row">
            <input
              className="w-full rounded-xl bg-black/40 border border-emerald/40 px-4 py-3 text-sm outline-none focus:border-moss"
              placeholder="Search tracks, artists, albums"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <select
              className="rounded-xl bg-black/40 border border-emerald/40 px-4 py-3 text-sm"
              value={genre}
              onChange={(event) => setGenre(event.target.value)}
            >
              <option value="">All genres</option>
              {genresData?.genres?.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
          {searchLoading && <Loading />}
          {searchError && <ErrorState message="Search failed." />}
          <div className="grid gap-3">
            {searchData?.items?.map((track) => (
              <div
                key={track.track_id}
                className="flex items-center justify-between rounded-xl border border-emerald/30 bg-black/30 px-4 py-3 text-sm"
              >
                <Link to={`/track/${track.track_id}`} className="flex-1 transition hover:text-moss">
                  <div>
                    <p className="font-medium text-white">{track.track_name}</p>
                    <p className="text-ash">{track.artists}</p>
                  </div>
                </Link>
                <div className="flex items-center gap-3">
                  <span className="text-xs uppercase text-ash">{track.track_genre}</span>
                  <a
                    href={`https://open.spotify.com/track/${track.track_id}`}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-full border border-emerald/30 px-3 py-1 text-xs uppercase tracking-widest text-moss transition hover:border-moss"
                  >
                    Play
                  </a>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Popularity pulse">
          {analyticsLoading && <Loading />}
          {analyticsError && <ErrorState message="Analytics unavailable." />}
          {analytics && (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={histogram}>
                  <defs>
                    <linearGradient id="mossFade" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#1db954" stopOpacity={0.8} />
                      <stop offset="100%" stopColor="#1db954" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2a24" />
                  <XAxis dataKey="bucket" tick={{ fill: '#9aa4a8', fontSize: 10 }} interval={1} />
                  <YAxis tick={{ fill: '#9aa4a8', fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      background: '#0b0f0c',
                      border: '1px solid #1db954',
                      color: '#fff'
                    }}
                  />
                  <Area type="monotone" dataKey="count" stroke="#1db954" fill="url(#mossFade)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>
      </div>

      <div className="space-y-6">
        <Card title="Cluster sizes">
          {analyticsLoading && <Loading />}
          {analyticsError && <ErrorState message="Analytics unavailable." />}
          {analytics && (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={clusterSizes}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2a24" />
                  <XAxis dataKey="cluster" tick={{ fill: '#9aa4a8', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#9aa4a8', fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      background: '#0b0f0c',
                      border: '1px solid #1db954',
                      color: '#fff'
                    }}
                  />
                  <Bar dataKey="size" fill="#1db954" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card title="Feature means">
          {analyticsLoading && <Loading />}
          {analyticsError && <ErrorState message="Analytics unavailable." />}
          {analytics && (
            <div className="grid grid-cols-2 gap-3 text-sm">
              {Object.entries(analytics.feature_means).map(([key, value]) => (
                <div key={key} className="rounded-xl bg-black/30 p-3 border border-emerald/20">
                  <p className="text-ash uppercase text-xs">{key}</p>
                  <p className="text-white font-medium">{value.toFixed(2)}</p>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}

export default Dashboard
