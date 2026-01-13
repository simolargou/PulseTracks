import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import TrackDetail from './pages/TrackDetail'
import ClusteringLab from './pages/ClusteringLab'

const App = () => {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/track/:trackId" element={<TrackDetail />} />
        <Route path="/cluster" element={<ClusteringLab />} />
      </Routes>
    </Layout>
  )
}

export default App
