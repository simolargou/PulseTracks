export const Loading = ({ label = 'Loading…' }) => (
  <div className="text-ash text-sm animate-pulse">{label}</div>
)

export const ErrorState = ({ message }) => (
  <div className="text-red-300 text-sm">{message || 'Something went wrong.'}</div>
)
