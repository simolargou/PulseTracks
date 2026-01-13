const Card = ({ title, children, actions }) => {
  return (
    <section className="glass rounded-2xl p-5 md:p-6 shadow-glow fade-in">
      <div className="flex items-start justify-between gap-4">
        <div>
          {title && <h2 className="text-lg font-display text-white">{title}</h2>}
        </div>
        {actions}
      </div>
      <div className="mt-4 space-y-3">{children}</div>
    </section>
  )
}

export default Card
