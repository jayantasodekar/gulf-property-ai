import { useEffect, useRef, useState } from 'react'

const SUGGESTIONS = [
  'Show me 3-bedroom apartments in Jeddah under 2 million SAR',
  'What DarGlobal projects are in Dubai?',
  'What is the average price of villas in Riyadh?',
  'Compare apartments for rent in Riyadh and Jeddah',
  'أرني شقق للبيع في الرياض',
]

const SOURCE_META = {
  wasalt: { label: 'Wasalt', cls: 'src-wasalt' },
  darglobal: { label: 'DarGlobal', cls: 'src-darglobal' },
}

function fmtPrice(p) {
  if (!p.price) return 'Price on application'
  const n = p.price >= 1e6 ? `${(p.price / 1e6).toFixed(2)}M` : p.price.toLocaleString()
  return `${n} ${p.currency || ''}`.trim()
}

function PropertyCard({ p }) {
  const meta = SOURCE_META[p.source] || { label: p.source, cls: '' }
  const facts = [
    p.property_type,
    p.bedrooms ? `${p.bedrooms} bed` : null,
    p.bathrooms ? `${p.bathrooms} bath` : null,
    p.area_sqm ? `${Math.round(p.area_sqm)} m²` : null,
  ].filter(Boolean)
  const place = [p.district, p.city, p.country].filter(Boolean).join(', ')

  return (
    <a className="card" href={p.url} target="_blank" rel="noopener noreferrer">
      <div className="card-img">
        {p.images?.[0] ? (
          <img src={p.images[0]} alt="" loading="lazy" />
        ) : (
          <div className="card-img-ph" aria-hidden="true">⌂</div>
        )}
        <span className={`badge ${meta.cls}`}>{meta.label}</span>
      </div>
      <div className="card-body">
        <div className="card-title">{p.title}</div>
        <div className="card-price">{fmtPrice(p)}</div>
        {facts.length > 0 && <div className="card-facts">{facts.join(' · ')}</div>}
        {place && <div className="card-place">{place}</div>}
        {p.completion_status && <div className="card-status">{p.completion_status}</div>}
      </div>
    </a>
  )
}

/** Minimal markdown: **bold**, bullets, paragraphs. Text is escaped by React. */
function Rich({ text }) {
  const blocks = text.split(/\n{2,}/)
  return (
    <>
      {blocks.map((block, bi) => {
        const lines = block.split('\n')
        const isList = lines.every((l) => /^\s*[-*]\s+/.test(l) || !l.trim())
        const render = (s, i) => {
          const parts = s.split(/(\*\*[^*]+\*\*)/g)
          return parts.map((part, pi) =>
            part.startsWith('**') && part.endsWith('**') ? (
              <strong key={`${i}-${pi}`}>{part.slice(2, -2)}</strong>
            ) : (
              <span key={`${i}-${pi}`}>{part}</span>
            ),
          )
        }
        if (isList && lines.some((l) => l.trim())) {
          return (
            <ul key={bi}>
              {lines
                .filter((l) => l.trim())
                .map((l, li) => (
                  <li key={li}>{render(l.replace(/^\s*[-*]\s+/, ''), li)}</li>
                ))}
            </ul>
          )
        }
        return <p key={bi}>{render(block, bi)}</p>
      })}
    </>
  )
}

function isRTL(s) {
  return /[؀-ۿ]/.test(s || '')
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const [stats, setStats] = useState(null)
  const [showInfo, setShowInfo] = useState(false)
  const [health, setHealth] = useState({ state: 'checking' })
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  // The free hosting tier suspends the instance after ~15 minutes idle, so a
  // cold visitor waits ~50s for the container to start. Without this the page
  // just looks broken. Poll /healthz on load and say what is happening.
  useEffect(() => {
    let cancelled = false
    let attempt = 0

    async function probe() {
      while (!cancelled && attempt < 30) {
        attempt += 1
        try {
          const r = await fetch('/healthz', { cache: 'no-store' })
          if (r.ok) {
            const h = await r.json()
            if (!cancelled) setHealth({ state: 'ready', ...h })
            fetch('/api/stats')
              .then((x) => x.json())
              .then((d) => !cancelled && setStats(d))
              .catch(() => {})
            return
          }
        } catch {
          /* instance still starting */
        }
        if (!cancelled) setHealth({ state: attempt > 2 ? 'waking' : 'checking' })
        await new Promise((res) => setTimeout(res, 3000))
      }
      if (!cancelled) setHealth({ state: 'down' })
    }

    probe()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, status])

  async function send(text) {
    const question = (text ?? input).trim()
    if (!question || busy) return
    setInput('')
    setBusy(true)
    setStatus('Thinking…')

    const history = messages
      .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content))
      .slice(-10)
      .map((m) => ({ role: m.role, content: m.content }))

    setMessages((m) => [
      ...m,
      { role: 'user', content: question },
      { role: 'assistant', content: '', properties: [], meta: null },
    ])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question, history }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.error || `Request failed (${res.status})`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const chunks = buffer.split('\n\n')
        buffer = chunks.pop() || ''

        for (const chunk of chunks) {
          const line = chunk.split('\n').find((l) => l.startsWith('data:'))
          if (!line) continue
          let ev
          try {
            ev = JSON.parse(line.slice(5).trim())
          } catch {
            continue
          }

          if (ev.type === 'status') setStatus(ev.message)
          else if (ev.type === 'citations') {
            setMessages((m) => {
              const c = [...m]
              c[c.length - 1] = { ...c[c.length - 1], properties: ev.properties || [] }
              return c
            })
          } else if (ev.type === 'token') {
            setStatus('')
            setMessages((m) => {
              const c = [...m]
              const last = c[c.length - 1]
              c[c.length - 1] = { ...last, content: last.content + ev.text }
              return c
            })
          } else if (ev.type === 'error') {
            setMessages((m) => {
              const c = [...m]
              c[c.length - 1] = { ...c[c.length - 1], content: ev.message, isError: true }
              return c
            })
          } else if (ev.type === 'done') {
            setMessages((m) => {
              const c = [...m]
              c[c.length - 1] = { ...c[c.length - 1], meta: ev.meta }
              return c
            })
          }
        }
      }
    } catch (e) {
      setMessages((m) => {
        const c = [...m]
        c[c.length - 1] = {
          ...c[c.length - 1],
          content: e.message || 'Could not reach the server.',
          isError: true,
        }
        return c
      })
    } finally {
      setBusy(false)
      setStatus('')
      inputRef.current?.focus()
    }
  }

  const total = stats?.corpus?.total
  const bySource = stats?.corpus?.by_source || {}

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo" aria-hidden="true">◆</span>
          <div>
            <h1>Gulf Property AI</h1>
            <p className="tagline">
              Grounded answers over {total ? total.toLocaleString() : '—'} listings from
              DarGlobal &amp; Wasalt
            </p>
          </div>
        </div>
        <button className="info-btn" onClick={() => setShowInfo((v) => !v)}>
          {showInfo ? 'Close' : 'How it works'}
        </button>
      </header>

      {showInfo && (
        <div className="info-panel">
          <div>
            <h3>Data</h3>
            <p>
              {bySource.wasalt?.toLocaleString() || 0} Wasalt listings (Saudi resale &amp;
              rental) and {bySource.darglobal || 0} DarGlobal developments, scraped from
              public pages and indexed as a point-in-time snapshot.
            </p>
          </div>
          <div>
            <h3>Retrieval</h3>
            <p>
              Hybrid search: SQL filters for hard constraints (price, bedrooms), BM25 for
              keywords, and multilingual embeddings for meaning — fused with Reciprocal
              Rank Fusion. Aggregates are computed in SQL, never estimated by the model.
            </p>
          </div>
          <div>
            <h3>Model</h3>
            <p>
              A free tool-calling model via OpenRouter, with automatic fallback across
              several models. Answers are grounded in retrieved listings only.
              {stats?.corpus?.embedding_model && (
                <>
                  {' '}
                  Embeddings: <code>{stats.corpus.embedding_model.split('/').pop()}</code>.
                </>
              )}
            </p>
          </div>
        </div>
      )}

      {health.state === 'waking' && (
        <div className="banner">
          <span className="spin" aria-hidden="true" />
          Waking the server up &mdash; free hosting suspends it when idle. This
          takes about a minute the first time, then it is fast.
        </div>
      )}
      {health.state === 'down' && (
        <div className="banner banner-warn">
          The server is not responding. It may be restarting &mdash; try
          reloading in a minute.
        </div>
      )}
      {health.state === 'ready' && health.ai_quota_exhausted && (
        <div className="banner">
          The daily free-tier AI allowance is used up, so replies are direct
          search results rather than written answers. Search is unaffected.
        </div>
      )}

      <main className="chat">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>Ask about properties across Saudi Arabia and the Gulf</h2>
            <p>
              Every answer is grounded in scraped listing data, with the source listings
              shown alongside. Note that DarGlobal publishes developments on a
              register-interest basis, so unit prices are not public.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => send(s)} dir={isRTL(s) ? 'rtl' : 'ltr'}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="avatar">{m.role === 'user' ? 'You' : 'AI'}</div>
            <div className="bubble-wrap">
              <div
                className={`bubble ${m.isError ? 'error' : ''}`}
                dir={isRTL(m.content) ? 'rtl' : 'ltr'}
              >
                {m.content ? (
                  <Rich text={m.content} />
                ) : (
                  m.role === 'assistant' && <span className="dots"><i /><i /><i /></span>
                )}
              </div>

              {m.properties?.length > 0 && (
                <>
                  <div className="cards-label">
                    {m.properties.length} source listing
                    {m.properties.length > 1 ? 's' : ''}
                  </div>
                  <div className="cards">
                    {m.properties.map((p) => (
                      <PropertyCard key={p.id} p={p} />
                    ))}
                  </div>
                </>
              )}

              {m.meta && (
                <div className="meta">
                  mode: {m.meta.mode}
                  {m.meta.model ? ` · ${m.meta.model.split('/').pop()}` : ''}
                </div>
              )}
            </div>
          </div>
        ))}

        {status && <div className="status">{status}</div>}
        <div ref={bottomRef} />
      </main>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          send()
        }}
      >
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about listings, prices, locations…"
          maxLength={2000}
          disabled={busy || health.state === 'waking'}
          dir={isRTL(input) ? 'rtl' : 'ltr'}
          aria-label="Your question"
        />
        <button
          type="submit"
          disabled={busy || !input.trim() || health.state === 'waking'}
        >
          {busy ? '…' : 'Send'}
        </button>
      </form>

      <footer className="footer">
        Public data from darglobal.co.uk and wasalt.sa · point-in-time snapshot, not a
        live feed · not financial advice
      </footer>
    </div>
  )
}
