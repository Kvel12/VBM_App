import { useState, useEffect, useRef } from 'react';
import Brain from '../components/Brain.jsx';
import Spinner from '../components/Spinner.jsx';

const ProcessingScreen = ({ model, info, file, onCancel, onDone }) => {
  const [done, setDone] = useState([]);
  const [cur, setCur] = useState(0);
  const timers = useRef([]);

  useEffect(() => {
    let delay = 600;
    model.steps.forEach((_, i) => {
      const t1 = setTimeout(() => setCur(i), delay);
      delay += 1200 + Math.random() * 1000;
      const t2 = setTimeout(() => setDone((p) => [...p, i]), delay);
      timers.current.push(t1, t2);
    });
    const tf = setTimeout(() => onDone(), delay + 700);
    timers.current.push(tf);
    return () => timers.current.forEach(clearTimeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pct = Math.round((done.length / model.steps.length) * 100);
  const allDone = done.length === model.steps.length;
  const st = (i) => (done.includes(i) ? 'done' : i === cur && !allDone ? 'act' : 'pend');

  return (
    <div className="fu">
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 30 }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 19 }}>{model.fullName}</div>
          <div style={{ fontSize: 13.5, color: 'var(--t2)', marginTop: 3 }}>
            Prueba: <strong>{info.test}</strong>
            {info.patient ? ` · ${info.patient}` : ''}
          </div>
        </div>
        <button className="btn btn-d btn-sm" onClick={onCancel}>⬛ Cancelar</button>
      </div>

      <div className="rgrid" style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 24 }}>
        {/* Left: brain + progress */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '22px 0', position: 'relative' }}>
            <div style={{ position: 'relative', width: 140, height: 140 }}>
              {!allDone &&
                [0, 1, 2].map((i) => (
                  <div
                    key={i}
                    style={{
                      position: 'absolute',
                      inset: 0,
                      borderRadius: '50%',
                      border: '2px solid oklch(0.50 0.18 242 / 0.22)',
                      animation: `ripple 2s ease ${i * 0.65}s infinite`,
                    }}
                  />
                ))}
              <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Brain size={78} color="var(--primary)" glow={!allDone} />
              </div>
            </div>
          </div>

          <div className="card" style={{ textAlign: 'center', padding: 20 }}>
            <div style={{ fontSize: 13, color: 'var(--t3)', marginBottom: 8 }}>Progreso general</div>
            <div className="pt" style={{ marginBottom: 10 }}>
              <div className="pf" style={{ width: `${pct}%` }} />
            </div>
            <div style={{ fontWeight: 800, fontSize: 28, color: 'var(--primary)' }}>{pct}%</div>
            <div style={{ fontSize: 13, color: 'var(--t3)', marginTop: 4 }}>
              Paso {Math.min(done.length + (allDone ? 0 : 1), model.steps.length)} de {model.steps.length}
            </div>
          </div>

          <div className="wb" style={{ fontSize: 12.5 }}>
            <span style={{ flexShrink: 0 }}>⏱</span>
            <span>
              El proceso puede tardar <strong>5–15 min</strong> en equipos reales. No cierre la aplicación.
            </span>
          </div>
        </div>

        {/* Right: steps */}
        <div className="card">
          <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 16 }}>Estado del proceso</div>
          <div className="slist">
            {model.steps.map((s, i) => {
              const status = st(i);
              return (
                <div key={i} className={`srow ${status}`}>
                  <div className="sdot">
                    {status === 'done' ? (
                      '✓'
                    ) : status === 'act' ? (
                      <Spinner />
                    ) : (
                      <span style={{ fontSize: 12.5, color: 'var(--t3)' }}>{i + 1}</span>
                    )}
                  </div>
                  <div className="sinfo">
                    <div className="sname">{s.name}</div>
                    <div className="sdet">
                      {status === 'done' ? '✓ Completado' : status === 'act' ? 'Procesando…' : s.det}
                    </div>
                  </div>
                  {status === 'act' && (
                    <span style={{ fontSize: 11.5, color: 'var(--primary)', fontWeight: 700, flexShrink: 0 }}>
                      EN CURSO
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          {allDone && (
            <div className="ib" style={{ marginTop: 16 }}>
              <span>✅</span>
              <span><strong>Proceso completado.</strong> Cargando resultados…</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProcessingScreen;
