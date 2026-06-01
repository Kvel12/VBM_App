import Brain from '../components/Brain.jsx';
import { MODELS } from '../data/models.js';

const HomeScreen = ({ onSelect }) => (
  <div className="fu">
    <div className="hero">
      <div style={{ display: 'inline-block', marginBottom: 20 }}>
        <Brain size={82} color="var(--primary)" glow />
      </div>
      <h1>
        Análisis de <span>Biomarcadores</span>
        <br />
        Neurológicos
      </h1>
      <p style={{ marginTop: 10 }}>
        Tres modelos de IA para la identificación de epilepsia mediante morfometría basada en vóxeles (VBM).
      </p>
      <p style={{ fontSize: 14, color: 'var(--t3)', marginTop: 6 }}>Seleccione un modelo para comenzar.</p>
    </div>

    <div className="ib" style={{ marginBottom: 24 }}>
      <span>⏱️</span>
      <span>
        El proceso completo tarda entre <strong>5 y 15 minutos</strong> dependiendo de la RAM y potencia del equipo.
      </span>
    </div>

    <div className="mgrid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 22 }}>
      {MODELS.map((m) => (
        <div key={m.id} className={`mcard${m.recommended ? ' rec' : ''}`} onClick={() => onSelect(m)}>
          {m.recommended && <div className="rec-lbl">★ MAYOR PRECISIÓN</div>}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: 14,
                background: 'var(--pl)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}
            >
              <Brain size={30} color="var(--primary)" />
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 3 }}>{m.name}</div>
              <span className={`badge ${m.badge === 'Segmentación' ? 'b-am' : 'b-bl'}`}>{m.badge}</span>
            </div>
          </div>
          <p style={{ fontSize: 13.5, color: 'var(--t2)', lineHeight: 1.6 }}>{m.desc}</p>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 2 }}>
            <div className="tw">
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  background: 'var(--pl)',
                  padding: '6px 12px',
                  borderRadius: 99,
                  cursor: 'help',
                }}
              >
                <span style={{ fontWeight: 800, color: 'var(--primary)', fontSize: 17 }}>{m.metricVal}</span>
                <span style={{ fontSize: 11.5, color: 'var(--pd)', fontWeight: 600 }}>{m.metricLabel}</span>
              </div>
              <div className="tt">{m.tooltip}</div>
            </div>
            <span style={{ fontSize: 12.5, color: 'var(--t3)' }}>{m.steps.length} pasos</span>
          </div>
        </div>
      ))}
    </div>
  </div>
);

export default HomeScreen;
