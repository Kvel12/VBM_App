import { useState } from 'react';
import DropZone from '../components/DropZone.jsx';

const UploadScreen = ({ model, onStart, onBack }) => {
  const [info, setInfo] = useState({ test: '', patient: '', notes: '' });
  const [file, setFile] = useState(null);
  const u = (k, v) => setInfo((p) => ({ ...p, [k]: v }));
  const ok = file && info.test.trim();

  return (
    <div className="fu">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 28 }}>
        <button className="btn btn-g btn-sm" onClick={onBack}>← Volver</button>
        <div>
          <div style={{ fontWeight: 700, fontSize: 19 }}>{model.fullName}</div>
          <div style={{ fontSize: 13.5, color: 'var(--t2)' }}>
            {model.metricLabel}: {model.metricVal} · {model.badge}
          </div>
        </div>
      </div>

      <div className="fgrid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 22 }}>
        {/* Form */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 17, marginBottom: 3 }}>Información del análisis</div>
            <div style={{ fontSize: 13, color: 'var(--t3)' }}>Se incluirá en el reporte exportado.</div>
          </div>
          <div className="ff">
            <label className="fl">Nombre de la prueba *</label>
            <input
              className="fi"
              placeholder="ej. Prueba_001_Control"
              value={info.test}
              onChange={(e) => u('test', e.target.value)}
            />
          </div>
          <div className="ff">
            <label className="fl">Nombre del paciente (opcional)</label>
            <input
              className="fi"
              placeholder="ej. Paciente A"
              value={info.patient}
              onChange={(e) => u('patient', e.target.value)}
            />
          </div>
          <div className="ff">
            <label className="fl">Notas adicionales</label>
            <textarea
              className="fta"
              rows={3}
              placeholder="Observaciones clínicas, contexto..."
              value={info.notes}
              onChange={(e) => u('notes', e.target.value)}
            />
          </div>
          <div className="ib" style={{ fontSize: 12.5 }}>
            <span>ℹ️</span>
            <span>Solo se acepta <strong>una imagen T1</strong> por análisis.</span>
          </div>
        </div>

        {/* Upload + step preview */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div className="card">
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 13 }}>Imagen T1 — .nii / .gz</div>
            <DropZone file={file} onChange={setFile} />
          </div>
          <div
            className="card"
            style={{ background: 'var(--pl)', borderColor: 'oklch(0.86 0.06 235)', padding: '18px 20px' }}
          >
            <div style={{ fontWeight: 700, fontSize: 13.5, color: 'var(--pd)', marginBottom: 10 }}>
              Pasos del proceso · {model.steps.length}
            </div>
            {model.steps.map((s, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 9,
                  padding: '5px 0',
                  borderBottom: i < model.steps.length - 1 ? '1px solid oklch(0.89 0.05 235)' : 'none',
                }}
              >
                <div
                  style={{
                    width: 22,
                    height: 22,
                    borderRadius: '50%',
                    background: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 11,
                    fontWeight: 700,
                    color: 'var(--primary)',
                    flexShrink: 0,
                  }}
                >
                  {i + 1}
                </div>
                <span style={{ fontSize: 13, color: 'var(--pd)' }}>{s.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          justifyContent: 'flex-end',
          alignItems: 'center',
          gap: 12,
          marginTop: 24,
          flexWrap: 'wrap',
        }}
      >
        {!ok && (
          <span style={{ fontSize: 13, color: 'var(--t3)' }}>
            {!file ? '⚠ Adjunta la imagen T1' : '⚠ Ingresa el nombre de la prueba'}
          </span>
        )}
        <button className="btn btn-g" onClick={onBack}>Cancelar</button>
        <button className="btn btn-p btn-lg" disabled={!ok} onClick={() => onStart(info, file)}>
          Iniciar análisis →
        </button>
      </div>
    </div>
  );
};

export default UploadScreen;
