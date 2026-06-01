import { useState, useEffect } from 'react';
import Brain from '../components/Brain.jsx';
import Gauge from '../components/Gauge.jsx';

const ResultsScreen = ({ model, info, file, onReset }) => {
  const [gv, setGv] = useState(0);
  const isClass = model.type === 'classification';
  const r = model.sim;

  useEffect(() => {
    const t = setTimeout(() => setGv(isClass ? r.epilepsy : r.dsc), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const genTxt = () => {
    const now = new Date().toLocaleString('es-CO');
    const sep = '═══════════════════════════════════════════════';
    const lines = [
      sep,
      '   REPORTE VBM App — Análisis Neurológico',
      sep,
      '',
      `Fecha:         ${now}`,
      `Modelo:        ${model.fullName}`,
      `Prueba:        ${info.test}`,
      `Paciente:      ${info.patient || 'No especificado'}`,
      `Archivo:       ${file.name}`,
      '',
      '──────────────── RESULTADO ──────────────────',
      isClass
        ? [
            `Clase:         ${r.epilepsy >= 50 ? 'Posible Epilepsia' : 'Posible Control'}`,
            `Prob. Epilepsia:   ${r.epilepsy} %`,
            `Prob. Control:     ${r.control} %`,
          ].join('\n')
        : [
            `Tipo:          Segmentación`,
            `DSC obtenido:  ${r.dsc} %`,
            `Máscara:       Disponible para descarga`,
          ].join('\n'),
      '',
      '──────────────── MÉTRICAS DEL MODELO ────────',
      ...Object.entries(model.metrics).map(([k, v]) => `${k.padEnd(18)}${v}`),
      '',
      '──────────────── ADVERTENCIA ─────────────────',
      'Este resultado es orientativo y NO constituye diagnóstico.',
      'Debe ser revisado por un neurólogo antes de tomar decisiones.',
      info.notes ? `\nNotas: ${info.notes}` : '',
      '',
      sep,
      'VBM App v1.0 — Morfometría Basada en Vóxeles',
    ]
      .filter((l) => l !== undefined)
      .join('\n');

    const blob = new Blob([lines], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `VBM_Reporte_${info.test.replace(/\s+/g, '_')}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const isEpi = isClass && r.epilepsy >= 50;

  return (
    <div className="fu">
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 28 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div
            style={{
              width: 42,
              height: 42,
              borderRadius: 12,
              background: 'var(--pl)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Brain size={26} color="var(--primary)" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 19 }}>Resultados del análisis</div>
            <div style={{ fontSize: 13.5, color: 'var(--t2)' }}>
              {model.fullName} · <strong>{info.test}</strong>
              {info.patient ? ` · ${info.patient}` : ''}
            </div>
          </div>
        </div>
        <button className="btn btn-g btn-sm" onClick={onReset}>← Nuevo análisis</button>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        {isClass ? (
          <div className="rgrid" style={{ display: 'grid', gridTemplateColumns: '230px 1fr', gap: 28, alignItems: 'start' }}>
            {/* Gauge */}
            <div className="gauge-zone">
              <div
                style={{
                  fontSize: 11.5,
                  color: 'var(--t3)',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '.5px',
                }}
              >
                Confianza del modelo
              </div>
              <Gauge value={gv} color={isEpi ? 'var(--bad)' : 'var(--ok)'} />
              <div className={`verdict ${isEpi ? 'v-pos' : 'v-neg'}`}>
                {isEpi ? '⚠ Posible Epilepsia' : '✓ Posible Control'}
              </div>
            </div>
            {/* Bars + metrics */}
            <div>
              <div style={{ fontWeight: 700, fontSize: 17, marginBottom: 16 }}>Probabilidades de clasificación</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div className="pbrow">
                  <div className="pblab">
                    <span>Epilepsia</span>
                    <span style={{ fontWeight: 700, color: 'var(--bad)' }}>{gv} %</span>
                  </div>
                  <div className="pbt">
                    <div className="pbf" style={{ width: `${gv}%`, background: 'var(--bad)' }} />
                  </div>
                </div>
                <div className="pbrow">
                  <div className="pblab">
                    <span>Control</span>
                    <span style={{ fontWeight: 700, color: 'var(--ok)' }}>{100 - gv} %</span>
                  </div>
                  <div className="pbt">
                    <div className="pbf" style={{ width: `${100 - gv}%`, background: 'var(--ok)' }} />
                  </div>
                </div>
              </div>
              <div className="mtiles">
                {Object.entries(model.metrics).map(([k, v]) => (
                  <div key={k} className="mtile">
                    <div className="mv">{v}</div>
                    <div className="ml">{k}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="rgrid" style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 28 }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
              <div className="seg-ph">
                <span style={{ fontSize: 42 }}>🧠</span>
                <span style={{ fontFamily: 'monospace', lineHeight: 1.45 }}>
                  máscara de segmentación
                  <br />
                  [zona epileptogénica]
                </span>
              </div>
              <button
                className="btn btn-ok btn-sm"
                onClick={() => alert('La descarga de la máscara estará disponible en la versión de producción.')}
              >
                ⬇ Descargar máscara (.nii)
              </button>
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 17, marginBottom: 12 }}>Segmentación completada</div>
              <p style={{ fontSize: 14.5, color: 'var(--t2)', lineHeight: 1.65, marginBottom: 20 }}>
                nnUNet ha localizado y segmentado la zona epileptogénica en la imagen T1. La máscara está lista para descarga en
                formato <strong>.nii</strong> o <strong>.gz</strong>.
              </p>
              <div className="mtiles" style={{ gridTemplateColumns: 'repeat(2,1fr)' }}>
                {Object.entries(model.metrics).map(([k, v]) => (
                  <div key={k} className="mtile">
                    <div className="mv">{v}</div>
                    <div className="ml">{k}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Disclaimer */}
      <div className="wb" style={{ marginBottom: 20 }}>
        <span style={{ fontSize: 22, flexShrink: 0 }}>⚕️</span>
        <div>
          <strong>Aviso médico importante:</strong> Este análisis es una herramienta de apoyo diagnóstico y{' '}
          <strong>no reemplaza el criterio clínico</strong>. Los resultados deben ser interpretados por un médico
          especialista (neurólogo). No tome decisiones clínicas basándose únicamente en este resultado.
        </div>
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end', flexWrap: 'wrap' }}>
        {model.type === 'segmentation' && (
          <button
            className="btn btn-ok"
            onClick={() => alert('La descarga de la máscara estará disponible en producción.')}
          >
            ⬇ Máscara (.nii / .gz)
          </button>
        )}
        <button className="btn btn-g" onClick={genTxt}>📄 Exportar reporte .txt</button>
        <button className="btn btn-p" onClick={onReset}>+ Nuevo análisis</button>
      </div>
    </div>
  );
};

export default ResultsScreen;
