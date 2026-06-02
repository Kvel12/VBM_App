import { useState, useEffect } from 'react';
import Brain from '../components/Brain.jsx';
import Gauge from '../components/Gauge.jsx';
import { useT, useLang } from '../i18n/LanguageContext.jsx';

const ResultsScreen = ({ model, info, file, onReset }) => {
  const t = useT();
  const [lang] = useLang();
  const [gv, setGv] = useState(0);
  const isClass = model.type === 'classification';
  const r = model.sim;

  useEffect(() => {
    const t = setTimeout(() => setGv(isClass ? r.epilepsy : r.dsc), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const genTxt = () => {
    const locale = lang === 'en' ? 'en-US' : 'es-CO';
    const now = new Date().toLocaleString(locale);
    const sep = '═══════════════════════════════════════════════';
    const lines = [
      sep,
      `   ${t('report.header')}`,
      sep,
      '',
      `${t('report.date').padEnd(14)} ${now}`,
      `${t('report.model').padEnd(14)} ${t(`models.${model.id}.fullName`)}`,
      `${t('report.test').padEnd(14)} ${info.test}`,
      `${t('report.patient').padEnd(14)} ${info.patient || t('report.notSpecified')}`,
      `${t('report.file').padEnd(14)} ${file.name}`,
      '',
      t('report.sectionResult'),
      isClass
        ? [
            `${t('report.class').padEnd(14)} ${r.epilepsy >= 50 ? t('report.classEpi') : t('report.classControl')}`,
            `${t('report.probEpi').padEnd(18)} ${r.epilepsy} %`,
            `${t('report.probControl').padEnd(18)} ${r.control} %`,
          ].join('\n')
        : [
            `${t('report.type').padEnd(14)} ${t('report.typeSeg')}`,
            `${t('report.dscObtained').padEnd(14)} ${r.dsc} %`,
            `${t('report.mask').padEnd(14)} ${t('report.maskAvail')}`,
          ].join('\n'),
      '',
      t('report.sectionMetrics'),
      ...model.metrics.map(([k, v]) => `${t(`metrics.${k}`).padEnd(18)}${v}`),
      '',
      t('report.sectionWarn'),
      t('report.warn1'),
      t('report.warn2'),
      info.notes ? `\n${t('report.notes')} ${info.notes}` : '',
      '',
      sep,
      t('report.footerLine'),
    ]
      .filter((l) => l !== undefined)
      .join('\n');

    const blob = new Blob([lines], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `VBM_${lang === 'en' ? 'Report' : 'Reporte'}_${info.test.replace(/\s+/g, '_')}.txt`;
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
            <div style={{ fontWeight: 700, fontSize: 19 }}>{t('results.title')}</div>
            <div style={{ fontSize: 13.5, color: 'var(--t2)' }}>
              {t(`models.${model.id}.fullName`)} · <strong>{info.test}</strong>
              {info.patient ? ` · ${info.patient}` : ''}
            </div>
          </div>
        </div>
        <button className="btn btn-g btn-sm" onClick={onReset}>{t('results.back')}</button>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        {isClass ? (
          <div className="rgrid" style={{ display: 'grid', gridTemplateColumns: '230px 1fr', gap: 28, alignItems: 'start' }}>
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
                {t('results.confidence')}
              </div>
              <Gauge value={gv} color={isEpi ? 'var(--bad)' : 'var(--ok)'} />
              <div className={`verdict ${isEpi ? 'v-pos' : 'v-neg'}`}>
                {isEpi ? t('results.verdictEpi') : t('results.verdictControl')}
              </div>
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 17, marginBottom: 16 }}>{t('results.probsTitle')}</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div className="pbrow">
                  <div className="pblab">
                    <span>{t('results.labelEpilepsy')}</span>
                    <span style={{ fontWeight: 700, color: 'var(--bad)' }}>{gv} %</span>
                  </div>
                  <div className="pbt">
                    <div className="pbf" style={{ width: `${gv}%`, background: 'var(--bad)' }} />
                  </div>
                </div>
                <div className="pbrow">
                  <div className="pblab">
                    <span>{t('results.labelControl')}</span>
                    <span style={{ fontWeight: 700, color: 'var(--ok)' }}>{100 - gv} %</span>
                  </div>
                  <div className="pbt">
                    <div className="pbf" style={{ width: `${100 - gv}%`, background: 'var(--ok)' }} />
                  </div>
                </div>
              </div>
              <div className="mtiles">
                {model.metrics.map(([k, v]) => (
                  <div key={k} className="mtile">
                    <div className="mv">{v}</div>
                    <div className="ml">{t(`metrics.${k}`)}</div>
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
                  {t('results.segCaption1')}
                  <br />
                  {t('results.segCaption2')}
                </span>
              </div>
              <button
                className="btn btn-ok btn-sm"
                onClick={() => alert(t('results.downloadMaskAlertShort'))}
              >
                {t('results.downloadMaskShort')}
              </button>
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 17, marginBottom: 12 }}>{t('results.segCompletedTitle')}</div>
              <p
                style={{ fontSize: 14.5, color: 'var(--t2)', lineHeight: 1.65, marginBottom: 20 }}
                dangerouslySetInnerHTML={{
                  __html: t('results.segCompletedDesc', { nii: '<strong>.nii</strong>', gz: '<strong>.gz</strong>' }),
                }}
              />
              <div className="mtiles" style={{ gridTemplateColumns: 'repeat(2,1fr)' }}>
                {model.metrics.map(([k, v]) => (
                  <div key={k} className="mtile">
                    <div className="mv">{v}</div>
                    <div className="ml">{t(`metrics.${k}`)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="wb" style={{ marginBottom: 20 }}>
        <span style={{ fontSize: 22, flexShrink: 0 }}>⚕️</span>
        <div>
          <strong>{t('results.disclaimerBold')}</strong> {t('results.disclaimerBody')}{' '}
          <strong>{t('results.disclaimerEm')}</strong>
          {t('results.disclaimerRest')}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end', flexWrap: 'wrap' }}>
        {model.type === 'segmentation' && (
          <button
            className="btn btn-ok"
            onClick={() => alert(t('results.downloadMaskAlertLong'))}
          >
            {t('results.downloadMaskLong')}
          </button>
        )}
        <button className="btn btn-g" onClick={genTxt}>{t('results.exportTxt')}</button>
        <button className="btn btn-p" onClick={onReset}>{t('results.newAnalysis')}</button>
      </div>
    </div>
  );
};

export default ResultsScreen;
