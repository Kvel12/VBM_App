import { useState } from 'react';
import Brain from './components/Brain.jsx';
import Footer from './components/Footer.jsx';
import LanguageToggle from './components/LanguageToggle.jsx';
import HomeScreen from './screens/HomeScreen.jsx';
import UploadScreen from './screens/UploadScreen.jsx';
import ProcessingScreen from './screens/ProcessingScreen.jsx';
import ResultsScreen from './screens/ResultsScreen.jsx';
import AboutScreen from './screens/AboutScreen.jsx';
import { useT } from './i18n/LanguageContext.jsx';

const App = () => {
  const t = useT();
  const [screen, setScreen] = useState('home');
  const [prevScreen, setPrevScreen] = useState('home');
  const [model, setModel] = useState(null);
  const [info, setInfo] = useState({});
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [result, setResult] = useState(null);

  const reset = () => {
    setModel(null);
    setInfo({});
    setFile(null);
    setJobId(null);
    setResult(null);
    setScreen('home');
  };

  const goAbout = () => {
    setPrevScreen(screen);
    setScreen('about');
  };

  const backFromAbout = () => setScreen(prevScreen);

  return (
    <div className="app">
      <nav className="nav">
        <div
          style={{
            cursor: screen !== 'processing' ? 'pointer' : 'default',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
          onClick={screen !== 'processing' ? reset : undefined}
        >
          <Brain size={32} color="var(--primary)" />
          <span className="nav-title">
            VBM <em>App</em>
          </span>
        </div>
        <span
          style={{
            fontSize: 12,
            color: 'var(--t3)',
            background: 'var(--s2)',
            padding: '2px 9px',
            borderRadius: 99,
            marginLeft: 4,
          }}
        >
          {t('nav.versionBadge')}
        </span>

        <div className="nav-right">
          {model && screen !== 'about' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className={`badge ${model.badge === 'segmentation' ? 'b-am' : 'b-bl'}`}>
                {t(`badge.${model.badge}`)}
              </span>
              <span style={{ fontSize: 13.5, color: 'var(--t2)', fontWeight: 500 }}>
                {t(`models.${model.id}.name`)}
              </span>
            </div>
          )}
          <button type="button" className="nav-about" onClick={goAbout}>
            {t('nav.about')}
          </button>
          <LanguageToggle />
        </div>
      </nav>

      <main className="main">
        {screen === 'home' && (
          <HomeScreen
            onSelect={(m) => {
              setModel(m);
              setScreen('upload');
            }}
          />
        )}
        {screen === 'upload' && (
          <UploadScreen
            model={model}
            onStart={(i, f, newJobId) => {
              setInfo(i);
              setFile(f);
              setJobId(newJobId);
              setResult(null);
              setScreen('processing');
            }}
            onBack={() => setScreen('home')}
          />
        )}
        {screen === 'processing' && (
          <ProcessingScreen
            model={model}
            info={info}
            jobId={jobId}
            onCancel={reset}
            onDone={(r) => {
              setResult(r);
              setScreen('results');
            }}
          />
        )}
        {screen === 'results' && (
          <ResultsScreen model={model} info={info} file={file} jobId={jobId} result={result} onReset={reset} />
        )}
        {screen === 'about' && <AboutScreen onBack={backFromAbout} />}
      </main>

      <Footer onAbout={goAbout} />
    </div>
  );
};

export default App;
