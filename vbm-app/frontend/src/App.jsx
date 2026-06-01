import { useState } from 'react';
import Brain from './components/Brain.jsx';
import HomeScreen from './screens/HomeScreen.jsx';
import UploadScreen from './screens/UploadScreen.jsx';
import ProcessingScreen from './screens/ProcessingScreen.jsx';
import ResultsScreen from './screens/ResultsScreen.jsx';

const App = () => {
  const [screen, setScreen] = useState('home');
  const [model, setModel] = useState(null);
  const [info, setInfo] = useState({});
  const [file, setFile] = useState(null);

  const reset = () => {
    setModel(null);
    setInfo({});
    setFile(null);
    setScreen('home');
  };

  return (
    <div>
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
          v1.0 prototipo
        </span>
        {model && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className={`badge ${model.badge === 'Segmentación' ? 'b-am' : 'b-bl'}`}>{model.badge}</span>
            <span style={{ fontSize: 13.5, color: 'var(--t2)', fontWeight: 500 }}>{model.name}</span>
          </div>
        )}
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
            onStart={(i, f) => {
              setInfo(i);
              setFile(f);
              setScreen('processing');
            }}
            onBack={() => setScreen('home')}
          />
        )}
        {screen === 'processing' && (
          <ProcessingScreen
            model={model}
            info={info}
            file={file}
            onCancel={reset}
            onDone={() => setScreen('results')}
          />
        )}
        {screen === 'results' && <ResultsScreen model={model} info={info} file={file} onReset={reset} />}
      </main>
    </div>
  );
};

export default App;
