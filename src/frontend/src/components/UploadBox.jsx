import { useRef } from 'react';
import { FileUp, FileCheck, Play, RotateCcw } from 'lucide-react';

export default function UploadBox({ file, status, error, onFile, onProcess, onReset }) {
  const inputRef = useRef(null);
  const chooseFile = (event) => {
    const selected = event.target.files?.[0];
    if (selected) onFile(selected);
  };
  return (
    <section className="card upload-card">
      <div className="section-heading"><div><p className="eyebrow">INPUT SOURCE</p><h2>Sonar file</h2></div><span className="format-badge">XTF / SONAR</span></div>
      {!file ? <button className="drop-zone" onClick={() => inputRef.current?.click()}><FileUp size={25} /><strong>Drop an XTF file here</strong><span>or click to browse from your computer</span><small>Local prototype · files never leave this browser</small></button> : <div className="file-selected"><span className="file-icon"><FileCheck size={20} /></span><div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · ready to process</span></div><button className="ghost-button" onClick={onReset}><RotateCcw size={15} /> Change</button></div>}
      <input ref={inputRef} type="file" accept=".xtf,.son,.dat" onChange={chooseFile} hidden />
      {file && status === 'idle' && <button className="primary-button process-button" onClick={onProcess}><Play size={15} fill="currentColor" /> Process locally</button>}
      {(status === 'queued' || status === 'processing') && <div className="progress-wrap"><div className="progress-label"><span>{status === 'queued' ? 'Queued for analysis...' : 'Analyzing sonar pings...'}</span><span className="pulse">{status === 'queued' ? 'QUEUED' : 'IN PROGRESS'}</span></div><div className="progress-track"><div className="progress-bar" /></div></div>}
      {status === 'complete' && <div className="completed-message"><FileCheck size={16} /> Analysis complete · {file ? 'Results ready for review' : 'No file selected'}</div>}
      {error && <div className="completed-message" style={{ color: '#a46f2f' }}>{error}</div>}
    </section>
  );
}
