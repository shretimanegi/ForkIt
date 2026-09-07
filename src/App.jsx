import { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import UploadBox from './components/UploadBox';
import StatusIndicator from './components/StatusIndicator';
import SonarViewer from './components/SonarViewer';
import DetectionTable from './components/DetectionTable';
import DetectionDetails from './components/DetectionDetails';
import MapView from './components/MapView';
import DownloadButtons from './components/DownloadButtons';
import { getJobStatus, getResults, getWaterfall, uploadFile } from './api/backend';

export default function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle');
  const [jobId, setJobId] = useState(null);
  const [detections, setDetections] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [waterfallUrl, setWaterfallUrl] = useState(null);
  const [error, setError] = useState(null);
  const selected = detections.find((detection) => detection.id === selectedId);

  useEffect(() => {
    if (!jobId) return undefined;
    let cancelled = false;
    let polling = false;
    let timeoutId;

    const pollStatus = async () => {
      if (polling || cancelled) return;
      polling = true;
      try {
        const job = await getJobStatus(jobId);
        if (cancelled) return;
        if (job.status === 'failed') {
          setStatus('failed');
          setError(job.error || 'The backend could not process this file.');
        } else if (job.status === 'completed') {
          const results = await getResults(jobId);
          if (cancelled) return;
          setDetections(results);
          setSelectedId(results[0]?.id ?? null);
          setWaterfallUrl(getWaterfall(jobId));
          setStatus('complete');
        } else {
          setStatus(job.status);
          timeoutId = window.setTimeout(pollStatus, 1500);
        }
      } catch (pollError) {
        if (!cancelled) {
          setStatus('failed');
          setError(pollError.message);
        }
      } finally {
        polling = false;
      }
    };

    pollStatus();
    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [jobId]);

  const processFile = async () => {
    setError(null);
    setJobId(null);
    setDetections([]);
    setSelectedId(null);
    setWaterfallUrl(null);
    setStatus('queued');
    try {
      const response = await uploadFile(file);
      setJobId(response.job_id);
      setStatus(response.status || 'queued');
    } catch (uploadError) {
      setStatus('failed');
      setError(uploadError.message);
    }
  };
  const reset = () => { setFile(null); setJobId(null); setDetections([]); setSelectedId(null); setWaterfallUrl(null); setError(null); setStatus('idle'); };
  const select = (id) => setSelectedId(id);
  return <><Navbar /><main className="dashboard"><div className="hero"><div><p className="eyebrow">SIH 2026 · PAIR 3 / B3</p><h1>Sonar debris <em>detection</em></h1><p className="hero-subtitle">Review, validate, and export underwater anomalies from your side-scan survey.</p></div><div className="hero-stat"><strong>{detections.length}</strong><span>detections<br />in current run</span></div></div><div className="top-grid"><UploadBox file={file} status={status} error={error} onFile={(value) => { setFile(value); setStatus('idle'); setError(null); }} onProcess={processFile} onReset={reset} /><StatusIndicator status={status} error={error} /></div><div className="workspace-grid"><SonarViewer detections={detections} selectedId={selectedId} onSelect={select} waterfallUrl={waterfallUrl} onWaterfallError={() => setWaterfallUrl(null)} /><MapView detections={detections} selectedId={selectedId} onSelect={select} /></div><div className="lower-grid"><DetectionTable detections={detections} selectedId={selectedId} onSelect={select} /><DetectionDetails detection={selected} /></div><DownloadButtons detections={detections} /><footer><span>SONARWATCH · B3 FRONTEND PROTOTYPE</span><span>Backend-connected workspace</span></footer></main></>;
}
