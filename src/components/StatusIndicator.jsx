import { CheckCircle2, Clock3, ScanLine } from 'lucide-react';

export default function StatusIndicator({ status, error }) {
  const done = status === 'complete';
  const processing = status === 'queued' || status === 'processing';
  const failed = status === 'failed';
  return <div className="status-card"><div className={`status-icon ${done ? 'done' : processing ? 'processing' : ''}`}>{done ? <CheckCircle2 size={20} /> : processing ? <ScanLine size={20} /> : <Clock3 size={20} />}</div><div><span className="eyebrow">PIPELINE STATUS</span><strong>{done ? 'Analysis complete' : failed ? 'Analysis failed' : processing ? 'Processing sonar data' : 'Awaiting sonar file'}</strong><small>{done ? 'Ready for review and export' : failed ? error : processing ? (status === 'queued' ? 'Waiting for the backend worker' : 'Backend preprocessing in progress') : 'Upload a file to begin analysis'}</small></div><span className={`status-pill ${done ? 'success' : processing || failed ? 'warning' : ''}`}>{done ? 'READY' : failed ? 'FAILED' : processing ? status.toUpperCase() : 'IDLE'}</span></div>;
}
