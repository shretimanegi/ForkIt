import { Download, FileJson, FileSpreadsheet, Globe2 } from 'lucide-react';
import { downloadCsv, downloadGeoJson, downloadJson } from '../utils/downloadReports';
export default function DownloadButtons({ detections }) {
  return <section className="card download-card"><div><p className="eyebrow">EXPORT REPORTS</p><h2>Download results</h2><span>Export the current local detection set</span></div><div className="download-actions"><button onClick={() => downloadJson(detections)}><FileJson size={16} /> JSON</button><button onClick={() => downloadCsv(detections)}><FileSpreadsheet size={16} /> CSV</button><button onClick={() => downloadGeoJson(detections)}><Globe2 size={16} /> GeoJSON</button><span className="download-icon"><Download size={17} /></span></div></section>;
}
