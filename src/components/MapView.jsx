import { useEffect } from 'react';
import { MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet';
import L from 'leaflet';
import { LocateFixed } from 'lucide-react';

const icon = (selected) => L.divIcon({ className: `map-marker ${selected ? 'selected' : ''}`, html: `<span>${selected ? '★' : '•'}</span>`, iconSize: [30, 30], iconAnchor: [15, 15] });
function Recenter({ detection }) { const map = useMap(); useEffect(() => { if (detection) map.flyTo([detection.lat, detection.lon], 16, { duration: 0.7 }); }, [detection, map]); return null; }
export default function MapView({ detections, selectedId, onSelect }) {
  const selected = detections.find((item) => item.id === selectedId);
  return <section className="card map-card"><div className="section-heading"><div><p className="eyebrow">GEOSPATIAL OVERVIEW</p><h2>Detection map</h2></div><span className="map-status"><span className="live-dot" /> LIVE VIEW</span></div><div className="map-wrap"><MapContainer center={[28.6139, 77.209]} zoom={14} scrollWheelZoom className="leaflet-map"><TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" /><Recenter detection={selected} />{detections.map((detection) => <Marker key={detection.id} position={[detection.lat, detection.lon]} icon={icon(selectedId === detection.id)} eventHandlers={{ click: () => onSelect(detection.id) }}><Popup><strong>#{String(detection.id).padStart(2, '0')} · {detection.class.replace('_', ' ')}</strong><br />Confidence: {Math.round(detection.confidence * 100)}%</Popup></Marker>)}</MapContainer><div className="map-overlay"><LocateFixed size={14} /> {selected ? `Centered on detection #${String(selected.id).padStart(2, '0')}` : 'Select a marker to center'}</div></div></section>;
}
