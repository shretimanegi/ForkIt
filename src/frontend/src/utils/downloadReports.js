const download = (content, filename, type) => {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};

export const downloadJson = (detections) =>
  download(JSON.stringify(detections, null, 2), 'sonar-detections.json', 'application/json');

export const downloadCsv = (detections) => {
  const headers = Object.keys(detections[0]);
  const rows = detections.map((detection) =>
    headers.map((header) => `"${Array.isArray(detection[header]) ? detection[header].join(';') : detection[header]}"`).join(','),
  );
  download([headers.join(','), ...rows].join('\n'), 'sonar-detections.csv', 'text/csv');
};

export const downloadGeoJson = (detections) => {
  const geoJson = {
    type: 'FeatureCollection',
    features: detections.map((detection) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [detection.lon, detection.lat] },
      properties: { ...detection },
    })),
  };
  download(JSON.stringify(geoJson, null, 2), 'sonar-detections.geojson', 'application/geo+json');
};
