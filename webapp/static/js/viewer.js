(() => {
  const payload = document.getElementById("mesh-data");
  const target = document.getElementById("viewer3d");
  if (!payload || !target) return;

  const result = JSON.parse(payload.textContent);
  const traces = [];

  if (result.hu_mesh) {
    traces.push({
      type: "mesh3d",
      x: result.hu_mesh.x,
      y: result.hu_mesh.y,
      z: result.hu_mesh.z,
      i: result.hu_mesh.i,
      j: result.hu_mesh.j,
      k: result.hu_mesh.k,
      opacity: 0.15,
      color: "#8f9baa",
      name: "CT Volume",
      flatshading: false,
    });
  }

  if (result.lesion_mesh) {
    traces.push({
      type: "mesh3d",
      x: result.lesion_mesh.x,
      y: result.lesion_mesh.y,
      z: result.lesion_mesh.z,
      i: result.lesion_mesh.i,
      j: result.lesion_mesh.j,
      k: result.lesion_mesh.k,
      opacity: 0.7,
      color: "#ef4444",
      name: "Lesion Prediction",
      flatshading: true,
    });
  }

  if (traces.length === 0) {
    target.innerHTML = "<p>Tidak ada mesh yang dapat divisualisasikan.</p>";
    return;
  }

  const layout = {
    paper_bgcolor: "#101828",
    plot_bgcolor: "#101828",
    font: { color: "#e2e8f0" },
    scene: {
      xaxis: { title: "X (mm)", gridcolor: "#334155" },
      yaxis: { title: "Y (mm)", gridcolor: "#334155" },
      zaxis: { title: "Z (mm)", gridcolor: "#334155" },
      aspectmode: "data",
      camera: { eye: { x: 1.4, y: 1.4, z: 1.1 } },
    },
    margin: { l: 0, r: 0, t: 20, b: 0 },
    legend: { x: 0.02, y: 0.98 },
  };

  Plotly.newPlot(target, traces, layout, { responsive: true, displaylogo: false });
})();
