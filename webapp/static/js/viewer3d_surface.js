(() => {
  const payload = document.getElementById("mesh-data");
  const target = document.getElementById("viewer3d-surface");
  if (!payload || !target) return;

  const result = JSON.parse(payload.textContent);
  const traces = [];

  const ctLighting = {
    ambient: 0.42,
    diffuse: 0.92,
    specular: 0.35,
    roughness: 0.45,
    fresnel: 0.12,
  };

  const lesionLighting = {
    ambient: 0.55,
    diffuse: 0.88,
    specular: 0.25,
    roughness: 0.35,
    fresnel: 0.08,
  };

  if (result.hu_mesh) {
    traces.push({
      type: "mesh3d",
      x: result.hu_mesh.x,
      y: result.hu_mesh.y,
      z: result.hu_mesh.z,
      i: result.hu_mesh.i,
      j: result.hu_mesh.j,
      k: result.hu_mesh.k,
      name: "Permukaan CT",
      color: "#e6e8ec",
      opacity: 0.92,
      flatshading: true,
      lighting: ctLighting,
      lightposition: { x: 200, y: -400, z: 200 },
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
      name: "Mask lesi",
      color: "#ea580c",
      opacity: 1,
      flatshading: true,
      lighting: lesionLighting,
      lightposition: { x: 200, y: -200, z: 400 },
    });
  }

  if (traces.length === 0) {
    target.innerHTML = "<p>Tidak ada data permukaan 3D (jalankan inferensi dengan volume yang valid).</p>";
    return;
  }

  const layout = {
    paper_bgcolor: "#000000",
    plot_bgcolor: "#000000",
    font: { color: "#cbd5e1" },
    scene: {
      bgcolor: "#000000",
      xaxis: {
        title: "X (mm)",
        showbackground: false,
        showgrid: true,
        gridcolor: "rgba(148,163,184,0.25)",
        zerolinecolor: "rgba(148,163,184,0.35)",
        color: "#94a3b8",
      },
      yaxis: {
        title: "Y (mm)",
        showbackground: false,
        showgrid: true,
        gridcolor: "rgba(148,163,184,0.25)",
        zerolinecolor: "rgba(148,163,184,0.35)",
        color: "#94a3b8",
      },
      zaxis: {
        title: "Z (mm)",
        showbackground: false,
        showgrid: true,
        gridcolor: "rgba(148,163,184,0.25)",
        zerolinecolor: "rgba(148,163,184,0.35)",
        color: "#94a3b8",
      },
      aspectmode: "data",
      camera: {
        eye: { x: 1.55, y: 1.45, z: 1.2 },
        center: { x: 0, y: 0, z: 0 },
        up: { x: 0, y: 0, z: 1 },
      },
    },
    margin: { l: 0, r: 0, t: 28, b: 0 },
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: "rgba(0,0,0,0.5)",
      bordercolor: "rgba(148,163,184,0.4)",
      font: { color: "#e2e8f0" },
    },
    title: {
      text: "Render permukaan dari DICOM + mask (marching cubes)",
      font: { size: 14, color: "#94a3b8" },
      y: 0.98,
      x: 0.5,
      xanchor: "center",
    },
  };

  Plotly.newPlot(target, traces, layout, { responsive: true, displaylogo: false });
})();
