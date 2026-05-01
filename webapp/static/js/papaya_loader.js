(() => {
  const host = document.getElementById("papaya-host");
  if (!host) return;

  const ctUrl = host.getAttribute("data-ct-url");
  const ctHuUrl = host.getAttribute("data-ct-hu-url");
  const maskUrl = host.getAttribute("data-mask-url");
  const runId = host.getAttribute("data-run-id") || "";

  const statusEl = document.getElementById("papaya-status");
  const barEl = document.getElementById("papaya-progress-bar");
  const pctEl = document.getElementById("papaya-progress-pct");
  const errEl = document.getElementById("papaya-error");

  const setStatus = (txt) => {
    if (statusEl) statusEl.textContent = txt;
  };

  const setPct = (pct) => {
    const clamped = Math.max(0, Math.min(100, pct));
    if (barEl) barEl.style.width = `${clamped}%`;
    if (pctEl) pctEl.textContent = `${Math.round(clamped)}%`;
  };

  const showError = (message) => {
    if (errEl) {
      errEl.style.display = "block";
      errEl.innerHTML = `
        <strong>Gagal memuat viewer</strong>
        <div style="margin-top:6px; opacity:0.9">${message}</div>
        <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap">
          <a class="button-link" href="${ctUrl}" target="_blank" rel="noreferrer">Download CT NIfTI</a>
          ${ctHuUrl ? `<a class="button-link" href="${ctHuUrl}" target="_blank" rel="noreferrer">Download CT HU (float)</a>` : ""}
          <a class="button-link" href="${maskUrl}" target="_blank" rel="noreferrer">Download Mask NIfTI</a>
        </div>
      `;
    }
    setStatus("Tidak bisa memuat data.");
  };

  async function fetchBlobWithProgress(url, onProgress) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status} saat mengambil ${url}`);

    const total = Number(res.headers.get("Content-Length") || "0") || 0;
    if (!res.body) {
      const blob = await res.blob();
      onProgress(blob.size, blob.size, total || blob.size);
      return blob;
    }

    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      onProgress(received, total);
    }

    return new Blob(chunks);
  }

  function startPapaya(ctBlobUrl, maskBlobUrl) {
    // Papaya expects a global params object.
    window.papayaParams = [];
    // worldSpace sering bikin tampilan "aneh/gelap" bila affine DICOM berbeda konvensi.
    // Untuk viewer web, off lebih aman.
    window.papayaParams["worldSpace"] = false;
    window.papayaParams["showOrientation"] = true;
    window.papayaParams["showRuler"] = true;
    window.papayaParams["smoothDisplay"] = false;
    window.papayaParams["interpolation"] = "none";
    window.papayaParams["images"] = [ctBlobUrl, maskBlobUrl];

    // CT ditampilkan sebagai uint8 0..255 (hasil windowing), jadi pasti terlihat.
    window.papayaParams[ctBlobUrl] = { min: 0, max: 255 };
    window.papayaParams[maskBlobUrl] = { min: 0, max: 1, alpha: 0.45, lut: "Spectrum" };

    host.innerHTML = `<div class="papaya" data-params="papayaParams"></div>`;

    if (window.papaya && window.papaya.Container && typeof window.papaya.Container.startPapaya === "function") {
      window.papaya.Container.startPapaya();
      return;
    }

    // Fallback: give Papaya a tick to initialize.
    setTimeout(() => {
      if (window.papaya && window.papaya.Container && typeof window.papaya.Container.startPapaya === "function") {
        window.papaya.Container.startPapaya();
      } else {
        showError("Papaya tidak terinisialisasi. Pastikan `papaya.js` termuat dengan benar.");
      }
    }, 0);
  }

  (async () => {
    if (!ctUrl || !maskUrl) {
      showError("URL data CT/mask tidak ditemukan.");
      return;
    }

    try {
      setStatus("Mengunduh CT (NIfTI)...");
      setPct(1);

      let ctReceived = 0;
      let ctTotal = 0;
      let maskReceived = 0;
      let maskTotal = 0;

      const updateOverall = () => {
        const aTotal = ctTotal || 0;
        const bTotal = maskTotal || 0;
        const knownTotal = aTotal + bTotal;

        if (knownTotal > 0) {
          setPct(((ctReceived + maskReceived) / knownTotal) * 100);
        } else {
          // Unknown sizes: simple staged progress.
          const stage = (ctReceived > 0 ? 50 : 0) + (maskReceived > 0 ? 50 : 0);
          setPct(Math.max(5, stage));
        }
      };

      const ctBlob = await fetchBlobWithProgress(ctUrl, (received, total) => {
        ctReceived = received;
        ctTotal = total || ctTotal;
        updateOverall();
      });

      setStatus("Mengunduh hasil prediksi (mask)...");
      const maskBlob = await fetchBlobWithProgress(maskUrl, (received, total) => {
        maskReceived = received;
        maskTotal = total || maskTotal;
        updateOverall();
      });

      setStatus("Menyiapkan viewer...");
      setPct(95);

      const ctBlobUrl = URL.createObjectURL(ctBlob);
      const maskBlobUrl = URL.createObjectURL(maskBlob);

      // Give the browser a moment to settle before heavy render.
      requestAnimationFrame(() => {
        startPapaya(ctBlobUrl, maskBlobUrl);
        setPct(100);
        setStatus(`Selesai (run ${runId}).`);
      });
    } catch (err) {
      console.error(err);
      showError(err?.message || String(err));
    }
  })();
})();

