(() => {
  const host = document.getElementById("slice-host");
  if (!host) return;

  const runId = host.getAttribute("data-run-id");
  const dir = host.getAttribute("data-overlay-dir");
  const total = Number(host.getAttribute("data-slices") || "0") || 0;
  const baseUrl = host.getAttribute("data-base-url");
  const urlTemplate = host.getAttribute("data-url-template");

  const img = document.getElementById("slice-img");
  const slider = document.getElementById("slice-slider");
  const label = document.getElementById("slice-label");

  if (!runId || !dir || (!baseUrl && !urlTemplate) || !img || !slider || !label || total <= 0) return;

  const pad4 = (n) => String(n).padStart(4, "0");
  const makeUrl = (rel) => {
    if (urlTemplate) {
      return urlTemplate.replace("__REL__", rel);
    }
    return `${baseUrl}/${encodeURIComponent(runId)}/${rel}`;
  };
  const setSlice = (z) => {
    const zi = Math.max(0, Math.min(total - 1, Number(z) || 0));
    const rel = `${dir}/${pad4(zi)}.png`;
    img.src = makeUrl(rel);
    label.textContent = `Slice: ${zi} / ${total - 1}`;
    slider.value = String(zi);
  };

  slider.min = "0";
  slider.max = String(total - 1);
  slider.step = "1";

  slider.addEventListener("input", (e) => setSlice(e.target.value));

  setSlice(Math.floor(total / 2));
})();

