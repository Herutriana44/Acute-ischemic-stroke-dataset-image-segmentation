(() => {
  const root = document.querySelector(".mode-chooser");
  if (!root) return;

  const buttons = Array.from(root.querySelectorAll(".mode-btn"));
  const panels = Array.from(document.querySelectorAll(".mode-panel"));

  const setMode = (mode) => {
    buttons.forEach((b) => {
      const on = b.getAttribute("data-mode") === mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    panels.forEach((p) => {
      const on = p.getAttribute("data-panel") === mode;
      p.classList.toggle("hidden", !on);
    });
  };

  buttons.forEach((b) => {
    b.addEventListener("click", () => setMode(b.getAttribute("data-mode")));
  });

  // default
  const active = buttons.find((b) => b.classList.contains("active")) || buttons[0];
  if (active) setMode(active.getAttribute("data-mode"));
})();

