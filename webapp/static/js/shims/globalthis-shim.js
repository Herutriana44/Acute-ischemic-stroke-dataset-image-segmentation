/** Minimal ESM shim: vtk.js imports `globalthis` as default export. */
export default function getGlobalThis() {
  return globalThis;
}
