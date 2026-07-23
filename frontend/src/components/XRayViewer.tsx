export function XRayViewer({ imageB64 }: { imageB64: string }) {
  return (
    <div className="relative w-full">
      <img
        src={`data:image/png;base64,${imageB64}`}
        alt="Radiografía anotada"
        className="w-full rounded-lg border border-gray-700"
        style={{ imageRendering: "pixelated" }}
      />
      <p className="text-xs text-gray-500 mt-1 text-center">
        Cada recuadro corresponde a un hueso TW2 — color indica estadio de maduración
      </p>
    </div>
  );
}
