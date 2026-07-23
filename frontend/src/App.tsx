import { useState, useRef } from "react";
import { AnalysisResult, Sex } from "./types";
import { XRayViewer } from "./components/XRayViewer";
import { ScoreTable } from "./components/ScoreTable";
import { GaussCurves } from "./components/GaussCurves";

type Tab = "imagen" | "scores" | "curvas";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [sex, setSex] = useState<Sex>("M");
  const [chronoAge, setChronoAge] = useState<string>("");
  const [scaleFactor, setScaleFactor] = useState<string>("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("imagen");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const fd = new FormData();
    fd.append("image", file);
    fd.append("sex", sex);
    if (chronoAge) fd.append("chronological_age_months", chronoAge);
    if (scaleFactor) fd.append("scale_factor", scaleFactor);

    try {
      const resp = await fetch("/analyze", { method: "POST", body: fd });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `Error ${resp.status}`);
      }
      const data: AnalysisResult = await resp.json();
      setResult(data);
      setTab("imagen");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (!file) return;
    const fd = new FormData();
    fd.append("image", file);
    fd.append("sex", sex);
    if (chronoAge) fd.append("chronological_age_months", chronoAge);
    if (scaleFactor) fd.append("scale_factor", scaleFactor);
    const resp = await fetch("/analyze/pdf", { method: "POST", body: fd });
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "bone_age_report.pdf";
    a.click();
    URL.revokeObjectURL(url);
  };

  const bones = result ? Object.keys(result.gaussian_data) : [];

  return (
    <div className="min-h-screen flex flex-col items-center py-8 px-4">
      <h1 className="text-2xl font-bold text-blue-300 mb-1">BoneAge TW2</h1>
      <p className="text-gray-500 text-sm mb-6">Maduración ósea — Método Tanner-Whitehouse 2 · 20 huesos</p>

      {/* Upload panel */}
      <div className="w-full max-w-2xl bg-gray-900 rounded-2xl p-5 mb-6 space-y-4 border border-gray-800">
        <div
          className="border-2 border-dashed border-gray-700 rounded-xl p-6 text-center cursor-pointer hover:border-blue-500 transition-colors"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files[0];
            if (f) setFile(f);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".dcm,.png,.jpg,.jpeg"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          {file ? (
            <p className="text-green-400 font-medium">{file.name}</p>
          ) : (
            <p className="text-gray-500">
              Arrastra una radiografía o haz clic<br />
              <span className="text-xs">.dcm · .png · .jpg</span>
            </p>
          )}
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Sexo</label>
            <select
              value={sex}
              onChange={(e) => setSex(e.target.value as Sex)}
              className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm border border-gray-700 focus:outline-none focus:border-blue-500"
            >
              <option value="M">Masculino</option>
              <option value="F">Femenino</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Edad cronológica (meses)</label>
            <input
              type="number"
              value={chronoAge}
              onChange={(e) => setChronoAge(e.target.value)}
              placeholder="opcional"
              className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm border border-gray-700 focus:outline-none focus:border-blue-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Factor escala (mm/px)</label>
            <input
              type="number"
              step="0.001"
              value={scaleFactor}
              onChange={(e) => setScaleFactor(e.target.value)}
              placeholder="auto-DICOM"
              className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm border border-gray-700 focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>

        <button
          onClick={handleAnalyze}
          disabled={!file || loading}
          className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold rounded-xl py-2.5 transition-colors"
        >
          {loading ? "Analizando..." : "Analizar radiografía"}
        </button>

        {error && (
          <p className="text-red-400 text-sm bg-red-900/20 rounded-lg p-3">{error}</p>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="w-full max-w-5xl">
          {/* Tab bar */}
          <div className="flex gap-2 mb-4">
            {(["imagen", "scores", "curvas"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  tab === t ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                }`}
              >
                {t === "imagen" ? "Radiografía anotada" : t === "scores" ? "Tabla TW2" : "Curvas Gauss"}
              </button>
            ))}
            <button
              onClick={handleDownloadPdf}
              className="ml-auto px-4 py-1.5 rounded-lg text-sm bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors"
            >
              Descargar PDF
            </button>
          </div>

          {/* Tab content */}
          <div className="bg-gray-900 rounded-2xl p-5 border border-gray-800">
            {tab === "imagen" && <XRayViewer imageB64={result.annotated_image_b64} />}
            {tab === "scores" && <ScoreTable result={result} />}
            {tab === "curvas" && (
              <GaussCurves gaussianData={result.gaussian_data} bones={bones} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
