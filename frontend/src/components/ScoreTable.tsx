import { AnalysisResult } from "../types";

const BONE_LABELS: Record<string, string> = {
  radius: "Radio", ulna: "Cúbito",
  mc1: "MC I", mc3: "MC III", mc5: "MC V",
  pp1: "FF I", pp3: "FF III", pp5: "FF V",
  mp3: "FM III", mp5: "FM V",
  dp1: "FD I", dp3: "FD III", dp5: "FD V",
  capitate: "Grande", hamate: "Ganchoso", triquetral: "Piramidal",
  lunate: "Semilunar", scaphoid: "Escafoides",
  trapezoid: "Trapezoides", trapezium: "Trapecio",
};

const STAGE_COLORS: Record<string, string> = {
  A: "bg-gray-500", B: "bg-blue-500", C: "bg-green-600", D: "bg-lime-500",
  E: "bg-yellow-500", F: "bg-orange-500", G: "bg-red-500", H: "bg-purple-500", I: "bg-violet-700",
};

const RUS_BONES = ["radius","ulna","mc1","mc3","mc5","pp1","pp3","pp5","mp3","mp5","dp1","dp3","dp5"];

export function ScoreTable({ result }: { result: AnalysisResult }) {
  const rusBones = Object.entries(result.bone_scores).filter(([b]) => RUS_BONES.includes(b));
  const carpalBones = Object.entries(result.bone_scores).filter(([b]) => !RUS_BONES.includes(b));

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-blue-900/40 rounded-xl p-3 text-center">
          <div className="text-2xl font-bold text-blue-300">{result.bone_age_years} años</div>
          <div className="text-sm text-gray-400">{result.bone_age_months} meses</div>
          <div className="text-xs text-gray-500 mt-1">IC 90%: {result.confidence_interval[0]}–{result.confidence_interval[1]} m</div>
        </div>
        <div className="bg-gray-800 rounded-xl p-3 space-y-1 text-sm">
          <div className="flex justify-between"><span className="text-gray-400">Score RUS</span><span className="font-mono text-white">{result.rus_score}</span></div>
          <div className="flex justify-between"><span className="text-gray-400">Edad RUS</span><span className="font-mono text-white">{result.rus_age_months} m</span></div>
          <div className="flex justify-between"><span className="text-gray-400">Score Carpal</span><span className="font-mono text-white">{result.carpal_score}</span></div>
          <div className="flex justify-between"><span className="text-gray-400">Edad Carpal</span><span className="font-mono text-white">{result.carpal_age_months} m</span></div>
        </div>
      </div>

      {/* RUS bones */}
      <BoneGroup title="Huesos RUS (13)" bones={rusBones} result={result} />
      <BoneGroup title="Carpianos (7)" bones={carpalBones} result={result} />
    </div>
  );
}

function BoneGroup({
  title, bones, result,
}: { title: string; bones: [string, number][]; result: AnalysisResult }) {
  return (
    <div>
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-1">{title}</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-500 text-xs">
            <th className="text-left py-0.5">Hueso</th>
            <th className="text-center">Estadio</th>
            <th className="text-right">Score</th>
          </tr>
        </thead>
        <tbody>
          {bones.map(([bone, score]) => {
            const stage = result.stages[bone] ?? "?";
            return (
              <tr key={bone} className="border-t border-gray-800">
                <td className="py-0.5 text-gray-300">{BONE_LABELS[bone] ?? bone}</td>
                <td className="text-center">
                  <span className={`inline-block px-2 rounded text-xs font-bold text-white ${STAGE_COLORS[stage] ?? "bg-gray-600"}`}>
                    {stage}
                  </span>
                </td>
                <td className="text-right font-mono text-gray-300">{score}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
