import { useState, useMemo } from "react";
import {
  ComposedChart, Area, ReferenceLine, XAxis, YAxis,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { GaussBoneData } from "../types";

const STAGE_COLORS: Record<string, string> = {
  A:"#9E9E9E", B:"#2196F3", C:"#4CAF50", D:"#8BC34A",
  E:"#FFC107", F:"#FF9800", G:"#F44336", H:"#9C27B0", I:"#673AB7",
};

function gaussian(x: number, mean: number, sd: number): number {
  const z = (x - mean) / sd;
  return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI));
}

function buildCurveData(boneData: GaussBoneData) {
  const ages = Array.from({ length: 241 }, (_, i) => i); // 0–240 months
  return ages.map((age) => {
    const point: Record<string, number> = { age };
    for (const s of boneData.stages) {
      point[s.stage] = gaussian(age, s.mean, s.sd);
    }
    return point;
  });
}

export function GaussCurves({
  gaussianData,
  bones,
}: {
  gaussianData: Record<string, GaussBoneData>;
  bones: string[];
}) {
  const [selectedBone, setSelectedBone] = useState<string>(bones[0] ?? "radius");
  const boneData = gaussianData[selectedBone];

  const curveData = useMemo(
    () => (boneData ? buildCurveData(boneData) : []),
    [boneData]
  );

  if (!boneData) return null;

  const detectedStage = boneData.detected_stage;
  const chronoAge = boneData.chrono_age_months;

  return (
    <div className="space-y-3">
      {/* Bone selector */}
      <div className="flex flex-wrap gap-1">
        {bones.map((bone) => (
          <button
            key={bone}
            onClick={() => setSelectedBone(bone)}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              bone === selectedBone
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700"
            }`}
          >
            {gaussianData[bone]?.label ?? bone}
          </button>
        ))}
      </div>

      {/* Title */}
      <div className="text-sm text-gray-300">
        <span className="font-semibold">{boneData.label}</span>
        {" — Estadio detectado: "}
        <span
          className="font-bold"
          style={{ color: STAGE_COLORS[detectedStage] ?? "#fff" }}
        >
          {detectedStage}
        </span>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={curveData} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
          <XAxis
            dataKey="age"
            type="number"
            domain={[0, 240]}
            tickFormatter={(v) => `${v}m`}
            tick={{ fontSize: 10, fill: "#9CA3AF" }}
          />
          <YAxis tick={{ fontSize: 10, fill: "#9CA3AF" }} />
          <Tooltip
            formatter={(v: number, name: string) => [v.toFixed(4), `Estadio ${name}`]}
            labelFormatter={(l) => `Edad: ${l} meses`}
            contentStyle={{ background: "#1F2937", border: "none", fontSize: 11 }}
          />
          {boneData.stages.map((s) => (
            <Area
              key={s.stage}
              type="monotone"
              dataKey={s.stage}
              stroke={STAGE_COLORS[s.stage] ?? "#888"}
              fill={STAGE_COLORS[s.stage] ?? "#888"}
              fillOpacity={s.stage === detectedStage ? 0.35 : 0.08}
              strokeWidth={s.stage === detectedStage ? 2.5 : 1}
              dot={false}
              name={s.stage}
            />
          ))}
          {chronoAge !== null && (
            <ReferenceLine
              x={chronoAge}
              stroke="#F87171"
              strokeDasharray="4 3"
              label={{ value: "Edad cronológica", position: "top", fontSize: 10, fill: "#F87171" }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-500">
        Curvas de distribución Gaussian por estadio TW2 · Área resaltada = estadio detectado · Línea roja = edad cronológica
      </p>
    </div>
  );
}
