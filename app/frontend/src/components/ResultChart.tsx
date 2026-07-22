const CLASS_COLORS: Record<string, string> = {
  Gold: "#d4af37",
  Pyrite: "#b8860b",
  Other: "#2a78d6",
};
const FALLBACK_COLOR = "#e34948";
const DISPLAY_ORDER = ["Gold", "Pyrite", "Other"];

interface ResultChartProps {
  classes: string[];
  probabilities: Record<string, number>;
}

export default function ResultChart({ classes, probabilities }: ResultChartProps) {
  const orderedClasses = [...classes].sort(
    (a, b) => DISPLAY_ORDER.indexOf(a) - DISPLAY_ORDER.indexOf(b)
  );

  return (
    <div className="chart" role="img" aria-label="Class confidence chart">
      {orderedClasses.map((cls) => {
        const pct = probabilities[cls] * 100;
        return (
          <div className="chart-row" key={cls}>
            <span className="chart-label">{cls}</span>
            <div className="chart-track">
              <div
                className="chart-bar"
                style={{ width: `${pct}%`, background: CLASS_COLORS[cls] ?? FALLBACK_COLOR }}
              />
            </div>
            <span className="chart-value">{pct.toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}
