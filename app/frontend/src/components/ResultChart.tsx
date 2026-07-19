const SERIES_COLORS = ["#2a78d6", "#008300", "#eda100", "#e34948"];

interface ResultChartProps {
  classes: string[];
  probabilities: Record<string, number>;
}

export default function ResultChart({ classes, probabilities }: ResultChartProps) {
  return (
    <div className="chart" role="img" aria-label="Class confidence chart">
      {classes.map((cls, i) => {
        const pct = probabilities[cls] * 100;
        return (
          <div className="chart-row" key={cls}>
            <span className="chart-label">{cls}</span>
            <div className="chart-track">
              <div
                className="chart-bar"
                style={{ width: `${pct}%`, background: SERIES_COLORS[i % SERIES_COLORS.length] }}
              />
            </div>
            <span className="chart-value">{pct.toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}
