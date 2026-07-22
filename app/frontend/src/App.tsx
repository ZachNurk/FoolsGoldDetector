import { useState } from "react";
import { predict, type PredictResponse } from "./api";
import UploadBox from "./components/UploadBox";
import ResultChart from "./components/ResultChart";

export default function App() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleFile(file: File) {
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const res = await predict(file);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-root">
      <h1>Fools Gold Detector</h1>
      <p>Upload a photo of a mineral specimen to classify it as Gold, Fools Gold (Pyrite), or Other.</p>

      <UploadBox onFile={handleFile} />

      {loading && <p className="status-text">Classifying...</p>}
      {error && <p className="status-text error-text">{error}</p>}

      {previewUrl && result && (
        <div className="result-row">
          <img src={previewUrl} alt="Uploaded specimen" className="preview-img" />
          <div className="result-chart">
            <h2>Prediction: {result.prediction}</h2>
            <ResultChart classes={result.classes} probabilities={result.probabilities} />
          </div>
        </div>
      )}
    </div>
  );
}
