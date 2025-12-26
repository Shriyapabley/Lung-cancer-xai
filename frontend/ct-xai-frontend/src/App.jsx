// src/App.jsx

import React, { useState } from "react";
import axios from "axios";

const API_BASE = "http://127.0.0.1:8000/api/v1";
const CLASS_LABELS = ["Normal", "Adenocarcinoma", "Squamous", "Large cell"];

function App() {
  const [file, setFile] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [explainLoading, setExplainLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [heatmapSrc, setHeatmapSrc] = useState(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("dashboard");

  const handleFileChange = (e) => {
    setFile(e.target.files[0] || null);
    setPrediction(null);
    setHeatmapSrc(null);
    setError("");
  };

  const uploadAndPredict = async () => {
    if (!file) {
      setError("Please select a CT image first.");
      return;
    }
    setPredictLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post(`${API_BASE}/predict_ct`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPrediction(res.data);
    } catch (err) {
      console.error(err);
      setError("Prediction request failed. Please verify backend is running.");
    } finally {
      setPredictLoading(false);
    }
  };

  const uploadAndExplain = async () => {
    if (!file) {
      setError("Please select a CT image first.");
      return;
    }
    setExplainLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post(`${API_BASE}/explain_ct`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPrediction({
        predicted_class: res.data.predicted_class,
        probabilities: res.data.probabilities,
      });

      const imgSrc = `data:image/png;base64,${res.data.heatmap_base64}`;
      setHeatmapSrc(imgSrc);
    } catch (err) {
      console.error(err);
      setError("Explainability request failed. Please verify backend is running.");
    } finally {
      setExplainLoading(false);
    }
  };

  const renderProbabilities = () => {
    if (!prediction) return null;
    const probs = prediction.probabilities || [];
    return probs.map((p, idx) => {
      const percentage = (p * 100).toFixed(2);
      const active = idx === prediction.predicted_class;
      return (
        <div key={idx} style={{ marginBottom: "0.6rem" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "0.85rem",
              marginBottom: "0.1rem",
            }}
          >
            <span style={{ fontWeight: 600 }}>
              {CLASS_LABELS[idx] || `Class ${idx}`}
            </span>
            <span>{percentage}%</span>
          </div>
          <div
            style={{
              height: "7px",
              width: "100%",
              background: "#020617",
              borderRadius: "999px",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${percentage}%`,
                background: active ? "#f97316" : "#38bdf8",
                boxShadow: active
                  ? "0 0 22px rgba(248,113,113,0.9)"
                  : "0 0 12px rgba(56,189,248,0.6)",
                transition: "width 0.4s ease",
              }}
            />
          </div>
        </div>
      );
    });
  };

  const LoadingPulse = ({ label }) => (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.45rem",
        fontSize: "0.85rem",
        opacity: 0.9,
      }}
    >
      <span
        style={{
          width: "10px",
          height: "10px",
          borderRadius: "999px",
          background:
            "radial-gradient(circle, #22c55e 0, #22c55e 40%, transparent 70%)",
          boxShadow: "0 0 14px rgba(34, 197, 94, 0.7)",
          animation: "pulse 1.2s infinite",
        }}
      />
      <span>{label}</span>
    </div>
  );

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100vw",
        overflow: "hidden",
        display: "flex",
        background:
          "radial-gradient(circle at top left, #1f2937 0, #020617 35%, #020617 60%, #111827 100%), linear-gradient(135deg, rgba(248,113,113,0.15), rgba(251,146,60,0.05))",
        color: "#e5e7eb",
        fontFamily:
          "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
      }}
    >
      {/* Side navigation */}
      <aside
        style={{
          width: "250px",
          borderRight: "1px solid #1f2937",
          background:
            "linear-gradient(180deg, rgba(15,23,42,0.98), rgba(15,23,42,0.94))",
          display: "flex",
          flexDirection: "column",
          padding: "1.2rem 1rem",
          boxShadow: "10px 0 40px rgba(15,23,42,0.9)",
          zIndex: 10,
        }}
      >
        <div
          style={{
            marginBottom: "1.2rem",
            display: "flex",
            alignItems: "center",
            gap: "0.6rem",
          }}
        >
          <div
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "8px",
              background:
                "radial-gradient(circle at 30% 20%, #38bdf8 0, #1d4ed8 45%, #020617 100%)",
              boxShadow: "0 0 22px rgba(56,189,248,0.8)",
            }}
          />
          <div>
            <div style={{ fontSize: "0.9rem", opacity: 0.7 }}>Lung‑XAI Suite</div>
            <div style={{ fontWeight: 700, fontSize: "1.05rem" }}>
              CT Decision Support
            </div>
          </div>
        </div>

        <nav style={{ marginTop: "1rem", flex: 1 }}>
          {[
            { key: "dashboard", label: "Live Analysis" },
            { key: "patients", label: "Patient queue (mock)" },
            { key: "qa", label: "Quality & XAI notes" },
          ].map((item) => {
            const active = activeTab === item.key;
            return (
              <button
                key={item.key}
                onClick={() => setActiveTab(item.key)}
                style={{
                  width: "100%",
                  textAlign: "left",
                  padding: "0.55rem 0.7rem",
                  borderRadius: "0.6rem",
                  border: "none",
                  marginBottom: "0.25rem",
                  cursor: "pointer",
                  background: active ? "#0b1120" : "transparent",
                  color: active ? "#e5e7eb" : "#9ca3af",
                  fontSize: "0.9rem",
                  fontWeight: active ? 600 : 500,
                  boxShadow: active
                    ? "0 0 26px rgba(248,113,113,0.4)"
                    : "none",
                  transition:
                    "background 0.15s ease, box-shadow 0.15s ease, transform 0.15s",
                  transform: active ? "translateX(3px)" : "translateX(0)",
                }}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        <div
          style={{
            fontSize: "0.75rem",
            opacity: 0.7,
            marginTop: "0.8rem",
          }}
        >
          <div>Model: ResNet‑18 CT classifier</div>
          <div>Explainability: Grad‑CAM (local)</div>
        </div>
      </aside>

      {/* Main workspace */}
      <main
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          padding: "1.3rem 1.7rem 1.6rem",
          overflowY: "auto",
        }}
      >
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: "1rem",
          }}
        >
          <div>
            <h1 style={{ fontSize: "1.7rem", marginBottom: "0.2rem" }}>
              Lung Cancer CT Detection – XAI Control Panel
            </h1>
            <p style={{ fontSize: "0.9rem", opacity: 0.8 }}>
              Upload CT slices, review AI predictions, and inspect Grad‑CAM
              heatmaps to validate decision‑making.
            </p>
          </div>
          <div style={{ textAlign: "right", fontSize: "0.8rem", opacity: 0.8 }}>
            <div>
              Status:{" "}
              <span style={{ color: "#22c55e", fontWeight: 600 }}>Online</span>
            </div>
            <div>Endpoint: 127.0.0.1:8000/api/v1</div>
          </div>
        </header>

        {activeTab === "dashboard" && (
          <>
            {/* Upload card */}
            <section
              style={{
                borderRadius: "14px",
                border: "1px solid #1f2937",
                padding: "1.1rem 1.2rem",
                marginBottom: "1.3rem",
                background:
                  "linear-gradient(115deg, rgba(15,23,42,0.98), rgba(15,23,42,0.85))",
                boxShadow:
                  "0 18px 48px rgba(15,23,42,0.9), 0 0 30px rgba(248,113,113,0.25)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "1.6rem",
              }}
            >
              <div style={{ flex: 1.4 }}>
                <div style={{ marginBottom: "0.7rem" }}>
                  <label
                    style={{
                      fontSize: "0.9rem",
                      fontWeight: 600,
                      marginBottom: "0.25rem",
                      display: "block",
                    }}
                  >
                    CT slice upload
                  </label>
                  <input
                    type="file"
                    accept="image/png,image/jpeg"
                    onChange={handleFileChange}
                    style={{ fontSize: "0.85rem" }}
                  />
                  {file && (
                    <p
                      style={{
                        fontSize: "0.8rem",
                        opacity: 0.8,
                        marginTop: "0.25rem",
                      }}
                    >
                      Selected: <strong>{file.name}</strong>
                    </p>
                  )}
                </div>
                <div style={{ display: "flex", gap: "0.7rem", marginTop: "0.5rem" }}>
                  <button
                    onClick={uploadAndPredict}
                    disabled={predictLoading || explainLoading}
                    style={{
                      padding: "0.5rem 1.1rem",
                      borderRadius: "999px",
                      border: "none",
                      background: "#22c55e",
                      color: "#022c22",
                      fontWeight: 600,
                      fontSize: "0.9rem",
                      cursor:
                        predictLoading || explainLoading ? "default" : "pointer",
                      boxShadow: "0 10px 26px rgba(34,197,94,0.6)",
                      transform:
                        predictLoading || explainLoading ? "scale(0.98)" : "scale(1)",
                      transition:
                        "transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s",
                      opacity: predictLoading || explainLoading ? 0.7 : 1,
                    }}
                  >
                    {predictLoading ? "Predicting…" : "Predict"}
                  </button>
                  <button
                    onClick={uploadAndExplain}
                    disabled={explainLoading}
                    style={{
                      padding: "0.5rem 1.1rem",
                      borderRadius: "999px",
                      border: "none",
                      background: "#3b82f6",
                      color: "#eff6ff",
                      fontWeight: 600,
                      fontSize: "0.9rem",
                      cursor: explainLoading ? "default" : "pointer",
                      boxShadow: "0 10px 26px rgba(59,130,246,0.7)",
                      transform: explainLoading ? "scale(0.98)" : "scale(1)",
                      transition:
                        "transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s",
                      opacity: explainLoading ? 0.75 : 1,
                    }}
                  >
                    {explainLoading ? "Generating heatmap…" : "Explain (Grad‑CAM)"}
                  </button>
                </div>
                {error && (
                  <p
                    style={{
                      color: "#f97316",
                      marginTop: "0.6rem",
                      fontSize: "0.8rem",
                    }}
                  >
                    {error}
                  </p>
                )}
              </div>

              <div
                style={{
                  flex: 1,
                  borderLeft: "1px solid #1f2937",
                  paddingLeft: "1.4rem",
                  fontSize: "0.8rem",
                  opacity: 0.85,
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: "0.3rem" }}>
                  Explainable AI notes
                </div>
                <ul style={{ paddingLeft: "1.1rem", margin: 0 }}>
                  <li>
                    Heatmaps highlight the **most influential** CT regions used
                    by the model for diagnosis.
                  </li>
                  <li>
                    Radiologists can confirm that focus lies on nodules, masses,
                    or consolidation, not artifacts.
                  </li>
                  <li>
                    Misaligned heatmaps reveal bias and guide model refinement
                    for safer deployment.
                  </li>
                </ul>
              </div>
            </section>

            {/* Main grid */}
            <section
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(0, 1.1fr) minmax(0, 1.4fr)",
                gap: "1.8rem",
                alignItems: "flex-start",
              }}
            >
              <div
                style={{
                  background: "#020617",
                  borderRadius: "14px",
                  padding: "1.1rem 1.2rem 1.3rem",
                  border: "1px solid #1f2937",
                  boxShadow: "0 16px 42px rgba(15,23,42,0.9)",
                  minHeight: "260px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "0.5rem",
                  }}
                >
                  <h2 style={{ fontSize: "1.05rem" }}>Prediction summary</h2>
                  {(predictLoading || explainLoading) && (
                    <LoadingPulse label="Running inference…" />
                  )}
                </div>

                {prediction ? (
                  <>
                    <p
                      style={{
                        fontSize: "0.9rem",
                        marginBottom: "0.5rem",
                      }}
                    >
                      Predicted class index:{" "}
                      <span style={{ fontWeight: 700 }}>
                        {prediction.predicted_class}
                      </span>{" "}
                      (
                      {
                        CLASS_LABELS[prediction.predicted_class] ||
                        "N/A"
                      }
                      )
                    </p>
                    <p
                      style={{
                        fontSize: "0.8rem",
                        opacity: 0.8,
                        marginBottom: "0.4rem",
                      }}
                    >
                      Softmax probabilities for each class provide a confidence
                      distribution for differential diagnosis.
                    </p>
                    {renderProbabilities()}
                  </>
                ) : (
                  <p style={{ fontSize: "0.9rem", opacity: 0.8 }}>
                    Upload a CT slice and run <strong>Predict</strong> to view
                    AI classification and confidence scores.
                  </p>
                )}
              </div>

              <div
                style={{
                  background: "#020617",
                  borderRadius: "14px",
                  padding: "1.1rem 1.2rem 1.3rem",
                  border: "1px solid #1f2937",
                  boxShadow: "0 16px 42px rgba(15,23,42,0.95)",
                  minHeight: "260px",
                }}
              >
                <h2 style={{ fontSize: "1.05rem", marginBottom: "0.5rem" }}>
                  Grad‑CAM explainability
                </h2>
                {heatmapSrc ? (
                  <>
                    <div
                      style={{
                        borderRadius: "12px",
                        overflow: "hidden",
                        border: "1px solid #0b1120",
                        boxShadow:
                          "0 22px 60px rgba(15,23,42,0.95), 0 0 35px rgba(248,113,113,0.55)",
                        animation: "pulse 3s infinite",
                      }}
                    >
                      <img
                        src={heatmapSrc}
                        alt="Grad-CAM heatmap"
                        style={{
                          width: "100%",
                          display: "block",
                        }}
                      />
                    </div>
                    <p
                      style={{
                        fontSize: "0.8rem",
                        opacity: 0.8,
                        marginTop: "0.6rem",
                      }}
                    >
                      Warmer colors (yellow–red) mark regions that most
                      influenced this prediction, helping clinicians confirm
                      that the AI relies on appropriate diagnostic markers on
                      the CT slice.
                    </p>
                  </>
                ) : (
                  <p style={{ fontSize: "0.9rem", opacity: 0.8 }}>
                    After prediction, click <strong>Explain (Grad‑CAM)</strong>{" "}
                    to generate a heatmap highlighting the most influential
                    regions of the CT scan.
                  </p>
                )}
              </div>
            </section>
          </>
        )}

        {activeTab !== "dashboard" && (
          <section
            style={{
              marginTop: "1.3rem",
              fontSize: "0.9rem",
              opacity: 0.75,
            }}
          >
            This area is reserved for future hospital workflow features such as
            patient lists, QA dashboards, or XAI audit logs.
          </section>
        )}
      </main>

      <style>{`
        @keyframes pulse {
          0% { transform: scale(1); opacity: 0.9; }
          50% { transform: scale(1.02); opacity: 0.6; }
          100% { transform: scale(1); opacity: 0.9; }
        }
      `}</style>
    </div>
  );
}

export default App;
