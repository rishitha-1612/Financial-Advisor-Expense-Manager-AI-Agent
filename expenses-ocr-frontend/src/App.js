import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fileInfo, setFileInfo] = useState(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    
    if (!selectedFile) {
      setError(null);
      setFile(null);
      setPreview(null);
      setFileInfo(null);
      return;
    }

    // Validate file type
    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      setError("❌ Invalid file type. Only JPEG and PNG images allowed.");
      setFile(null);
      setPreview(null);
      return;
    }

    // Validate file size
    if (selectedFile.size > MAX_FILE_SIZE) {
      setError(`❌ File too large. Maximum size: 10MB. Your file: ${(selectedFile.size / 1024 / 1024).toFixed(2)}MB`);
      setFile(null);
      setPreview(null);
      return;
    }

    // Set file and create preview
    setFile(selectedFile);
    setError(null);
    setResult(null);

    // File info
    setFileInfo({
      name: selectedFile.name,
      size: (selectedFile.size / 1024).toFixed(2),
      type: selectedFile.type
    });

    // Create image preview
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.onerror = () => {
      setError("Failed to read file");
      setFile(null);
      setPreview(null);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch(`${BACKEND_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok || data.error) {
        setError(`❌ ${data.error || "Unknown error occurred"}`);
        if (data.details) {
          console.error("Error details:", data.details);
        }
      } else {
        setResult(data);
        setError(null);
      }
    } catch (err) {
      setError(`❌ Server not reachable at ${BACKEND_URL}. Make sure backend is running.`);
      console.error("Fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  const parseCSV = (csv) => {
    try {
      const lines = csv.trim().split("\n");
      if (lines.length < 2) return [];

      // Parse headers
      const headers = lines[0]
        .split(",")
        .map((h) => h.trim().replace(/^"|"$/g, ""));

      // Parse rows
      return lines.slice(1).map((line) => {
        const values = line
          .split(",")
          .map((v) => v.trim().replace(/^"|"$/g, ""));
        const row = {};
        headers.forEach((h, i) => {
          row[h] = values[i] || "";
        });
        return row;
      });
    } catch (err) {
      console.error("CSV parsing error:", err);
      return [];
    }
  };

  const rows = result?.csv ? parseCSV(result.csv) : [];
  const data = rows[0] || {};

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "1rem",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "700px",
          background: "#fff",
          padding: "2rem",
          borderRadius: "15px",
          boxShadow: "0 20px 60px rgba(0, 0, 0, 0.3)",
        }}
      >
        {/* Header */}
        <h1
          style={{
            textAlign: "center",
            marginBottom: "0.5rem",
            color: "#333",
            fontSize: "2.2rem",
            fontWeight: "700",
          }}
        >
          💰 Expense OCR
        </h1>
        <p
          style={{
            textAlign: "center",
            color: "#666",
            fontSize: "0.95rem",
            marginBottom: "2rem",
          }}
        >
          Extract receipt details automatically using AI
        </p>

        {/* File Input */}
        <label
          style={{
            display: "block",
            marginBottom: "1.5rem",
          }}
        >
          <p style={{ marginBottom: "0.75rem", color: "#333", fontWeight: "600" }}>
            📷 Upload Receipt Image
          </p>
          <input
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            onChange={handleFileChange}
            disabled={loading}
            style={{
              width: "100%",
              padding: "0.75rem",
              border: "2px dashed #667eea",
              borderRadius: "8px",
              cursor: loading ? "not-allowed" : "pointer",
              fontSize: "1rem",
              background: loading ? "#f5f5f5" : "#fff",
              opacity: loading ? 0.6 : 1,
              transition: "all 0.3s",
            }}
          />
        </label>

        {/* File Info */}
        {fileInfo && (
          <div
            style={{
              marginBottom: "1rem",
              padding: "0.75rem",
              background: "#f0f4ff",
              borderRadius: "8px",
              fontSize: "0.9rem",
              color: "#444",
            }}
          >
            <strong>📄 {fileInfo.name}</strong> · {fileInfo.size} KB
          </div>
        )}

        {/* Image Preview */}
        {preview && (
          <div style={{ marginBottom: "1.5rem", textAlign: "center" }}>
            <img
              src={preview}
              alt="preview"
              style={{
                maxWidth: "100%",
                maxHeight: "250px",
                borderRadius: "8px",
                border: "1px solid #e0e0e0",
                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
              }}
            />
          </div>
        )}

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          style={{
            width: "100%",
            padding: "0.85rem",
            borderRadius: "8px",
            border: "none",
            background: !file || loading ? "#ccc" : "#667eea",
            color: "#fff",
            fontSize: "1rem",
            fontWeight: "600",
            cursor: !file || loading ? "not-allowed" : "pointer",
            transition: "all 0.3s",
            opacity: !file || loading ? 0.7 : 1,
            transform: !file || loading ? "scale(1)" : "scale(1.02)",
          }}
          onMouseOver={(e) => {
            if (file && !loading) {
              e.target.style.background = "#5568d3";
              e.target.style.transform = "scale(1.02)";
            }
          }}
          onMouseOut={(e) => {
            if (file && !loading) {
              e.target.style.background = "#667eea";
              e.target.style.transform = "scale(1)";
            }
          }}
        >
          {loading ? "Processing..." : " Upload & Extract"}
        </button>

        {/* Error Display */}
        {error && (
          <div
            style={{
              marginTop: "1.5rem",
              background: "#ffebee",
              padding: "1rem",
              borderRadius: "8px",
              color: "#c33",
              borderLeft: "4px solid #c33",
              fontSize: "0.95rem",
            }}
          >
            {error}
          </div>
        )}

        {/* Results */}
        {result && !error && (
          <div style={{ marginTop: "2rem" }}>
            {/* Extracted Data Summary */}
            {data && Object.keys(data).length > 0 && (
              <div style={{ marginBottom: "2rem" }}>
                <h3
                  style={{
                    color: "#333",
                    marginBottom: "1rem",
                    fontSize: "1.1rem",
                    fontWeight: "600",
                  }}
                >
                  ✅ Extracted Data
                </h3>
                <div
                  style={{
                    background: "#f9f9f9",
                    padding: "1rem",
                    borderRadius: "8px",
                    border: "1px solid #e0e0e0",
                  }}
                >
                  {Object.entries(data).map(([key, value]) => (
                    <div
                      key={key}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        padding: "0.75rem 0",
                        borderBottom: "1px solid #eee",
                      }}
                    >
                      <strong style={{ color: "#667eea" }}>{key}:</strong>
                      <span style={{ color: "#333" }}>{value || "—"}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Full Data Table */}
            {rows.length > 0 && (
              <div style={{ marginBottom: "2rem" }}>
                <h3
                  style={{
                    color: "#333",
                    marginBottom: "1rem",
                    fontSize: "1.1rem",
                    fontWeight: "600",
                  }}
                >
                  📋 Full Data Table
                </h3>
                <div style={{ overflowX: "auto" }}>
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      fontSize: "0.9rem",
                      border: "1px solid #ddd",
                      background: "#fff",
                    }}
                  >
                    <thead>
                      <tr style={{ background: "#f5f5f5", borderBottom: "2px solid #ddd" }}>
                        {Object.keys(rows[0]).map((key) => (
                          <th
                            key={key}
                            style={{
                              padding: "0.9rem",
                              textAlign: "left",
                              fontWeight: "600",
                              color: "#333",
                            }}
                          >
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, idx) => (
                        <tr key={idx} style={{ borderBottom: "1px solid #eee" }}>
                          {Object.values(row).map((val, i) => (
                            <td
                              key={i}
                              style={{
                                padding: "0.9rem",
                                color: "#555",
                              }}
                            >
                              {val || "—"}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Raw OCR Output */}
            {result?.raw_output && (
              <div>
                <h3
                  style={{
                    color: "#333",
                    marginBottom: "1rem",
                    fontSize: "1.1rem",
                    fontWeight: "600",
                  }}
                >
                  🔍 Raw OCR Output
                </h3>
                <pre
                  style={{
                    background: "#f5f5f5",
                    padding: "1rem",
                    borderRadius: "8px",
                    maxHeight: "300px",
                    overflowY: "auto",
                    fontSize: "0.8rem",
                    whiteSpace: "pre-wrap",
                    wordWrap: "break-word",
                    border: "1px solid #ddd",
                    color: "#333",
                    lineHeight: "1.5",
                  }}
                >
                  {result.raw_output}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;