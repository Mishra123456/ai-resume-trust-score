import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileText, CheckCircle, AlertCircle, X } from "lucide-react";

interface CSVUploaderProps {
  onDataLoaded: (data: any) => void; // backend response
  uploadRef: React.RefObject<HTMLDivElement>;
}

const REQUIRED_COLUMNS = [
  "date",
  "model_decision",
  "human_decision",
  "confidence_note",
];

export const CSVUploader = ({ onDataLoaded, uploadRef }: CSVUploaderProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadState, setUploadState] = useState<
    "idle" | "loading" | "success" | "error"
  >("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  // --------------------------------
  // Upload CSV → Backend
  // --------------------------------
  const uploadToBackend = async (file: File) => {
    setUploadState("loading");
    setProgress(0);
    setError(null);
    setFileName(file.name);

    // fake progress animation (UI only)
    let current = 0;
    const interval = setInterval(() => {
      current += Math.random() * 12;
      setProgress(Math.min(current, 95));
    }, 120);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Backend error");
      }

      const data = await res.json();

      clearInterval(interval);
      setProgress(100);
      setUploadState("success");

      // Small delay for animation smoothness
      setTimeout(() => {
        onDataLoaded(data);
      }, 600);
    } catch (err) {
      clearInterval(interval);
      setError("Failed to connect to backend. Is the server running?");
      setUploadState("error");
    }
  };

  // --------------------------------
  // File Handling
  // --------------------------------
  const handleFile = useCallback((file: File) => {
    if (!file.name.endsWith(".csv")) {
      setError("Please upload a CSV file");
      setUploadState("error");
      return;
    }

    uploadToBackend(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const reset = () => {
    setUploadState("idle");
    setProgress(0);
    setError(null);
    setFileName(null);
  };

  // --------------------------------
  // UI
  // --------------------------------
  return (
    <section ref={uploadRef} className="relative py-24 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-radial from-purple/5 via-transparent to-transparent" />

      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Upload Your Data</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Upload a CSV file to generate AI trust analytics in real time
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="max-w-2xl mx-auto"
        >
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`
              relative p-12 rounded-2xl border-2 border-dashed transition-all duration-300
              ${
                isDragging
                  ? "border-primary bg-primary/10 scale-[1.02]"
                  : "border-muted hover:border-muted-foreground/50 hover:bg-card/50"
              }
              ${uploadState === "success" ? "border-cyan bg-cyan/10" : ""}
              ${
                uploadState === "error"
                  ? "border-destructive bg-destructive/10"
                  : ""
              }
            `}
          >
            <AnimatePresence mode="wait">
              {uploadState === "idle" && (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-center"
                >
                  <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-purple/20 to-cyan/20 mb-6">
                    <Upload className="w-10 h-10 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">
                    Drop your CSV file here
                  </h3>
                  <p className="text-muted-foreground mb-6">
                    or click to browse
                  </p>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileInput}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <div className="flex flex-wrap justify-center gap-2 text-xs text-muted-foreground">
                    {REQUIRED_COLUMNS.map((c) => (
                      <span key={c} className="px-3 py-1 rounded-full bg-muted">
                        {c}
                      </span>
                    ))}
                  </div>
                </motion.div>
              )}

              {uploadState === "loading" && (
                <motion.div key="loading" className="text-center">
                  <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-purple/20 to-cyan/20 mb-6">
                    <FileText className="w-10 h-10 text-primary animate-pulse" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">{fileName}</h3>
                  <p className="text-muted-foreground mb-6">
                    Analyzing trust signals…
                  </p>

                  <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                    <motion.div
                      animate={{ width: `${progress}%` }}
                      className="h-full rounded-full"
                      style={{ background: "var(--gradient-primary)" }}
                    />
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground">
                    {Math.round(progress)}%
                  </p>
                </motion.div>
              )}

              {uploadState === "success" && (
                <motion.div key="success" className="text-center">
                  <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-cyan/20 mb-6">
                    <CheckCircle className="w-10 h-10 text-cyan" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2 text-cyan">
                    Upload Complete
                  </h3>
                  <p className="text-muted-foreground">Dashboard is loading…</p>
                </motion.div>
              )}

              {uploadState === "error" && (
                <motion.div key="error" className="text-center">
                  <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-destructive/20 mb-6">
                    <AlertCircle className="w-10 h-10 text-destructive" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2 text-destructive">
                    Upload Failed
                  </h3>
                  <p className="text-muted-foreground mb-4">{error}</p>
                  <button
                    onClick={reset}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-muted hover:bg-muted/80 transition-colors"
                  >
                    <X className="w-4 h-4" />
                    Try Again
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </section>
  );
};
