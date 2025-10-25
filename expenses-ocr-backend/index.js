const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs").promises;
const fsSync = require("fs");
const crypto = require("crypto");

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || "*",
  methods: ["GET", "POST", "OPTIONS"],
  credentials: true
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb' }));

// Directories
const UPLOADS = path.join(__dirname, "uploads");
const DATA = path.join(__dirname, "data");
const SCRIPTS = path.join(__dirname, "scripts");

// Ensure directories exist
[UPLOADS, DATA, SCRIPTS].forEach(dir => {
  if (!fsSync.existsSync(dir)) {
    fsSync.mkdirSync(dir, { recursive: true });
  }
});

// Multer config with validation
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const name = crypto.randomBytes(8).toString("hex");
    cb(null, `${name}${ext}`);
  }
});

const fileFilter = (req, file, cb) => {
  const ALLOWED_MIMETYPES = ['image/jpeg', 'image/png', 'image/jpg'];
  const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png'];
  
  const ext = path.extname(file.originalname).toLowerCase();
  
  if (!ALLOWED_MIMETYPES.includes(file.mimetype) || !ALLOWED_EXTENSIONS.includes(ext)) {
    return cb(new Error('Invalid file type. Only JPEG and PNG allowed.'));
  }
  cb(null, true);
};

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB
});

// Multer error handler
const handleUploadError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'FILE_TOO_LARGE') return res.status(400).json({ error: 'File too large. Max 10MB.' });
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  } else if (err) {
    return res.status(400).json({ error: err.message });
  }
  next();
};

// Helper: parse CSV safely
const parseCSV = (csvContent) => {
  if (!csvContent || csvContent.trim().length === 0) return [];
  const lines = csvContent.trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  return lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim().replace(/^"|"$/g, ''));
    const row = {};
    headers.forEach((h, i) => row[h] = values[i] || '');
    return row;
  });
};

// Helper: execute Python script
const executePythonScript = (scriptPath, args, timeout = 30000) => {
  return new Promise((resolve, reject) => {
    const py = spawn('python', [scriptPath, ...args]);
    let stdout = '', stderr = '';
    const timeoutId = setTimeout(() => {
      py.kill('SIGTERM');
      reject(new Error(`Python script timed out after ${timeout / 1000}s`));
    }, timeout);

    py.stdout.on('data', (data) => stdout += data.toString());
    py.stderr.on('data', (data) => stderr += data.toString());

    py.on('close', (code) => {
      clearTimeout(timeoutId);
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(stderr || `Python exited with code ${code}`));
    });

    py.on('error', (err) => {
      clearTimeout(timeoutId);
      reject(new Error(`Failed to spawn Python: ${err.message}`));
    });
  });
};

// POST /upload: main route
app.post("/upload", upload.single("image"), handleUploadError, async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });
  
  const imagePath = req.file.path;
  const fileName = req.file.originalname;

  try {
    console.log(`📸 Processing: ${fileName}`);

    // Call Python OCR + categorization + advice script
    const result = await executePythonScript(
      path.join(SCRIPTS, "extract_expenses.py"),
      [imagePath],
      40000
    );

    const stdout = result.stdout;

    if (!stdout.includes("CSV generated")) throw new Error("CSV generation failed");

    // Read CSV if exists
    const csvPath = path.join(DATA, "expenses.csv");
    let csvContent = '';
    try { csvContent = await fs.readFile(csvPath, 'utf-8'); } 
    catch (err) { console.warn('CSV not found:', err.message); }

    // Extract raw OCR text
    const rawStart = stdout.indexOf("RAW_OUTPUT_START");
    const rawEnd = stdout.indexOf("RAW_OUTPUT_END");
    const rawOutput = rawStart !== -1 && rawEnd !== -1
      ? stdout.substring(rawStart + 16, rawEnd).trim()
      : '';

    // Parse CSV
    const csvData = parseCSV(csvContent);

    console.log(`✅ Processed: ${fileName}`);

    // Response includes CSV, first row, and raw OCR
    return res.json({
      success: true,
      file: fileName,
      csv: csvContent,
      raw_output: rawOutput,
      data: csvData[0] || {}
    });

  } catch (err) {
    console.error(`❌ Processing error: ${err.message}`);
    return res.status(500).json({ error: "Processing failed", details: err.message });
  } finally {
    // Clean up uploaded file
    try { await fs.unlink(imagePath); } catch (err) { console.warn(`Failed to delete temp file: ${err.message}`); }
  }
});

// Health check
app.get("/health", (req, res) => res.json({ status: "ok", timestamp: new Date().toISOString() }));

// Stats
app.get("/stats", async (req, res) => {
  try {
    const uploadCount = (await fs.readdir(UPLOADS)).length;
    const dataFiles = (await fs.readdir(DATA)).length;
    res.json({ uploads: uploadCount, dataFiles, port: PORT });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 404 handler
app.use((req, res) => res.status(404).json({ error: "Endpoint not found" }));

// Global error
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ error: "Internal server error", message: process.env.NODE_ENV === 'development' ? err.message : undefined });
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`✅ Backend running on port ${PORT}`);
  console.log(`📁 Uploads:  ${UPLOADS}`);
  console.log(`📁 Data:     ${DATA}`);
  console.log(`📁 Scripts:  ${SCRIPTS}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => process.exit(0));
});
