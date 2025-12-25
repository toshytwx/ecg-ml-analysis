const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs-extra');
const { v4: uuidv4 } = require('uuid');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'client/build')));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    fs.ensureDirSync(uploadDir);
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueId = uuidv4();
    const ext = path.extname(file.originalname);
    cb(null, `${uniqueId}${ext}`);
  }
});

const upload = multer({ 
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['.hea', '.dat'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error('Only .hea and .dat files are allowed'), false);
    }
  },
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  }
});

// Age group mapping
const AGE_GROUPS = {
  0: '18-25 years',
  1: '26-30 years', 
  2: '31-35 years',
  3: '36-40 years',
  4: '41-45 years',
  5: '46-50 years',
  6: '51-55 years',
  7: '56-60 years',
  8: '61-65 years',
  9: '66-70 years',
  10: '71-75 years',
  11: '76-80 years',
  12: '81-85 years',
  13: '86-90 years',
  14: '91+ years'
};

// Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'ECG Age Prediction API is running' });
});

app.post('/api/upload', upload.fields([
  { name: 'heaFile', maxCount: 1 },
  { name: 'datFile', maxCount: 1 }
]), async (req, res) => {
  console.log('Received upload request');
  try {
    if (!req.files || !req.files.heaFile || !req.files.datFile) {
      return res.status(400).json({ 
        error: 'Both .hea and .dat files are required' 
      });
    }

    const heaFile = req.files.heaFile[0];
    const datFile = req.files.datFile[0];
    
    // Validate file types
    if (path.extname(heaFile.originalname).toLowerCase() !== '.hea') {
      return res.status(400).json({ 
        error: 'First file must be a .hea file' 
      });
    }
    
    if (path.extname(datFile.originalname).toLowerCase() !== '.dat') {
      return res.status(400).json({ 
        error: 'Second file must be a .dat file' 
      });
    }

    // Create a unique session directory
    const sessionId = uuidv4();
    const sessionDir = path.join(__dirname, 'temp', sessionId);
    await fs.ensureDir(sessionDir);

    // Copy files to session directory with original names
    const heaPath = path.join(sessionDir, heaFile.originalname);
    const datPath = path.join(sessionDir, datFile.originalname);
    
    console.log('Copying files with original names:');
    console.log('HEA:', heaFile.originalname, '->', heaPath);
    console.log('DAT:', datFile.originalname, '->', datPath);
    
    await fs.copy(heaFile.path, heaPath);
    await fs.copy(datFile.path, datPath);

    // Clean up uploaded files
    await fs.remove(heaFile.path);
    await fs.remove(datFile.path);

    // Run Python prediction script using spawn
    const pythonProcess = spawn('python3', [
      'predict_ecg.py',
      sessionDir,
      path.join(__dirname, '..', 'ecg_cnn_outputs', 'best_model.pth')
    ], {
      cwd: __dirname,
      timeout: 30000
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', async (code) => {
      console.log('Python script completed with code:', code);
      console.log('STDOUT:', stdout);
      console.log('STDERR:', stderr);
      
      // Clean up session directory
      try {
        await fs.remove(sessionDir);
      } catch (cleanupErr) {
        console.error('Cleanup error:', cleanupErr);
      }

      if (code !== 0) {
        console.error('Python script error, exit code:', code);
        return res.status(500).json({
          error: 'Failed to process ECG files',
          details: stderr || 'Python script failed'
        });
      }

      try {
        // Extract JSON from stdout (skip warning messages)
        const lines = stdout.trim().split('\n');
        const jsonLine = lines.find(line => line.startsWith('{'));
        
        if (!jsonLine) {
          throw new Error('No JSON output found in Python script');
        }
        
        const prediction = JSON.parse(jsonLine);
        const ageGroup = AGE_GROUPS[prediction.predicted_class] || 'Unknown';
        
        res.json({
          predicted_age_group: prediction.predicted_class,
          confidence: prediction.confidence,
          all_probabilities: prediction.probabilities,
          signal_data: prediction.signal_data
        });
      } catch (parseErr) {
        console.error('Parse error:', parseErr);
        res.status(500).json({ 
          error: 'Failed to parse prediction results',
          details: parseErr.message 
        });
      }
    });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ 
      error: 'Upload failed',
      details: error.message 
    });
  }
});

// Serve React app
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'client/build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📊 ECG Age Prediction API ready`);
});
