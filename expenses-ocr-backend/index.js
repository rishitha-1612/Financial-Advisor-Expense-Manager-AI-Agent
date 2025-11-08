const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Python Flask API URL
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:5000';

// In-memory storage for expenses (replace with database in production)
let expenses = [];

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    message: 'Node.js backend is running',
    pythonAPI: PYTHON_API_URL
  });
});

// Upload and process receipt
app.post('/api/upload-receipt', upload.single('receipt'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Forward image to Python Flask API
    const formData = new FormData();
    formData.append('image', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });

    const response = await axios.post(
      `${PYTHON_API_URL}/api/extract-receipt`,
      formData,
      {
        headers: formData.getHeaders(),
        timeout: 30000 // 30 second timeout
      }
    );

    if (response.data.success) {
      // Store the expense
      const expense = {
        id: Date.now().toString(),
        ...response.data.data,
        uploaded_at: new Date().toISOString()
      };
      expenses.push(expense);

      res.json({
        success: true,
        message: 'Receipt processed successfully',
        data: expense,
        raw_ocr: response.data.raw_ocr
      });
    } else {
      res.status(500).json({ error: 'Failed to process receipt' });
    }
  } catch (error) {
    console.error('Error processing receipt:', error.message);
    res.status(500).json({ 
      error: 'Failed to process receipt',
      details: error.message
    });
  }
});

// Get all expenses
app.get('/api/expenses', (req, res) => {
  res.json({
    success: true,
    expenses: expenses,
    total: expenses.reduce((sum, exp) => sum + (exp.total_amount || 0), 0),
    count: expenses.length
  });
});

// Get single expense
app.get('/api/expenses/:id', (req, res) => {
  const expense = expenses.find(e => e.id === req.params.id);
  if (expense) {
    res.json({ success: true, expense });
  } else {
    res.status(404).json({ error: 'Expense not found' });
  }
});

// Delete expense
app.delete('/api/expenses/:id', (req, res) => {
  const index = expenses.findIndex(e => e.id === req.params.id);
  if (index !== -1) {
    expenses.splice(index, 1);
    res.json({ success: true, message: 'Expense deleted' });
  } else {
    res.status(404).json({ error: 'Expense not found' });
  }
});

// Get financial advice
app.post('/api/financial-advice', async (req, res) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }

    const response = await axios.post(
      `${PYTHON_API_URL}/api/financial-advice`,
      { question },
      { timeout: 30000 }
    );

    res.json(response.data);
  } catch (error) {
    console.error('Error getting financial advice:', error.message);
    res.status(500).json({ 
      error: 'Failed to get financial advice',
      details: error.message
    });
  }
});

// Analyze expenses
app.post('/api/analyze-expenses', async (req, res) => {
  try {
    const response = await axios.post(
      `${PYTHON_API_URL}/api/analyze-expenses`,
      { expenses },
      { timeout: 30000 }
    );

    res.json(response.data);
  } catch (error) {
    console.error('Error analyzing expenses:', error.message);
    res.status(500).json({ 
      error: 'Failed to analyze expenses',
      details: error.message
    });
  }
});

// Get expense statistics
app.get('/api/statistics', (req, res) => {
  const total = expenses.reduce((sum, exp) => sum + (exp.total_amount || 0), 0);
  const avgExpense = expenses.length > 0 ? total / expenses.length : 0;
  
  // Group by date
  const byDate = expenses.reduce((acc, exp) => {
    const date = exp.date || 'Unknown';
    acc[date] = (acc[date] || 0) + (exp.total_amount || 0);
    return acc;
  }, {});

  res.json({
    success: true,
    statistics: {
      total_expenses: total,
      transaction_count: expenses.length,
      average_expense: avgExpense,
      by_date: byDate
    }
  });
});

// Clear all expenses (for testing)
app.delete('/api/expenses', (req, res) => {
  expenses = [];
  res.json({ success: true, message: 'All expenses cleared' });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ 
    error: 'Internal server error',
    message: err.message
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Node.js backend server running on port ${PORT}`);
  console.log(`📡 Python API URL: ${PYTHON_API_URL}`);
  console.log(`💡 Make sure Python Flask server is running on port 5000`);
});

module.exports = app;
