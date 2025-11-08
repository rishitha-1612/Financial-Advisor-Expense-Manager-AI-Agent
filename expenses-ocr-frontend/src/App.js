import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:5000';

function App() {
  const [expenses, setExpenses] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [activeTab, setActiveTab] = useState('upload');
  const [error, setError] = useState('');

  // Uploaded images state
  const [uploadedImages, setUploadedImages] = useState([]);
  const [selectedImageView, setSelectedImageView] = useState(null);
  const [showImageModal, setShowImageModal] = useState(false);

  // Multi-guru state
  const [multiGuruQuestion, setMultiGuruQuestion] = useState('');
  const [multiGuruAdvice, setMultiGuruAdvice] = useState(null);

  // Goals state
  const [goals, setGoals] = useState([]);
  const [newGoal, setNewGoal] = useState({
    name: '', target_amount: '', deadline: '', category: 'savings'
  });

  // Reports state
  const [monthlyReport, setMonthlyReport] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [dashboard, setDashboard] = useState(null);
  const [advancedAnalytics, setAdvancedAnalytics] = useState(null);

  // Investment state
  const [investmentForm, setInvestmentForm] = useState({
    total_savings: '', risk_profile: 'moderate'
  });
  const [investmentRecs, setInvestmentRecs] = useState('');

  useEffect(() => {
    checkApiHealth();
    fetchExpenses();
    fetchGoals();
    fetchUploadedImages();
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/health`, { timeout: 5000 });
      if (response.data.status === 'healthy') {
        console.log('[SUCCESS] Enhanced API is healthy');
        setError('');
      }
    } catch (error) {
      console.error('[ERROR] API health check failed:', error);
      setError('Cannot connect to backend. Make sure Flask server is running.');
    }
  };

  const fetchExpenses = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/expenses`, { timeout: 5000 });
      if (response.data.success) {
        setExpenses(response.data.expenses);
      }
    } catch (error) {
      console.error('Error fetching expenses:', error);
    }
  };

  const fetchGoals = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/goals`, { timeout: 5000 });
      if (response.data.success) {
        setGoals(response.data.goals);
      }
    } catch (error) {
      console.error('Error fetching goals:', error);
    }
  };

  const fetchUploadedImages = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/uploaded-images`, { timeout: 5000 });
      if (response.data.success) {
        setUploadedImages(response.data.images);
      }
    } catch (error) {
      console.error('Error fetching images:', error);
    }
  };

  const handleImageClick = async (imageId) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/uploaded-images/${imageId}`, { timeout: 10000 });
      if (response.data.success) {
        setSelectedImageView(response.data.image);
        setShowImageModal(true);
      }
    } catch (error) {
      console.error('Error loading image:', error);
      alert('Failed to load image');
    } finally {
      setLoading(false);
    }
  };

  const fetchMonthlyReport = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/reports/monthly`, { timeout: 10000 });
      if (response.data.success) {
        setMonthlyReport(response.data.report);
      }
    } catch (error) {
      console.error('Error fetching report:', error);
      setMonthlyReport({ error: 'Failed to generate report' });
    } finally {
      setLoading(false);
    }
  };

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/predict-expenses`, { timeout: 10000 });
      if (response.data.success) {
        setPredictions(response.data.predictions);
      }
    } catch (error) {
      console.error('Error fetching predictions:', error);
      setPredictions({ error: error.response?.data?.error || 'Failed to predict' });
    } finally {
      setLoading(false);
    }
  };

  const fetchDashboard = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/analytics/dashboard`, { timeout: 10000 });
      if (response.data.success) {
        setDashboard(response.data.dashboard);
      }
    } catch (error) {
      console.error('Error fetching dashboard:', error);
      setDashboard({ error: 'Failed to load dashboard' });
    } finally {
      setLoading(false);
    }
  };

  const fetchAdvancedAnalytics = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/analytics/advanced`, { timeout: 10000 });
      if (response.data.success) {
        setAdvancedAnalytics(response.data.analytics);
      }
    } catch (error) {
      console.error('Error fetching analytics:', error);
      setAdvancedAnalytics({ error: 'Failed to load analytics' });
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Input validation
      const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
      if (!validTypes.includes(file.type)) {
        setUploadStatus('Error: Invalid file type. Please select PNG, JPEG, GIF, or BMP.');
        setSelectedFile(null);
        return;
      }

      const maxSize = 10 * 1024 * 1024; // 10MB
      if (file.size > maxSize) {
        setUploadStatus('Error: File is too large. Maximum size is 10MB.');
        setSelectedFile(null);
        return;
      }

      if (file.size < 1024) { // Less than 1KB
        setUploadStatus('Error: File is too small. Please select a valid image.');
        setSelectedFile(null);
        return;
      }

      setSelectedFile(file);
      setUploadStatus('');
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('Error: Please select a file first');
      return;
    }

    setLoading(true);
    setUploadStatus('Processing receipt with enhanced OCR...');

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      
      const response = await axios.post(`${API_URL}/api/extract-receipt`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000
      });

      if (response.data.success) {
        setUploadStatus('Success: ' + (response.data.message || 'Receipt processed successfully!'));
        setSelectedFile(null);
        setPreviewUrl(null);
        await fetchExpenses();
        await fetchUploadedImages();
        
        setTimeout(() => {
          setUploadStatus('');
        }, 3000);
      } else {
        setUploadStatus('Error: ' + (response.data.error || 'Processing failed'));
      }
    } catch (error) {
      console.error('Upload error:', error);
      let errorMessage = 'Error: ';
      
      if (error.response?.data?.error) {
        errorMessage += error.response.data.error;
      } else if (error.code === 'ECONNABORTED') {
        errorMessage += 'Request timed out. Please try again with a clearer image.';
      } else if (error.request) {
        errorMessage += 'Cannot reach server. Please ensure the backend is running.';
      } else {
        errorMessage += 'An unexpected error occurred. Please try again.';
      }
      
      setUploadStatus(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteExpense = async (id) => {
    if (!window.confirm('Delete this expense?')) return;
    try {
      await axios.delete(`${API_URL}/api/expenses/${id}`);
      await fetchExpenses();
    } catch (error) {
      alert('Failed to delete expense');
    }
  };

  const handleMultiGuruAdvice = async () => {
    if (!multiGuruQuestion.trim()) return;
    setLoading(true);
    setMultiGuruAdvice(null);

    try {
      const response = await axios.post(`${API_URL}/api/advice/multi-guru`, {
        question: multiGuruQuestion
      }, { timeout: 20000 });

      if (response.data.success) {
        setMultiGuruAdvice(response.data.advice);
      }
    } catch (error) {
      setMultiGuruAdvice({ error: 'Failed to get advice' });
    } finally {
      setLoading(false);
    }
  };

  const handleCreateGoal = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(`${API_URL}/api/goals`, newGoal);
      if (response.data.success) {
        setGoals([...goals, response.data.goal]);
        setNewGoal({ name: '', target_amount: '', deadline: '', category: 'savings' });
      }
    } catch (error) {
      alert('Failed to create goal');
    }
  };

  const handleUpdateGoalProgress = async (goalId, newAmount) => {
    try {
      const response = await axios.put(`${API_URL}/api/goals/${goalId}`, {
        current_amount: newAmount
      });
      if (response.data.success) {
        await fetchGoals();
      }
    } catch (error) {
      alert('Failed to update goal');
    }
  };

  const handleDeleteGoal = async (goalId) => {
    if (!window.confirm('Delete this goal?')) return;
    try {
      await axios.delete(`${API_URL}/api/goals/${goalId}`);
      await fetchGoals();
    } catch (error) {
      alert('Failed to delete goal');
    }
  };

  const handleGetInvestmentRecs = async () => {
    if (!investmentForm.total_savings) return;
    setLoading(true);
    setInvestmentRecs('');

    try {
      const response = await axios.post(`${API_URL}/api/investment-recommendations`, 
        investmentForm, { timeout: 15000 });
      if (response.data.success) {
        setInvestmentRecs(response.data.recommendations);
      }
    } catch (error) {
      setInvestmentRecs('Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value) => parseFloat(value || 0).toFixed(2);

  return (
    <div className="App">
      <header className="app-header">
        <h1>Smart Expense Tracker Pro</h1>
        <p>Enhanced OCR | Multi-Guru Advice | Goal Tracking | Predictions</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="tabs">
        <button className={activeTab === 'upload' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('upload')}>Upload Receipt</button>
        <button className={activeTab === 'expenses' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('expenses')}>Expenses ({expenses.length})</button>
        <button className={activeTab === 'reports' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('reports')}>Reports</button>
        <button className={activeTab === 'goals' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('goals')}>Goals ({goals.length})</button>
        <button className={activeTab === 'multi-guru' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('multi-guru')}>Multi-Guru</button>
        <button className={activeTab === 'invest' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('invest')}>Invest</button>
        <button className={activeTab === 'dashboard' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('dashboard')}>Dashboard</button>
      </div>

      <div className="content">
        {/* UPLOAD TAB */}
        {activeTab === 'upload' && (
          <div className="upload-section">
            <div className="upload-subsection">
              <h2>Upload Receipt</h2>
              <div className="upload-container">
                <input type="file" accept="image/*" onChange={handleFileChange}
                  className="file-input" id="file-upload" disabled={loading} />
                <label htmlFor="file-upload" className="file-label">
                  {selectedFile ? selectedFile.name : 'Choose receipt image'}
                </label>
                {previewUrl && (
                  <div className="image-preview">
                    <img src={previewUrl} alt="Preview" />
                  </div>
                )}
                <button onClick={handleUpload} disabled={!selectedFile || loading}
                  className="btn btn-primary">
                  {loading ? 'Processing...' : 'Process Receipt'}
                </button>
              </div>
              {uploadStatus && (
                <div className={`status ${uploadStatus.toLowerCase().includes('success') ? 'success' : 'error'}`}>
                  {uploadStatus}
                </div>
              )}
            </div>

            <div className="upload-subsection">
              <h2>View Uploaded Images</h2>
              {uploadedImages.length === 0 ? (
                <p className="empty-message">No images uploaded yet</p>
              ) : (
                <div className="uploads-gallery">
                  {uploadedImages.map((image) => (
                    <div 
                      key={image.id} 
                      className="upload-thumbnail"
                      onClick={() => handleImageClick(image.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <div className="thumbnail-placeholder">
                        <span className="thumbnail-icon">Click to View</span>
                      </div>
                      <div className="thumbnail-info">
                        <p className="thumbnail-filename">{image.filename}</p>
                        <p className="thumbnail-date">{new Date(image.uploaded_at).toLocaleDateString()}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Image Modal */}
        {showImageModal && selectedImageView && (
          <div className="modal-overlay" onClick={() => setShowImageModal(false)}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <button className="modal-close" onClick={() => setShowImageModal(false)}>×</button>
              <img 
                src={`data:image/jpeg;base64,${selectedImageView.image_base64}`} 
                alt="Receipt" 
                className="modal-image"
              />
              <div className="modal-info">
                <p><strong>Store:</strong> {selectedImageView.store}</p>
                <p><strong>Amount:</strong> ${formatCurrency(selectedImageView.amount)}</p>
                <p><strong>Uploaded:</strong> {new Date(selectedImageView.uploaded_at).toLocaleString()}</p>
              </div>
            </div>
          </div>
        )}

        {/* EXPENSES TAB */}
        {activeTab === 'expenses' && (
          <div className="expenses-section">
            <h2>Your Expenses</h2>
            {expenses.length === 0 ? (
              <div className="empty-state">
                <p>No expenses yet.</p>
                <p>Upload a receipt to get started!</p>
                <button
                  className="btn btn-primary"
                  onClick={() => setActiveTab('upload')}
                >
                  Upload Your First Receipt
                </button>
              </div>
            ) : (
              <div className="expenses-list">
                {expenses.map((expense) => (
                  <div key={expense.id} className="expense-card">
                    <div className="expense-header">
                      <h3>
                        {expense.address || 'Unknown'}
                        <span className="category-badge">{expense.category}</span>
                      </h3>
                      <button onClick={() => handleDeleteExpense(expense.id)}
                        className="btn-delete">Delete</button>
                    </div>
                    <div className="expense-details">
                      <p><strong>Date:</strong> {expense.date || 'N/A'}</p>
                      {expense.items && expense.items.length > 0 && (
                        <div className="items-list">
                          <strong>Items:</strong>
                          <ul>
                            {expense.items.map((item, idx) => (
                              <li key={idx}>{item.item_name} - ${formatCurrency(item.price)}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <div className="expense-totals">
                        <p className="total"><strong>Total:</strong> ${formatCurrency(expense.total_amount)}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* REPORTS TAB */}
        {activeTab === 'reports' && (
          <div className="reports-section">
            <h2>Financial Reports & Predictions</h2>
            
            <div className="report-buttons">
              <button onClick={fetchMonthlyReport} disabled={loading || expenses.length === 0}
                className="btn btn-primary">
                {loading ? 'Loading...' : 'Generate Monthly Report'}
              </button>
              <button onClick={fetchPredictions} disabled={loading || expenses.length < 3}
                className="btn btn-primary">
                {loading ? 'Loading...' : 'Predict Next Month'}
              </button>
            </div>

            {monthlyReport && !monthlyReport.error && (
              <div className="report-card">
                <h3>Monthly Report</h3>
                <div className="stats-grid">
                  <div className="stat-box">
                    <div className="stat-value">${formatCurrency(monthlyReport.total_spent)}</div>
                    <div className="stat-label">Total Spent</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-value">{monthlyReport.transaction_count}</div>
                    <div className="stat-label">Transactions</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-value">${formatCurrency(monthlyReport.average_transaction)}</div>
                    <div className="stat-label">Average</div>
                  </div>
                </div>
                <div className="category-breakdown">
                  <h4>Spending by Category</h4>
                  {monthlyReport.top_categories.map((cat, idx) => (
                    <div key={idx} className="category-item">
                      <span>{cat.category}</span>
                      <span className="amount">${formatCurrency(cat.amount)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {predictions && !predictions.error && (
              <div className="report-card">
                <h3>Next Month Predictions</h3>
                <p className="info-text">Based on your spending patterns:</p>
                <div className="predictions-list">
                  {Object.entries(predictions).map(([category, amount]) => (
                    category !== 'total' && (
                      <div key={category} className="prediction-item">
                        <span>{category}</span>
                        <span className="amount">${formatCurrency(amount)}</span>
                      </div>
                    )
                  ))}
                  <div className="prediction-item total">
                    <strong>Total Predicted</strong>
                    <strong className="amount">${formatCurrency(predictions.total)}</strong>
                  </div>
                </div>
              </div>
            )}

            {expenses.length === 0 && (
              <div className="info-box">
                <p>Upload expenses to generate reports and predictions</p>
              </div>
            )}
          </div>
        )}

        {/* GOALS TAB */}
        {activeTab === 'goals' && (
          <div className="goals-section">
            <h2>Financial Goals</h2>
            
            <div className="goal-form">
              <h3>Create New Goal</h3>
              <form onSubmit={handleCreateGoal}>
                <input type="text" placeholder="Goal name" value={newGoal.name}
                  onChange={(e) => setNewGoal({...newGoal, name: e.target.value})} required />
                <input type="number" placeholder="Target amount" value={newGoal.target_amount}
                  onChange={(e) => setNewGoal({...newGoal, target_amount: e.target.value})} required />
                <input type="date" value={newGoal.deadline}
                  onChange={(e) => setNewGoal({...newGoal, deadline: e.target.value})} required />
                <select value={newGoal.category}
                  onChange={(e) => setNewGoal({...newGoal, category: e.target.value})}>
                  <option value="savings">Savings</option>
                  <option value="emergency">Emergency Fund</option>
                  <option value="investment">Investment</option>
                  <option value="debt">Debt Payoff</option>
                  <option value="purchase">Major Purchase</option>
                </select>
                <button type="submit" className="btn btn-primary">Add Goal</button>
              </form>
            </div>

            <div className="goals-list">
              {goals.length === 0 ? (
                <p className="empty-message">No goals yet. Create your first financial goal!</p>
              ) : (
                goals.map((goal) => {
                  const progress = (goal.current_amount / goal.target_amount) * 100;
                  return (
                    <div key={goal.id} className="goal-card">
                      <div className="goal-header">
                        <h3>{goal.name}</h3>
                        <button onClick={() => handleDeleteGoal(goal.id)} className="btn-delete">Delete</button>
                      </div>
                      <div className="goal-details">
                        <p><strong>Target:</strong> ${formatCurrency(goal.target_amount)}</p>
                        <p><strong>Current:</strong> ${formatCurrency(goal.current_amount)}</p>
                        <p><strong>Deadline:</strong> {goal.deadline}</p>
                        <p><strong>Category:</strong> {goal.category}</p>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{width: `${Math.min(progress, 100)}%`}}></div>
                      </div>
                      <p className="progress-text">{progress.toFixed(1)}% Complete</p>
                      <div className="goal-actions">
                        <input type="number" placeholder="Update amount" id={`goal-${goal.id}`}
                          step="0.01" />
                        <button onClick={() => {
                          const input = document.getElementById(`goal-${goal.id}`);
                          if (input.value) handleUpdateGoalProgress(goal.id, input.value);
                        }} className="btn btn-secondary">Update Progress</button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {/* MULTI-GURU TAB */}
        {activeTab === 'multi-guru' && (
          <div className="multi-guru-section">
            <h2>Multi-Guru Financial Advice</h2>
            <p className="section-description">
              Get perspectives from three different financial philosophies:
            </p>
            <div className="guru-badges">
              <span className="guru-badge conservative">Conservative</span>
              <span className="guru-badge balanced">Balanced</span>
              <span className="guru-badge aggressive">Aggressive</span>
            </div>

            <div className="advice-input">
              <textarea value={multiGuruQuestion}
                onChange={(e) => setMultiGuruQuestion(e.target.value)}
                placeholder="Ask a financial question (e.g., Should I invest aggressively or play it safe?)"
                rows="4" disabled={loading} />
              <button onClick={handleMultiGuruAdvice}
                disabled={loading || !multiGuruQuestion.trim()}
                className="btn btn-primary">
                {loading ? 'Consulting Gurus...' : 'Get Multi-Guru Advice'}
              </button>
            </div>

            {multiGuruAdvice && !multiGuruAdvice.error && (
              <div className="multi-guru-responses">
                <div className="guru-response conservative">
                  <h3>Conservative Guru</h3>
                  <p>{multiGuruAdvice.conservative}</p>
                </div>
                <div className="guru-response balanced">
                  <h3>Balanced Guru</h3>
                  <p>{multiGuruAdvice.balanced}</p>
                </div>
                <div className="guru-response aggressive">
                  <h3>Aggressive Guru</h3>
                  <p>{multiGuruAdvice.aggressive}</p>
                </div>
              </div>
            )}

            <div className="example-questions">
              <h4>Try asking:</h4>
              <button onClick={() => setMultiGuruQuestion("Should I pay off debt or invest?")} 
                className="example-btn">Should I pay off debt or invest?</button>
              <button onClick={() => setMultiGuruQuestion("How much should I save each month?")}
                className="example-btn">How much should I save each month?</button>
              <button onClick={() => setMultiGuruQuestion("Is now a good time to take risks?")}
                className="example-btn">Is now a good time to take risks?</button>
            </div>
          </div>
        )}

        {/* INVESTMENT TAB */}
        {activeTab === 'invest' && (
          <div className="investment-section">
            <h2>Investment Recommendations</h2>
            <p className="section-description">Get personalized investment advice based on your savings and risk tolerance</p>

            <div className="investment-form">
              <div className="form-group">
                <label>Total Savings Available:</label>
                <input type="number" placeholder="e.g., 10000" value={investmentForm.total_savings}
                  onChange={(e) => setInvestmentForm({...investmentForm, total_savings: e.target.value})} />
              </div>
              <div className="form-group">
                <label>Risk Profile:</label>
                <select value={investmentForm.risk_profile}
                  onChange={(e) => setInvestmentForm({...investmentForm, risk_profile: e.target.value})}>
                  <option value="conservative">Conservative (Safety first)</option>
                  <option value="moderate">Moderate (Balanced)</option>
                  <option value="aggressive">Aggressive (High growth)</option>
                </select>
              </div>
              <button onClick={handleGetInvestmentRecs}
                disabled={loading || !investmentForm.total_savings}
                className="btn btn-primary">
                {loading ? 'Analyzing...' : 'Get Recommendations'}
              </button>
            </div>

            {investmentRecs && (
              <div className="investment-recommendations">
                <h3>Your Personalized Investment Plan</h3>
                <pre className="recommendation-text">{investmentRecs}</pre>
              </div>
            )}
          </div>
        )}

        {/* DASHBOARD TAB */}
        {activeTab === 'dashboard' && (
          <div className="dashboard-section">
            <h2>Analytics Dashboard</h2>
            <button onClick={fetchDashboard} disabled={loading || expenses.length === 0}
              className="btn btn-primary">
              {loading ? 'Loading...' : 'Refresh Dashboard'}
            </button>

            <div style={{marginTop: '20px', marginBottom: '20px'}}>
              <button onClick={fetchAdvancedAnalytics} disabled={loading || expenses.length === 0}
                className="btn btn-secondary">
                {loading ? 'Loading...' : 'Advanced Analytics'}
              </button>
            </div>

            {advancedAnalytics && !advancedAnalytics.error && (
              <div className="analytics-panel">
                <h3>Advanced Financial Insights</h3>
                
                <div className="insights-grid">
                  <div className="insight-card">
                    <h4>Spending Insights</h4>
                    <p><strong>Daily Average:</strong> ${formatCurrency(advancedAnalytics.insights.daily_average)}</p>
                    <p><strong>Monthly Projection:</strong> ${formatCurrency(advancedAnalytics.insights.monthly_projection)}</p>
                    <p><strong>Recommended Budget:</strong> ${formatCurrency(advancedAnalytics.insights.recommended_monthly_budget)}</p>
                  </div>
                  
                  <div className="insight-card">
                    <h4>Top Category</h4>
                    <p><strong>{advancedAnalytics.insights.highest_spending_category}</strong></p>
                    <p className="highlight-amount">${formatCurrency(advancedAnalytics.insights.highest_category_amount)}</p>
                  </div>
                </div>

                {advancedAnalytics.warnings && advancedAnalytics.warnings.length > 0 && (
                  <div className="warnings-section">
                    <h4>Warnings & Recommendations</h4>
                    {advancedAnalytics.warnings.map((warning, idx) => (
                      <div key={idx} className="warning-item">
                        {warning}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {dashboard && !dashboard.error && (
              <div className="dashboard-grid">
                <div className="dashboard-card">
                  <h3>Overall Stats</h3>
                  <p><strong>Total Spent:</strong> ${formatCurrency(dashboard.total_spent)}</p>
                  <p><strong>Transactions:</strong> {dashboard.transaction_count}</p>
                </div>

                {dashboard.category_breakdown && (
                  <div className="dashboard-card">
                    <h3>Category Breakdown</h3>
                    {Object.entries(dashboard.category_breakdown).map(([cat, amt]) => (
                      <p key={cat}>{cat}: ${formatCurrency(amt)}</p>
                    ))}
                  </div>
                )}

                {dashboard.predictions && (
                  <div className="dashboard-card">
                    <h3>Predictions</h3>
                    <p><strong>Next Month Est.:</strong> ${formatCurrency(dashboard.predictions.total)}</p>
                  </div>
                )}

                {dashboard.goals_progress && (
                  <div className="dashboard-card">
                    <h3>Goals Progress</h3>
                    <p><strong>Total Target:</strong> ${formatCurrency(dashboard.goals_progress.total_target)}</p>
                    <p><strong>Achieved:</strong> ${formatCurrency(dashboard.goals_progress.total_achieved)}</p>
                    <div className="progress-bar">
                      <div className="progress-fill" 
                        style={{width: `${dashboard.goals_progress.percentage}%`}}></div>
                    </div>
                    <p>{dashboard.goals_progress.percentage}% Complete</p>
                  </div>
                )}
              </div>
            )}

            {expenses.length === 0 && (
              <div className="info-box">
                <p>Upload expenses to see your analytics dashboard</p>
              </div>
            )}
          </div>
        )}
      </div>

      <footer className="app-footer">
        <p>Made with love | Enhanced with Multi-Guru AI, OCR & Predictions</p>
      </footer>
    </div>
  );
}

export default App;
