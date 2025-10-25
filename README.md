# Financial Advisor & Expense Manager AI Agent

An AI-powered financial advisor and expense management system that provides personalized financial guidance and automated expense tracking. The project combines financial data analysis, OCR-based extraction, and AI-driven advisory systems to help users understand, manage, and optimize their personal finances.

# Project Overview

1. This system serves as a comprehensive personal finance assistant, helping users to:
2. Automatically extract expenses from payment screenshots, bank statements, and Splitwise data.
3. Analyze spending habits and categorize expenses intelligently.
4. Generate personalized financial advice based on established principles from top financial experts.
5. Create budget recommendations, spending summaries, and goal-based planning insights.
6. Allow users to upload financial books, articles, and documents for AI-driven synthesis of advice tailored to their unique financial patterns.

# Project Demo 
https://drive.google.com/drive/folders/1Un1Wsu3ganXEa4Y4WwzcKQpiVdq-KG89?usp=sharing

# Key Features

1. Automated Expense Extraction – Uses OCR to extract text and transaction data from payment screenshots, receipts, and statements.
2. Expense Categorization – Classifies expenses into categories such as food, entertainment, transport, and more.
3. Financial Data Integration – Supports data from Splitwise, UPI payments, and CSV uploads.
4. Personalized Financial Advice – Provides AI-generated financial insights inspired by popular financial philosophies (e.g., Warren Buffett, Robert Kiyosaki, Ramit Sethi).
5. Budgeting & Analytics – Visual dashboards for spending analysis, budgeting, and financial summaries.
6. Content Uploads – Users can upload PDFs or articles for financial content processing.
7. Indian Context Support – Handles Indian banking formats, UPI transactions, and investment suggestions (SIP, ELSS, tax-saving plans).

# Technologies Used

1. Frontend:
Streamlit / React (for interactive financial dashboards)
Chart.js / Matplotlib (for expense visualization)

2. Backend:
Python
LangChain (for AI orchestration)
Pandas (for financial data processing)
OCR Libraries (Tesseract / Google Vision API / Amazon Textract)

3. Database:
SQLite / PostgreSQL for storing expense data and user preferences

4. AI/LLM Integration:
OpenAI GPT or Google Gemini models for generating financial insights and advice

5. Other Tools & Libraries:
PyPDF2, python-docx for financial book/document processing
Regular Expressions for transaction message parsing
Splitwise API for expense sharing analysis

# System Workflow

1. **Upload & Extraction**  
   - Users upload payment screenshots or receipts through the frontend.  
   - Python OCR script (`extract_expenses.py`) extracts transaction data.

2. **Expense Categorization**  
   - Extracted data is categorized automatically.  
   - Categorized data is sent back to the frontend.

3. **Visualization**  
   - Users view all extracted transactions in a clean React interface.  
   - Expenses are grouped by category for easy analysis.


# How to Run the Project

### 1. Backend (NodeJS + Python)
```bash
cd expenses-ocr-backend
npm install
# Start backend server
node index.js
```
Make sure the following modules are installed
```bash
npm install express multer cors
npm install --save-dev nodemon
pip install opencv-python pytesseract pandas numpy
```

2. Frontend (React)
```bash
cd expenses-ocr-frontend
npm install
npm start
```
The frontend will run on http://localhost:3000 by default and communicate with the backend API.


# Example Use Cases

Upload a Google Pay screenshot and get your transaction auto-recorded and categorized.
Link Splitwise data to view your shared group expenses.
Upload a financial book (PDF) and receive simplified summaries and practical budgeting principles.
Get personalized spending insights and monthly savings recommendations.

# Team Members
Project developed by a team of three Computer Science Engineering students from RV Institute of Technology and Management, working collaboratively on all modules including AI development, OCR processing, and financial data analysis.

## Team Members

| Name              | GitHub                              | Email ID                                               |
|-------------------|-------------------------------------|--------------------------------------------------------|
| Sowmya P R        | https://github.com/2406-Sowmya      | srsb2406@gmail.com                                     |
| Aishwarya R       | https://github.com/AISHWARYA251166  | ar2573564@gmail.com    |
| Rishitha Rasineni | https://github.com/rishitha-1612    | rishitharasineni@gmail.com                             |

