@echo off
echo 🚀 Starting Graph-Enhanced Agentic RAG System
echo ============================================

echo.
echo 1. Installing minimal requirements...
pip install streamlit sentence-transformers scikit-learn PyMuPDF pdfminer.six numpy pandas

echo.
echo 2. Testing core components...
python test_core.py

echo.
echo 3. Starting Streamlit application...
echo    Access your app at: http://localhost:8501
echo.
streamlit run app_simple.py

pause
