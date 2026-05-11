@echo off
echo ====================================
echo  Face Mask Detection - Starting...
echo ====================================

:: Start FastAPI in background
start "FastAPI" cmd /k "python app.py"

:: Wait 3 seconds then start Streamlit
timeout /t 3 /nobreak > nul
start "Streamlit" cmd /k "python -m streamlit run streamlit_app.py"

echo.
echo Both servers started!
echo.
echo FastAPI   -^> http://localhost:8000/docs
echo Streamlit -^> http://localhost:8501
echo.
pause
