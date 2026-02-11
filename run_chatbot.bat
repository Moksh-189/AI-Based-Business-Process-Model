@echo off
echo Starting AI Process Chatbot...
if "%GEMINI_API_KEY%"=="" (
    echo [INFO] GEMINI_API_KEY not set. You will be prompted.
)
python chatbot_tester.py
pause
