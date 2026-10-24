@echo off
cd /d "%~dp0"
py -3 -m pip install -r requirements.txt
py -3 src\run_local.py
echo.
echo Upload the file on your Desktop named predictions.csv
echo at https://hackathon.ukcbt.org  step 2
pause
