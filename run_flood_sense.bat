@echo off
REM Go to the folder where this batch file is located
cd "%~dp0"

REM Run Streamlit app
streamlit run Flood_sense_spider.py

pause
