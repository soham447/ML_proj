@echo off
echo ==========================================
echo Custom General Sign Recognition
echo ==========================================
echo.
echo Installing dependencies...
python -m pip install -r requirements.txt
echo.
echo To collect signs:
echo     python collect_sign.py
echo.
echo To train:
echo     python train_model.py
echo.
echo To recognize:
echo     python recognize.py
echo.
pause
