@echo off
REM Setup script for Neural Network Project
REM This script will install all required dependencies and verify the installation

echo ========================================
echo Neural Network Project - Setup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python found:
python --version
echo.

REM Upgrade pip
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo [3/4] Installing required packages...
echo This may take a few minutes...
python -m pip install -r requirements.txt
echo.

REM Verify installation
echo [4/4] Verifying installation...
python -c "import numpy; print('  - NumPy:', numpy.__version__)"
python -c "import matplotlib; print('  - Matplotlib:', matplotlib.__version__)"
python -c "import sklearn; print('  - Scikit-learn:', sklearn.__version__)"
python -c "import tensorflow; print('  - TensorFlow:', tensorflow.__version__)"
python -c "import tqdm; print('  - tqdm:', tqdm.__version__)"
echo.

if errorlevel 1 (
    echo.
    echo WARNING: Some packages may not have installed correctly.
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo You can now run the neural network with:
echo   python main.py --dataset mnist
echo.
echo Or run examples with:
echo   python examples.py
echo.
echo For more information, see README.md or QUICKSTART.md
echo.
pause
