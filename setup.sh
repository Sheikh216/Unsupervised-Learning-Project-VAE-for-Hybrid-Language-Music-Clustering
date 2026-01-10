#!/bin/bash
# Setup script for Neural Network Project (Linux/Mac)
# This script will install all required dependencies and verify the installation

echo "========================================"
echo "Neural Network Project - Setup Script"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

echo "[1/4] Python found:"
python3 --version
echo ""

# Upgrade pip
echo "[2/4] Upgrading pip..."
python3 -m pip install --upgrade pip
echo ""

# Install requirements
echo "[3/4] Installing required packages..."
echo "This may take a few minutes..."
python3 -m pip install -r requirements.txt
echo ""

# Verify installation
echo "[4/4] Verifying installation..."
python3 -c "import numpy; print('  - NumPy:', numpy.__version__)"
python3 -c "import matplotlib; print('  - Matplotlib:', matplotlib.__version__)"
python3 -c "import sklearn; print('  - Scikit-learn:', sklearn.__version__)"
python3 -c "import tensorflow; print('  - TensorFlow:', tensorflow.__version__)"
python3 -c "import tqdm; print('  - tqdm:', tqdm.__version__)"
echo ""

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Some packages may not have installed correctly."
    echo "Please check the error messages above."
    exit 1
fi

echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "You can now run the neural network with:"
echo "  python3 main.py --dataset mnist"
echo ""
echo "Or run examples with:"
echo "  python3 examples.py"
echo ""
echo "For more information, see README.md or QUICKSTART.md"
echo ""
