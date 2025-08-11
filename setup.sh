#!/bin/bash

set -e

ENV_DIR="tft_venv"

# Create venv
echo "Creating virtualenv..."
python3.10 -m venv $ENV_DIR
source $ENV_DIR/bin/activate

# Install packages
echo "Installing packages from requirements.txt..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create outputs directory for metrics, plots, etc.
mkdir -p outputs

# Auto-activate venv for future shells in this directory
ACTIVATE_LINE="source $(pwd)/$ENV_DIR/bin/activate"
SHELL_RC="$HOME/.bashrc"

if ! grep -Fxq "$ACTIVATE_LINE" "$SHELL_RC"; then
    echo -e "\n# Auto-activate tft_venv when entering project directory" >> "$SHELL_RC"
    echo "if [ \"\$PWD\" = \"$(pwd)\" ]; then" >> "$SHELL_RC"
    echo "    $ACTIVATE_LINE" >> "$SHELL_RC"
    echo "fi" >> "$SHELL_RC"
    echo "Auto-activation added to $SHELL_RC"
else
    echo "Auto-activation already configured in $SHELL_RC"
fi

echo -e "\nSetup complete!"
echo -e "Virtual environment will auto-activate when you open a new shell in this project directory."
