#!/bin/bash

# -----------------------------------------------
# Purpose: Configure system paths for QuickQuake
# Works with: Base shell and Anaconda environments
# -----------------------------------------------

# Configuration variables
PROFILE_FILE='.bash_profile'  # Profile file to modify
QQ_DIR='quickquake'           # Directory with scripts

# -----------------------------------------------
# 1. Make scripts executable
# -----------------------------------------------
echo "Step 1/3: Making scripts executable..."

if [ -d "$QQ_DIR" ]; then
  cd "$QQ_DIR"
  # Safer loop (handles spaces in filenames)
  find . -maxdepth 1 -name "*.py" -exec chmod a+x {} +
  cd ..
else
  echo "Error: Directory $QQ_DIR not found!"
  exit 1
fi

# -----------------------------------------------
# 2. Configure PATH in shell profile
# -----------------------------------------------
echo "Step 2/3: Configuring system paths..."

# Get current directory using modern syntax
ROOT_DIR=$(pwd)
QQ_FULL_PATH="$ROOT_DIR/$QQ_DIR"

# Safe write to profile (creates backup first)
cp ~/"$PROFILE_FILE" ~/"${PROFILE_FILE}.bak" 2>/dev/null || true
echo -e "\n# QuickQuake PATH configuration" >> ~/"$PROFILE_FILE"
echo "export PATH=\"$QQ_FULL_PATH:\$PATH\"" >> ~/"$PROFILE_FILE"

# -----------------------------------------------
# 3. Final instructions for user
# -----------------------------------------------
echo "Step 3/3: Almost done!"

# Detect if running in Conda environment
if [ -n "$CONDA_PREFIX" ]; then
  echo -e "\n[IMPORTANT] Detected Anaconda environment:"
  echo "--------------------------------------------------"
  echo "Conda environments have isolated PATH settings."
  echo "To make scripts available in Conda environments:"
  echo "1. Deactivate current environment: conda deactivate"
  echo "2. Run: source ~/$PROFILE_FILE"
  echo "3. Reactivate your environment"
else
  echo -e "\nTo complete setup:"
  echo "source ~/$PROFILE_FILE  # Load new PATH settings"
  echo "Or simply restart your terminal"
fi

echo -e "\n[SUCCESS] Setup complete! Test with: QQ_predict.py --help"
