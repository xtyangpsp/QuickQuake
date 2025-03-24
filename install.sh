#!/bin/bash

# =================================================================
# COMPLETE INSTALLATION OF QuickQuake 
# =================================================================

# 1. Create a Conda environment from environment.yml
echo "Creating QuickQuake environment..."
conda env create -f environment.yml || {
    echo "failed to create the environment. Check environment.yml";
    exit 1;
}

# 2. Activate the environment 
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate QuickQuake || {
    echo "Could not activate the environment. Is it installed?";
    exit 1;
}

# 3. Install local packages in editable mode
echo "Installing PhaseNet and Gamma..."
pip install -e dependencies/PhaseNet || {
    echo " Failed to install PhaseNet. Check the dependencies/PhaseNet folder";
    exit 1;
}

pip install -e dependencies/GaMMA || {
    echo "Failed to install Gamma. Check the dependencies/GaMMA folder";
    exit 1;
}

# 4. Set permissions for scripts
echo "Setting permissions..."
chmod +x quickquake/*.py

# 5. Configure environment variables (hypoinverse should be already compiled)
echo "Configuring environment variables..."
cat <<EOT >> ~/.bashrc
# QuickQuake configuration
export QUICKQUAKE_ROOT="$(pwd)"
export PATH="$(pwd)/quickquake:\$PATH"
export HYPO_BIN="$(pwd)/dependencies/hyp1.40/source/hyp1.40"
EOT

source ~/.bashrc

# 6. Final verification
echo ""
echo "================================================================"
echo "Installation complete. Verify with:"
echo "  \$ conda activate QuickQuake"
echo "  \$ ls -l \$HYPO_BIN  # It should display the executable"
echo "================================================================"

