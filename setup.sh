#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install pillow
# Install required packages
pip install ursina
pip install --prefer-binary opencv-python

pip install screeninfo

# Install mediapipe with specific platform support
# python -m pip install mediapipe-silicon # For
#  Mac M1/M2
 # For other platforms

pip install numpy

# Create assets directory if it doesn't exist
mkdir -p assets

# Download required assets
# Note: You'll need to add your own water background and fish body textures to the assets folder:
# - assets/water
# - assets/fish_body

echo "Setup complete! Make sure to add required texture files to the assets folder."
echo "To run the game, activate the virtual environment with 'source venv/bin/activate'"
echo "Then run 'python game.py'"
