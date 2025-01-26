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

# pip install --prefer-binary mediapipe #might need to use system package.

pip install numpy

# Create assets directory if it doesn't exist
mkdir -p assets

echo "Setup complete! Make sure to add required texture files to the assets folder."
echo "To run the game, activate the virtual environment with 'source venv/bin/activate'"
echo "Then run 'python game.py'"
