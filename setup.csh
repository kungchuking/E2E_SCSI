#!/bin/csh

python -m virtualenv venv

# -- CUDA 12.6
pip install torch==2.1.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
python -m pip install pip==24.0
pip install -r requirements.txt
