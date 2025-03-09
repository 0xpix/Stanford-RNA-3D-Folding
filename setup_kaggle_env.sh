#!/bin/bash

echo "🚀 Updating system and installing dependencies..."
sudo apt update -y
sudo apt upgrade -y

echo "🔧 Installing dependencies..."
sudo apt install -y software-properties-common build-essential zlib1g-dev libffi-dev libssl-dev \
    libsqlite3-dev libbz2-dev liblzma-dev libreadline-dev libncursesw5-dev libgdbm-dev tk-dev \
    libgdbm-compat-dev libdb-dev gcc make wget curl

echo "🐍 Installing Python 3.12.4..."
PYTHON_VERSION="3.12.4"
cd /usr/src
sudo rm -rf Python-${PYTHON_VERSION}*
sudo wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
sudo tar xvf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}
sudo ./configure --enable-optimizations
sudo make -j$(nproc)
sudo make altinstall

echo "✅ Python 3.12.4 installed!"
python3.12 --version

echo "🔗 Setting 'python' to point to 'python3'..."
sudo ln -sf /usr/local/bin/python3.12 /usr/bin/python
python --version

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo "📜 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "🚀 Installing JAX with GPU support..."
pip install jax jaxlib
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "🔧 Installing Optax, Flax, and Termcolor..."
pip install optax flax termcolor

echo "🖥 Installing NVIDIA utilities..."
sudo apt install nvidia-utils-515 -y

echo "✅ Setup complete! 🎉"
