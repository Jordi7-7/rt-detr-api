#!/bin/bash

echo "Instalando dependencias principales..."
pip install -r requirements.txt

echo "Desinstalando dependencias innecesarias relacionadas con NVIDIA/CUDA..."
pip uninstall -y nvidia-cublas-cu12 \
                  nvidia-cudnn-cu12 \
                  nvidia-cuda-runtime-cu12 \
                  nvidia-cufft-cu12 \
                  nvidia-curand-cu12 \
                  nvidia-cusolver-cu12 \
                  nvidia-cusparse-cu12 \
                  nvidia-nccl-cu12 \
                  nvidia-nvjitlink-cu12 \
                  nvidia-nvtx-cu12

echo "Iniciando la aplicación..."
gunicorn --bind=0.0.0.0:5000 app:app
