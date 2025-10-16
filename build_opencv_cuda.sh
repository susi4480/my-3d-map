#!/bin/bash
set -e

echo "🚀 OpenCV CUDA build start on DGX Station"

# === 依存関係のインストール ===
apt update
apt install -y build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libcanberra-gtk3-module \
    libatlas-base-dev gfortran python3-dev python3-numpy

# === ソース取得 ===
cd /workspace
if [ ! -d "opencv" ]; then
    git clone https://github.com/opencv/opencv.git
    git clone https://github.com/opencv/opencv_contrib.git
fi
cd opencv && git checkout 4.10.0 && cd ../opencv_contrib && git checkout 4.10.0

# === ビルドフォルダ作成 ===
cd /workspace
rm -rf build_opencv
mkdir build_opencv && cd build_opencv

# === CMake設定 ===
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local/opencv_cuda \
      -D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib/modules \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=/usr/bin/python3.10 \
      -D PYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
      -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include \
      -D WITH_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_ARCH_BIN=7.0 \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_TBB=ON \
      -D WITH_OPENMP=ON \
      -D WITH_IPP=ON \
      -D BUILD_SHARED_LIBS=ON \
      ../opencv

# === ビルド & インストール ===
make -j$(nproc)
make install

# === Python環境に登録 ===
echo "export PYTHONPATH=\$PYTHONPATH:/usr/local/opencv_cuda/lib/python3.10/site-packages" >> ~/.bashrc
source ~/.bashrc

echo "✅ OpenCV with CUDA build finished successfully!"
