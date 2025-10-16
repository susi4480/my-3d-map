#!/bin/bash
set -e

echo "=============================="
echo "🔧 Open3D GPU版 (CUDA 12.4) ビルド開始"
echo "=============================="

# ---- 依存パッケージのインストール ----
apt update
apt install -y git cmake build-essential ninja-build libgl1-mesa-dev libxi-dev libxmu-dev libglu1-mesa-dev python3-dev python3-pip

# ---- 一時作業ディレクトリ ----
cd /workspace
rm -rf Open3D_build
mkdir -p Open3D_build
cd Open3D_build

# ---- Open3D 取得 ----
git clone --recursive https://github.com/isl-org/Open3D.git
cd Open3D
git checkout v0.18.0

# ---- ビルド用フォルダ ----
mkdir build && cd build

# ---- CMake 設定 ----
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_CUDA_MODULE=ON \
  -DBUILD_GUI=OFF \
  -DWITH_JUPYTER=OFF \
  -DWITH_OPENMP=ON \
  -DUSE_SYSTEM_CUDA=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DCMAKE_INSTALL_PREFIX=/usr/local/open3d_install

# ---- ビルド & インストール ----
make -j$(nproc)
make install

# ---- Pythonモジュール作成 ----
make install-pip-package -j$(nproc)
pip install /usr/local/open3d_install/lib/python_package/pip_package/open3d-*.whl

echo "=============================="
echo "✅ Open3D GPU対応版インストール完了"
echo "=============================="

# ---- 動作確認 ----
python3 - <<'PY'
import open3d as o3d
print("✅ Open3D version:", o3d.__version__)
print("✅ CUDA available:", o3d.core.cuda.is_available())
try:
    dev = o3d.core.Device("CUDA:0")
    print("✅ Tensor Device OK:", dev)
except Exception as e:
    print("❌ CUDA error:", e)
PY
