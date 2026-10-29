#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE="$ROOT/src/GR00T-WholeBodyControl4OpenHLM/gear_sonic_deploy"
export TensorRT_ROOT="$ROOT/native/tensorrt"
export onnxruntime_ROOT="$ROOT/native/onnxruntime"
export HAS_ROS2=0 CUDA_HOME=/usr/local/cuda CUDAToolkit_ROOT=/usr/local/cuda
mkdir -p "$TensorRT_ROOT/lib" "$onnxruntime_ROOT/lib"
for library in nvinfer nvinfer_plugin nvonnxparser; do
  ln -sfn "$ROOT/native/python/tensorrt_libs/lib${library}.so.10" "$TensorRT_ROOT/lib/lib${library}.so"
done
ln -sfn "$ROOT/native/ortpython/onnxruntime/capi/libonnxruntime.so.1.22.0" "$onnxruntime_ROOT/lib/libonnxruntime.so"
ln -sfn "$ROOT/native/ortpython/onnxruntime/capi/libonnxruntime.so.1.22.0" "$onnxruntime_ROOT/lib/libonnxruntime.so.1"
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$ROOT/native/python/tensorrt_libs:$onnxruntime_ROOT/lib:$ROOT/native/ortpython/onnxruntime/capi:${LD_LIBRARY_PATH:-}"
/usr/bin/cmake -S "$SOURCE" -B "$SOURCE/build-pi05" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DTENSORRT_LIBRARY_DIR="$TensorRT_ROOT/lib" \
  -Donnxruntime_INCLUDE_DIR="$onnxruntime_ROOT/include" \
  -Donnxruntime_LIBRARY="$onnxruntime_ROOT/lib/libonnxruntime.so"
/usr/bin/cmake --build "$SOURCE/build-pi05" --target g1_deploy_onnx_ref -j8
