"""Fetch official C++ headers matching the isolated TensorRT/ORT runtimes."""
from pathlib import Path
import re
import urllib.request
import time

root = Path(__file__).resolve().parents[2] / "native"

def get(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = response.read()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    destination.write_bytes(data)
    print(destination, len(data), flush=True)
    return data.decode()

queue = ["NvInfer.h", "NvInferRuntime.h", "NvInferPlugin.h", "NvInferVersion.h"]
seen = set()
while queue:
    name = queue.pop()
    if name in seen:
        continue
    seen.add(name)
    data = get("https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.13.3/include/" + name,
               root / "tensorrt/include" / name)
    queue.extend(re.findall(r'#include\s*[<"](Nv\w+\.h)[>"]', data))
get("https://raw.githubusercontent.com/onnx/onnx-tensorrt/10.13-GA/NvOnnxParser.h",
    root / "tensorrt/include/NvOnnxParser.h")
for name in ["onnxruntime_c_api.h", "onnxruntime_cxx_api.h", "onnxruntime_cxx_inline.h",
             "onnxruntime_float16.h"]:
    get("https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/include/onnxruntime/core/session/" + name,
        root / "onnxruntime/include" / name)
