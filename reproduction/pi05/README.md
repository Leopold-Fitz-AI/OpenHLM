# pi05 服务器复现

服务器使用已有 SSH 别名 `pi0.5`，工作目录为 `/mlplatform/OpenHLM-repro`。
训练及推理源码部署自本地提交 `f724202` 的 Git archive，位于该目录下的
`src/openpi4OpenHLM`；未复制本地未跟踪数据。后续已部署 G1 仿真所需的机器人客户端、
SONIC 控制器和模型；HuMI 与真实机器人硬件端未部署。

## 资源和环境

- 8 × NVIDIA A100 80GB；部署时全部已有任务占用，因此验证使用 CPU。
- 独立虚拟环境 `.venv`，Python 3.11；复制已有环境后安装项目缺失依赖，
  并升级至项目要求的 PyTorch 2.9.0，CUDA 12.6 构建。
- transformers 4.53.2 的替换文件只应用到此独立环境，使用复制模式避免修改共享缓存。
- `logs/environment-final.txt` 记录实际依赖；`uv pip check` 检查依赖一致性。
- 基础模型缓存 `/root/models/pi05/pi05_base_pytorch` 可读。CPU 冒烟训练从此基础
  模型开始，按项目逻辑将动作头从 32 维扩展至 34 维，仅验证两步训练。

公开资源从 Hugging Face 镜像下载，仓库、提交和选择范围分别记录在
`models/source_revision.json` 和 `public_data/source_revision.json`：

- 权重：`OpenHLM/OpenHLM-ckpts`，提交 `f3a109f817c65ea76a659d0b7b6f7cfff6d6825c`，
  `12tasks_12full-teleop`。
- 原始示例：`OpenHLM/OpenHLM-data`，提交 `fe7f13138d0a22a2a2b06b775e4effef9eb54903`，
  `example_demonstration`。
- 模型文件预期 SHA256：`9ce8b54690e824248a8134b81e04fe49f6b83ad885c4430b94a3feb227395b8b`。

项目 README 中的 `LEROBOT_HOME` 与锁定的 LeRobot 版本不兼容；这里使用
`HF_LEROBOT_HOME=/mlplatform/OpenHLM-repro/lerobot_home`。

## CPU 验证

```bash
ssh pi0.5
cd /mlplatform/OpenHLM-repro
bash reproduction/pi05/run_when_ready.sh
```

该脚本等待公开资源和 PyTorch 2.9.0 就绪后依次执行：依赖检查、保存版本清单、
权重 SHA256、现有 transforms/normalize 测试、官方权重推理、公开示例转换、
归一化统计和两步训练。验证结果保存在 `logs/validation.log`、
`logs/inference_cpu.json`、`logs/inference_cpu.actions.npy`。
本次分阶段运行的日志另外保存在 `logs/validation-cpu.log` 和 `logs/training-cpu.log`。
推理采用合成图像及状态，检查动作形状 `(50, 34)` 和有限数值，不衡量任务成功率。
脚本默认训练实验名 `pi05_cpu_base_smoke`；重复训练应使用新实验名或显式选择恢复，
避免覆盖已有 checkpoint。
单独重跑训练可设置新的 `EXPERIMENT_NAME`，执行 `bash reproduction/pi05/train_cpu.sh`。
训练入口默认会扩展 32 维动作头，不适合直接用已是 34 维的 OpenHLM 微调权重启动新训练。

已验证：269 个包的依赖一致性检查通过；现有 transforms/normalize 测试 13 项通过；
官方权重 SHA256 完全匹配；直接推理和 WebSocket 推理均返回有限的 `(50, 34)` 动作。
官方公开示例已转换为 4 个 episode、7,548 帧，并生成归一化统计。
合成观测下 CPU 10 步采样约 14.3 秒，不能用作 GPU 性能或任务成功率结论。
CPU 两步训练已完成，loss 分别为 `0.4919`、`0.5236`，梯度范数为 `3.37`、`6.13`。
checkpoint 位于服务器
`/mlplatform/OpenHLM-repro/src/openpi4OpenHLM/checkpoints/openhlm_example/pi05_cpu_base_smoke/2`。
两步训练仅证明训练链路可运行，不构成收敛、完整训练或论文结果复现。
该 checkpoint 已重新加载并通过推理验证，动作形状 `(50, 34)`，数值全部有效；
结果见 `logs/training_checkpoint_inference.json` 及本目录的 `results/`。

## GPU 训练

准备好空闲 GPU 后，使用基础模型在公开示例数据上启动训练：

```bash
cd /mlplatform/OpenHLM-repro
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=8 TRAIN_STEPS=30000 \
  bash reproduction/pi05/train_gpu.sh
```

脚本拒绝使用已有超过 2GB 显存分配的 GPU。多卡可指定 `0,1,2,3`，
batch size 为全局批量，需根据显存及卡数调整。使用示例数据训练只验证流程；
要复现论文结果，还需准备完整对应数据集、实验配置、训练预算和评估条件。

## 推理服务

当前服务使用 CPU，监听服务器本机 `127.0.0.1:18005`。启动命令：

```bash
cd /mlplatform/OpenHLM-repro
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=16 \
  .venv/bin/python reproduction/pi05/serve_local.py \
  --checkpoint models/12tasks_12full-teleop
```

本地建立端口转发：`ssh -N -L 18005:127.0.0.1:18005 pi0.5`，
然后 OpenPI WebSocket 客户端可连接本地 `127.0.0.1:18005`。
服务日志为 `logs/policy-server.log`。CPU 不满足实时机器人控制的延迟要求，
该 CPU 服务用于初期连通性验证；后续 GPU 服务及机器人仿真评估见下节。

## GPU 与 G1 仿真验证

GPU 3 用于 VLA 推理，GPU 4 用于两步训练验证和之后的 SONIC 控制器评估。
两步 GPU 训练实验 `pi05_gpu_smoke` 已保存 checkpoint，重新加载 GPU 推理通过。
预热后 10 次 VLA 推理平均 322ms，P95 332ms，峰值分配显存 7.12GiB；
这些结果关闭了 torch.compile，未包含相机采集或网络往返。

GPU 策略服务当前监听 `127.0.0.1:18006`，独立于旧 CPU 服务的 18005 端口：

```bash
cd /mlplatform/OpenHLM-repro
CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cpu TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=8 \
  .venv/bin/python reproduction/pi05/serve_local.py \
  --device cuda --port 18006 --checkpoint models/12tasks_12full-teleop
```

本地转发：`ssh -N -L 18006:127.0.0.1:18006 pi0.5`。

仿真使用独立 `.venv_sim`，MuJoCo 3.3.7、Pinocchio 2.7.0、NumPy 1.26.4、
CycloneDDS 0.10.2 和项目自带 Unitree SDK。原训练环境保留 MuJoCo 2.3.7。
原生依赖位于 `native/`：TensorRT 10.13.3.9、匹配的官方头文件、ONNX Runtime 1.22.0。
`prepare_native_headers.py` 和 `download_sonic.py` 获取官方资源；
`build_controller.sh` 在 `gear_sonic_deploy/build-pi05` 内构建原始 C++ 控制器。
构建修复仅涉及 TensorRT 版本识别、已移除的 nvparsers 组件和 GoogleTest 查找，未改控制策略。

```bash
# GPU 策略服务运行后，选择新的结果目录，避免覆盖历史试验。
cd /mlplatform/OpenHLM-repro
RUN_NAME=g1_sim_new_trial bash reproduction/pi05/run_sim_evaluation.sh vla
```

评估使用 GPU 3 运行 VLA、GPU 4 执行原始 SONIC TensorRT 控制器，仿真使用 EGL 离屏渲染。
DDS 接口固定为 `lo`，ZMQ 端口 20554–20558 绑定回环地址。
先发送默认姿态，将支撑在 2 秒内逐步放低到 0.78m，再撤掉支撑并落地稳定 3 秒，
检查高度和倾斜后执行 20 秒策略评估。底层为原项目的 50Hz 控制和 500Hz DDS 命令线程。
默认姿态基线已经完成撤除支撑后的 20 秒站立，未跌倒。

结果分别保存为 `simulation.json`、`client.json`、`telemetry.csv` 和 `simulation.mp4`。
初始未等待落地稳定的试验记录保留在 `logs/g1_sim_vla`；它撤支撑后约 0.56 秒跌倒，
不能作为有效的稳定站立起步策略评估。改为落地稳定后，`logs/g1_sim_vla_settled` 已完成
20 秒、431 动作步、18 次请求且未跌倒，客户端约 21.54Hz。
回看该次视频发现初始虚拟腕部相机被本体遮挡，已调整挂载后另行复测，结果见评估报告。

此评估只覆盖空地上的控制链、有限数值和站立稳定性。腕部模拟相机为近似挂载，
机器人模型为 Dex3 手，论文中的 CTAG 夹爪状态暂用张开值表示；没有瓶子、垃圾桶或任务成功判据。
因此不能据此报告抓取成功率、论文任务成功率或实机评估通过。

最终修正支撑点浮点坐标、平稳落地和相机遮挡后的同协议复测，
基线完成 20 秒、599 步；VLA 完成 20 秒、426 步、18 次请求，两组均未跌倒。
完整指标、排除试验和范围限制见 [GPU_ROBOT_REPORT.md](GPU_ROBOT_REPORT.md)。

## 瓶子抓取投放任务

后续已加入真实碰撞的瓶子、平台、开口垃圾桶和物理双指夹爪近似模型，
替换任务模式中的夹爪张开占位反馈，并加入接触、抬升、松手后稳定入桶判据。
三次官方 VLA 90 秒尝试均未完成抓取；独立脚本夹爪物理对照完整落桶，不能算作 VLA/G1 成功。
场景规格、判据、结果、录像和复测命令见 [BOTTLE_TASK_REPORT.md](BOTTLE_TASK_REPORT.md)。


## VLA 相机、观测和动作排查

后续已完成固定噪声的官方示范回放、图像/状态交叉对照，修复转向速度差分、
图像宽高比、近似投影/视野、异步动作执行、流过期检测及渲染/物理隔离。
已观察到官方策略开始闭爪接触瓶子，完整抓取投放以复测结果为准。
详细证据、剩余标定差异、结果和录像见 [OBSERVATION_ACTION_AUDIT.md](OBSERVATION_ACTION_AUDIT.md)。


## 相机与 TCP 继续对齐

本轮增加夹指投影拟合、50Hz发送、起步原点标准化及三维接触/FK诊断。
75°拟合组3次和58°标称先验组2次完整任务均未成功；定位了远离瓶子闭爪和单指提前碰瓶的问题。
新录像、全部结果、标定局限和内参导出方法见 [TCP_ALIGNMENT_REPORT.md](TCP_ALIGNMENT_REPORT.md)。
