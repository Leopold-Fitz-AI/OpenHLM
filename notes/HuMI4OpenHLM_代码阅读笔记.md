# HuMI4OpenHLM 代码阅读笔记

> 人类动作数据（HuMI）采集管线：HTC Vive tracker（5 个：双手/腰部 root/双脚）+ 双腕 GoPro
> → 重定向 IK 到 Unitree G1（29 DoF）→ 与 GoPro 腕部视频时间对齐 → 导出 OpenHLM 风格 episode
> → 与机器人遥操作数据混合转 LeRobot 做 VLA co-training。
>
> 根目录：`src/HuMI4OpenHLM`。以下路径均相对此目录。

---

## 1. 目录与模块组织

| 路径 | 性质 | 职责 |
|---|---|---|
| `packages/htc_interface/` | 独立 uv workspace 包（Windows 安装） | OpenVR 读 tracker 位姿、坐标变换、按 episode 记录原始 JSON、ZMQ RPC 服务 |
| `src/ikumi/` | 本仓库核心库 | IK 求解器（mink + MuJoCo）、关键点位姿容器、episode 读写、viser 可视化 |
| `src/processing/` | 本仓库核心库 | 视频/轨迹元信息建模（VideoInfo/TrajectoryInfo/DemoGroup）、关节顺序映射、插值、鱼眼去畸变与 ArUco 工具 |
| `src/scripts/` | CLI 入口（`pyproject.toml [project.scripts]` 注册） | `online-ik` / `offline-ik` / `replay` / `trim` / `export-pkl` / `run-pipeline` / `generate-final` / `visualize-final` / `backfill-final-undistort` |
| `src/scripts/processing_pipeline/` | run-pipeline 的子步骤 | group_demos → export_to_pkl → gyro → aruco → calibrations → match → timestamp 标定 → dataset_plan |
| `configs/htc/pose_mapping.yaml` | 配置 | tracker 角色 → gripper_id 映射（right_hand:0, left_hand:1, root:2, right_foot:3, left_foot:4） |
| `configs/calibration/` | 配置 | GoPro 2.7K 鱼眼内参 `gopro_intrinsics_2_7k.json`、ArUco 配置 `aruco_config.yaml` |
| `configs/prompts/` | 配置 | 每行一条语言指令的 txt，按 episode 序号循环分配 |
| `g1_description/` | 机器人模型 | Unitree G1 MJCF/URDF（拷贝自 unitree_ros，加了 damping/armature/frictionloss 默认值和可视化 site），IK 用 `g1_29dof_rev_1_0.xml` |

### htc_interface 包（Windows 侧）
- `htc/pose_common.py`：`PoseFrame.read_from_vr_system()`（L29，OpenVR 读位姿）、`_TX_OROBOT_OTRACKER`（L96，tracker 世界系→机器人系的全局变换）、`_ROLE_TO_TX_TRACKER`（L105，每种角色 tracker 安装偏置的硬编码 4x4 矩阵）、`transform_poses()`（L157）、`PoseData`（L218，ZMQ 消息体：pos/quat_wxyz 各 5 行，顺序 = `DEFAULT_ROLES` = root, left_hand, right_hand, left_foot, right_foot）。
- `htc/rpc_protocol.py`：JSON-RPC 风格 TypedDict（RpcRequest/RpcResponse/RpcError）+ `WireMessage`，ZMQ multipart 帧 `[client_id, 空帧, payload]`。
- `htc_scripts/record_pose.py`：`record-pose` 入口（见 §2）。
- `tracker_config.json`：tracker 序列号 → 角色映射。

### CLI 一览（pyproject.toml L29-38）
`offline-ik`=`scripts.offline_ik:cli`，`replay`，`online-ik`=`scripts.online_ik_viewer:cli`，`trim`（裁剪 JSON 帧区间），`export-pkl`（导出 PoseLib 风格 pkl，供全身控制侧用），`run-pipeline`，`generate-final`，`visualize-final`，`backfill-final-undistort`。

---

## 2. online-ik 流程（`uv run online-ik pick --rpc-address tcp://<win_ip>:4242`）

### 2.1 两台机器的分工
- **Windows**（连 Vive）：`record-pose --rpc.serve --output-dir ...`
  - `record_pose.py` `RawPoseRecorder.__init__`（L127）强制 Windows，`openvr.init` + `VRSystem()`。
  - 主循环 `run()`（L304）：以 `frequency=120Hz` 调 `PoseFrame.read_from_vr_system` 读所有 GenericTracker → 用 `tracker_config.json` 的 serial→role 映射 → `transform_poses()` 把位姿从 Vive 世界系变换到"机器人系"并施加每个角色的安装偏置 → 组装 `PoseData` 缓存为 `_latest_pose_data`。
  - `RpcServer`（L454，ZMQ ROUTER bind `tcp://0.0.0.0:4242`）暴露方法：`ping / get_pose_data / start_recording(prompt) / stop_recording / get_num_recorded_episodes / get_current_recording_frame_count / delete_last_episode / peak_last / is_recording`（handler 表 L287-301）。
  - **录制实际发生在 Windows 侧**：`start_recording`（L200）清空帧缓冲、记下起始时间戳和 prompt；主循环里 `_build_frame()`（L383）把每帧 5 个角色位姿追加进 `_episode_frames`；`stop_recording`（L211）先做时间/空间跳变校验（`_validate_episode` L414，dt>0.1s 或位移>0.1m 告警），然后写 `recording_<时间戳>.json`，结构为 `{"episode": [...], "prompt": ...}`，每帧含 `timestamp` + `{role}_pose{position, quaternion_wxyz}`。
  - 键盘也可独立控制（s 开始 / t 停止 / Backspace 删除），不依赖 RPC。

- **Linux/macOS**（跑 IK）：`scripts/online_ik_viewer.py`
  - `RecorderConfig`（L32）：`rpc_address`、`ui_freq=60`、`start_countdown=3s`、`prompt`/`prompt_file`、内嵌 `ViserConfig` + `IKConfig`。
  - `default_configs`（L56-89）：8 个任务名（proposal/walk/squat_pick_ground/pick-low/toss/unsheathe/pick/shelf），tyro 的 `overridable_config_cli` 解析 `online-ik pick ...` 为选 `pick` 这套 `RecorderConfig(ik=get_config("pick"))`。
  - `ZMQRecorderClient`（L110）：ZMQ DEALER + 心跳 + 单锁串行化 RPC 调用；`_run_in_client_loop`（L228）处理跨 event loop 调用。
  - `Recorder.run()`（L345）起三个协程：
    1. `_ik_loop()`（L400）：循环 `get_pose_data` 拉最新位姿 → `_decode_target_msg()`（L695）把 `pos[5]`/`quat_wxyz[5]`（顺序 root, left_hand, right_hand, left_foot, right_foot）解码成 `HandRootFootKeyPoints`（5 个 `SE3`）→ 首帧到达时 `recompute_reference_transform()` → `IKSolver.solve_one_step(target, dt)` → `visualize_ik_solution()` 推到 viser。
    2. `_ui_update_loop()`（L448）：60Hz 刷新状态机（WAITING_FOR_CONNECTION / READY_TO_RECORD / COUNTDOWN / RECORDING）、episode 计数、当前 prompt。
    3. `_align_server_state_loop()`（L498）：100ms 轮询服务器 `is_recording`，纠偏本地状态。
  - 网页 UI（viser）按钮：Start/Stop Recording、Remove Last（带确认弹窗）、Recompute Transforms；`_start_recording()`（L600）倒计时 3s → `ik_solver.reset()` + 重算参考变换 → **RPC `start_recording(prompt=...)`**（prompt 来自 `--prompt` 或 `--prompt-file` 按 episode 序号取行，L334）；`_stop_recording()`（L624）RPC `stop_recording` 拿回 `{path, frames, error}`。
  - **关键点：online IK 只做实时预览/监控，不落盘 IK 结果**。落盘的是 Windows 侧的原始 tracker 位姿 JSON；机器人关节角由后面的 `offline-ik` 重算。

### 2.2 IK 如何把 tracker 位姿重定向到 G1（`src/ikumi/solver.py`）

- 库：**mink**（基于 MuJoCo 的微分 IK，`solve_ik` + QP 求解器 `daqp`），模型 `g1_description/g1_29dof_rev_1_0.xml`（浮动基 + 29 个驱动关节，`q` 长度 36）。
- `IKSolver.__init__`（L28）构建任务集合（mink tasks）：
  - 4 个 FrameTask：`left_hand_site`/`right_hand_site`（site）、`left/right_ankle_pitch_link`（body）——对应双手双脚 tracker（L42-55）。
  - `PostureTask`（L156）：把全身关节拉向 `stand` keyframe，可对 waist_roll/pitch 单独加权。
  - `ComTask`（L188）：质心 xy 对齐双脚中点。
  - `pelvis` FrameTask（L226）：跟踪腰部 root tracker。
  - 可选 `torso_link` 直立任务（L251，`torso_upright_orientation_cost>0` 才启用）。
- 约束（limits）：`ConfigurationLimit`（关节限位，L79）、`CollisionAvoidanceLimit`（手臂-躯干、手臂-大腿、全身-地面，L301）、`VelocityLimit`（仅下肢关节限速，L347）。
- **重定向的核心在参考系归一化**，`recompute_reference_transform()`（L591）在每次开始录制前调用：
  1. `_setup_reference_from_feet`（L442）：假设录制开始时人站立且朝向机器人 home 朝向，计算 `_reference_transform`——把"此刻双脚中点位姿"映射到 G1 home 的双脚中点位姿的 SE3；同时估计左右脚 tracker 相对脚踝的安装变换（含 `foot_tracker_x_offset` 平移偏置 + 把脚 tracker 旋转投影到 yaw）。
  2. `_update_root_transform`（L520）：腰部 tracker 去除 roll/pitch 只保留 yaw，施加 `root_tracker_x_offset`，并算 `_root_z_scale_factor = standing_pelvis_height / 当前root高度`——**用 z 缩放补偿人和 G1 的身高差**。
- 每帧 `_update_task_targets()`（L611）：tracker 位姿先经 `HandRootFootKeyPoints.transformed()`（keypoints.py L20）乘 `_reference_transform`（脚再乘各自的 tracker→foot 变换），然后设置各 task 目标；若 `use_root_tracker=False`（如不用腰 tracker 的场景），pelvis 目标由双脚中点 + home 偏移合成（`_get_feet_based_root_pose` L416）；`suppress_foot_pitch=True` 时把脚目标 pitch 清零（L561）。
- `solve_one_step()`（L646）：外层 dt（默认 0.02s）内做 `inner_steps` 次 `mink.solve_ik` + `configuration.integrate_inplace`，返回 `IKSolution`（solution.py L53）：`q`（36 维）、各 frame 任务误差、COM 目标/实际、原始/变换后/实际达到的 5 个关键点位姿。
- 任务权重按任务名定义在 `ikumi/config.py` `_config_dict`（L155-254），例如 `pick`：`hand_position_cost=50`、`pelvis_xy_cost=20`、`root_tracker_x_offset=-0.22`、`foot_tracker_x_offset=0.18`、`torso_upright_orientation_cost=10` 等；`shelf` 由 `pick` 派生（`dataclasses.replace`，去掉 pelvis 位置约束、手位置权重提到 100）。

### 2.3 离线重算：`offline-ik`
`scripts/offline_ik.py`：对目录下所有 `recording*.json` 逐帧重跑同一个 `IKSolver`（第 0 帧重算参考变换；dt 用真实时间戳差；丢弃前 `trim_start_index=20` 帧去掉 reset 后的抖动），多进程并行，输出 `{"mjcf_path": ..., "episode": [...], "prompt": ...}` 到 `<input>_ik_recomputed_<时间戳>/`，并把所用 IKConfig 存成 `ik_config_used.yaml`。episode 帧含原始 target + `q` + error + transformed/realized target（`IKSolution.to_frame_dict` solution.py L147）。`replay` 可用 viser 回放这些关节轨迹。

---

## 3. run-pipeline 流程（`uv run run-pipeline data/<session>`）

入口 `scripts/run_processing_pipeline.py`，对每个 session 串行调 7 个子脚本（subprocess）。前置输入：`card0/`、`card1/` 两个 GoPro 的视频 + `*_ik_recomputed/` 的 IK JSON。

### Step 00 `group_demos.py` —— 按时间戳把视频和轨迹配成 demo
- `VideoInfo.from_file`（processing/dataset.py L25）：用 exiftool 读 `QuickTime:CameraSerialNumber`，用 `mp4_get_start_datetime`（processing/video.py：creation_time 取日期 + MP4 timecode 折算当天秒数，得到帧级精度的起始时刻）+ fps/帧数算出 `[start_timestamp, end_timestamp]`；`--gopro_timezone` 可指定 GoPro 所在时区。
- `TrajectoryInfo.from_file`（dataset.py L191）：读 JSON 首末帧时间戳；**`is_ik_processed = "mjcf_path" in data`**——靠这个字段自动跳过 Windows 原始录制，只处理 offline-ik 输出。
- 每个相机序列号**最早的一段视频被当作 gripper 标定视频**（L78-81），移入 `demos/gripper_calibration_*`。
- 双指针法配对两个相机时间重叠的视频 → `VideoPair`；再找**时间完全覆盖视频对**的轨迹（轨迹可复用，视频只用一次）→ `DemoGroup`。
- `DemoGroup.save_and_symlink()`（dataset.py L291）：视频/轨迹移入 `raw_videos/`、`raw_trajectories/`，在 `demos/demo_<时间戳>/` 下建软链；同时 `_trim_trajectory()`（L402）把轨迹裁到视频区间（头部多留 5s），存 `trimmed_trajectories/`。配不上对的视频/轨迹改名标记 `*_unpaired` / `*_unmatched*`。

### Step 01 `export_trajectories_to_pkl.py` —— PoseLib pkl
把 trimmed 轨迹按 30Hz 重采样（root 平移 interp1d、root 旋转 Slerp、关节 interp1d），关节顺序经 `get_recording_joint_idx_to_poselib()`（processing/robot.py L9，按 MJCF 关节名映射到 `EXPECTED_ACTUATED_JOINTS` 的 29 维固定顺序：双腿各 6 + 腰 3 + 双臂各 7）重排，joblib 存 `<session>.pkl`（`{episode_000: {fps, root_trans_offset, root_rot, dof}}`）。这步产物主要服务全身控制/PoseLib 侧，**不是 generate-final 的输入**。

### Step 02 `extract_gopro_gyro.py` —— 提取陀螺仪
用 Docker 镜像 `richardnai/gpmf-extract-gyro` 从每个 `raw_video.mp4` 的 GPMF 流提取 `gyro_data.csv`（相对视频起始的秒级时间戳 + gyro_x/y/z）。

### Step 03 `batch_detect_aruco.py` → `detect_aruco.py` —— ArUco 检测
对每个视频逐帧做鱼眼去畸变 + ArUco 检测定位（`cv_util.detect_localize_aruco_tags`），输出 `tag_detection.pkl`（每帧 `{"time", "tag_dict": {tag_id: 6D pose}}`）。用途有二：识别该视频属于哪只夹爪（tag id 分组，每夹爪 6 个 tag）、测量夹爪开合宽度。

### Step 04 `run_calibrations.py` → `calibrate_gripper_range.py` —— 夹爪量程标定
对标定视频的 `tag_detection.pkl`：统计各 tag 检出率确定 gripper_id（L44-60），再用 `get_gripper_width()`（cv_util.py L371，由左右手指 tag 的三维位置在 nominal_z 平面上算宽度）取全段 min/max，写 `gripper_range.json`。**这对应"人握拳/张开"的标定**——人手里拿着带 ArUco tag 的夹爪道具。

### Step 05 `match_camera_and_trajectory.py` —— 相机→手部轨迹关联
对每个 demo：用 `detect_gripper_id` 判定两个视频各是哪只手（gripper_id 0=right_hand、1=left_hand，由 `pose_mapping.yaml` 反推），然后把该手在轨迹中的位姿序列导出为 `camera_trajectory.csv`（`traj_to_pd_dataframe`，processing/trajectory.py L102；注意用的是 `realized_target`——IK 实际达到的位姿，不是原始 tracker 位姿）。识别不出的 demo 改名 `unk_gid_*` 跳过。

### Step 06 `batch_calibrate_tracker_camera_timestamp.py` → `calibrate_tracker_camera_timestamp.py` —— 时间偏移标定
核心思路：**同一刚体（手腕 tracker 与腕部 GoPro）的角速度应一致**。
- tracker 侧：由 `camera_trajectory.csv` 四元数差分算角速度（L22-47）；
- GoPro 侧：`gyro_data.csv` 经固定轴变换 `ROT_TRACKER_CAMERA`（L323）转到 tracker 坐标；
- 两路信号取模长 → 重采样到 60Hz → 3 阶 Butterworth 高通（0.25Hz）+ z-score → 互相关在 ±1s 内搜峰值（`find_optimal_time_offset` L117），得 `optimal_time_offset_seconds`，写 `timestamp_calibration.json`。相关峰值 <0.3 会告警。

### Step 07 `generate_dataset_plan.py` —— 生成 `dataset_plan.pkl`
这是 pipeline 的汇总产物（见 `DemoPlan` L354）：
1. `CameraPlan.from_video_pair()`（L132）：读 `timestamp_calibration.json` 修正两路视频的起止时间戳（**以 tracker 时间为基准**）；取交集区间，选起始时刻更接近帧网格的一路为参考，把对齐起点 snap 到帧时刻，算出两路各自的 `video_start_end` 帧区间和统一的 `episode_timestamps`（视频 fps 网格）。
2. `GripperPlan.from_traj_info_and_cam_plans()`（L257）：对 pose_mapping 里每个角色（0 右手/1 左手/2 root/3 右脚/4 左脚）用 `get_tcp_pose_interpolator`（位置线性 + 旋转 Slerp）把 `realized_target` 轨迹插值到视频帧时刻；手部的 `gripper_width` 从该腕相机 `tag_detection.pkl` 逐帧测宽、经 `gripper_range.json` 标定插值、再重采样对齐。
3. `get_joint_pos_array()`（L89）：从 IK JSON 的 `q[7:]`（去掉浮动基 7 维）插值出 29 关节角并按 `EXPECTED_ACTUATED_JOINTS` 重排到视频帧时刻。
4. prompt 从轨迹 JSON 顶层 `prompt` 字段读出。
5. 全部 `DemoPlan` 按 episode 起始时间排序，`pickle.dump` 到 `<session>/dataset_plan.pkl`。

**`dataset_plan.pkl` 是什么**：一个 `list[DemoPlan dict]`，每个元素 =
`{episode_timestamps, grippers: [{gripper_id, tcp_pose(N,6 rotvec), gripper_width(N,) 或 None, demo_start/end_pose}], cameras: [{gripper_id, video_path, video_start_end}], joint_pos(N,29), prompt}`。
即"每个 episode 在统一时间轴上的全部对齐结果"，是 generate-final 的唯一输入。

---

## 4. generate-final 流程（`uv run generate-final data/<session>`）

`scripts/processing_pipeline/generate_final.py`，读 `dataset_plan.pkl` + `demos/gripper_calibration_*/gripper_range.json`，多进程逐 episode 输出：

```text
data/<session>/final_data/
└── episode_0000/
    ├── data.json                      # info + text + data[frames]
    └── videos/
        ├── wrist_rgb_left.mp4         # 224x224、90° FOV 去畸变后
        └── wrist_rgb_right.mp4
```

逐帧逻辑（`_process_episode` L169）：
- **采样**：`downsample_stride=2` 降采样（GoPro 约 60fps → 约 30fps）；`teleop_delay_sec=0.2` 换算成 `delay_frames`，动作取未来第 `k+delay_frames` 帧（对齐遥操作的控制延迟），尾部相应截断。
- **视频**：按 `CameraPlan.video_start_end` 从原 GoPro 视频抽出对应帧，`FisheyeRectConverter`（cv_util.py L163，鱼眼→针孔 224×224/90°）逐帧去畸变，mp4v 写成腕部视频。gripper_id 0 → `wrist_rgb_right`，1 → `wrist_rgb_left`。
- **每帧 JSON 条目**（L433-453），与真实遥操作 recorder 同构：

| 字段 | 内容 | 维度 |
|---|---|---|
| `wrist_rgb_left/right` | `{video_path, frame_index}` 视频引用（不落图像本体） | — |
| `state_body` | `[root_roll, root_pitch, root_yaw_vel]` + 29 关节角 | **32** |
| `action_body` | 同上，取延迟后的未来帧 | 32 |
| `state_hand_left/right` | 归一化夹爪开合度 `clip(width/(max-min),0,1)` | 1+1 |
| `action_hand_left/right` | 未来帧开合度 > `hand_action_threshold=0.7` 的二值化 | 1+1 |

  其中 root 的 roll/pitch/yaw_vel 由 root（gripper_id=2）的 `tcp_pose`（rotvec 转欧拉 + yaw 对时间梯度）而来。
- `info` 记录图像尺寸/fps/视频编码与两路相机元信息；`text.goal/text.desc` = episode 的 prompt（录制时 RPC 传入，或 `--goal` 兜底）。
- HuMI **没有头部相机**，只有双腕视角。

---

## 5. 与 openpi4OpenHLM 训练的衔接

转换脚本：`src/openpi4OpenHLM/examples/unitree_g1/convert_g1_data_to_lerobot_multi.py`

```bash
uv run examples/unitree_g1/convert_g1_data_to_lerobot_multi.py \
    --parent_dir .. \
    --dataset_folders openpi4OpenHLM/example_demonstration/<teleop_session> \
    --humi_dataset_folders HuMI4OpenHLM/data/<humi_session>/final_data \
    --repo_name <org>/<dataset>
```

- `--dataset_folders`（机器人遥操作）与 `--humi_dataset_folders`（人类数据）走**同一套 episode 目录约定**（`episode_*/data.json` + 视频引用），分别收集后合并进同一个 LeRobotDataset（L587-597；humi 侧可用 `--humi_max_episodes_per_folder` 限数量，控制混合比例）。
- `process_episode()`（L344）逐帧：用 `EpisodeVideoReader` 按 `{video_path, frame_index}` 读腕部图像，`resize_with_pad` 到 224×224；**HuMI 缺 `rgb_left`（头相机）时用全黑图占位**（L333-339，`has_head_camera` 由首帧是否有 `rgb_left` 引用判定）。
- **维度重排**（L392-438）：HuMI/遥操作的 32 维 body（root3 + leg_l6 + leg_r6 + waist3 + arm_l7 + arm_r7）+ 双手各 1 维 → OpenPI 的 **34 维** state/actions：`arm_left(7), hand_left(1), arm_right(7), hand_right(1), leg_left(6), leg_right(6), waist(3), root(3)`。三路图像 + state + actions + task（=`text.goal`）组成 LeRobot features，`robot_type="unitree_g1"`，fps=30。
- 之后就是标准 OpenPI 流程（算归一化统计、训练 π0 等），见 `openpi4OpenHLM/README.md`。

---

## 6. 关键算法点小结

1. **IK 求解**：mink（MuJoCo 模型上的 QP 微分 IK，daqp 求解器），`solve_one_step` = 外层 dt × `inner_steps` 次子步积分；任务=双手/双脚 FrameTask + pelvis + posture + COM (+torso upright)，约束=关节限位/碰撞/下肢速度。模型 `g1_29dof_rev_1_0.xml`。
2. **重定向不是骨骼映射，而是"任务空间跟踪"**：人体只有 5 个 SE3 关键点；通过录制起始时刻的标定把 tracker 世界系对齐到 G1 home（双脚中点为锚），脚/腰 tracker 的安装偏置用标定矩阵 + 配置里的 x_offset 补偿；**身高差用 root z 缩放因子吸收**；中间关节（肘、膝等）由 posture 正则 + 关节限位 + 碰撞约束隐式决定。
3. **手部/夹爪**：人手握带 ArUco tag 的夹爪道具，GoPro 腕部视频里检测手指 tag 间距 → 量程标定（min/max）→ 归一化为连续开合度 state；动作侧以 0.7 阈值二值化成"夹爪闭合"指令。IK 不管手指（G1 29DoF 无手）。
4. **时间对齐三级**：(a) GoPro MP4 timecode 给帧级起始时刻（需时区参数）；(b) tracker 角速度 vs GoPro 陀螺仪互相关得到亚秒级偏移修正（以 tracker 时钟为基准）；(c) 视频帧网格对齐选参考相机 snap，轨迹用插值/Slerp 重采样到帧时刻。
5. **轨迹用 realized（IK 实际达到的）而非 target 位姿**做下游（camera_trajectory、tcp_pose、root 姿态），保证导出的图像-状态一致性。
6. **数据契约**：HuMI 的 `final_data` 与遥操作数据共用同一 episode schema（data.json + 视频引用 + 32 维 body + 双手标量），差异仅在没有头相机（下游黑图占位），因此能与遥操作数据无缝混合进同一 LeRobot 数据集做异构 co-training。

---

## 附：数据流总览

```text
[Windows] Vive trackers ──OpenVR──▶ record_pose.py ──ZMQ RPC──▶ [Mac/Linux] online_ik_viewer.py
                │  (record-pose 落盘原始 recording_*.json)            │  (mink IK 实时预览, 不落盘)
                ▼                                                     │
        原始 tracker JSON ──拷贝──▶ offline_ik.py ──▶ *_ik_recomputed/recording_*.json (含 q、realized_target)
                                                              │
        GoPro MP4 (card0/card1) ──────────┐                   │
                                          ▼                   ▼
                              run_processing_pipeline.py（7 步：配对/裁剪、pkl、陀螺仪、
                               ArUco、夹爪标定、手-相机匹配、时间偏移标定、汇总）
                                          │
                                          ▼
                                dataset_plan.pkl（时间轴对齐后的 DemoPlan 列表）
                                          │
                                          ▼
                                generate_final.py ──▶ final_data/episode_*/{data.json, videos/}
                                          │
                                          ▼
                convert_g1_data_to_lerobot_multi.py（+ 遥操作 episodes, 头相机黑图占位, 重排 34 维）
                                          │
                                          ▼
                              LeRobot 数据集 ──▶ OpenPI 混合训练
```
