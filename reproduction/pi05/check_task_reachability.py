"""Check geometric reachability using privileged simulator state, not a VLA trial."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
args = parser.parse_args()
run = args.run_dir
state = json.loads((run / "status.json").read_text())
spec = json.loads((run / "task_spec.json").read_text())
model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
data = mujoco.MjData(model); data.qpos[:] = state["qpos"]
names = [f"right_{part}_joint" for part in
         ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]]
joints = [model.joint(n).id for n in names]
indices = model.jnt_qposadr[joints]
bounds = np.array([model.jnt_range[j] for j in joints])
site = model.site("rh_tcp").id
neutral = data.qpos[indices].copy()
b = np.array(spec["bottle_initial"])
targets = {"pregrasp": b + [-.10, 0, .02], "grasp": b + [0, 0, .02],
           "lift": np.array([b[0], b[1], 1.07]),
           "above_bin_center": np.array([*spec["bin_center"], 1.07]),
           "above_bin_inner_point": np.array([spec["bin_center"][0]-.04, spec["bin_center"][1]+.04, 1.07])}
results = {}
for name, target in targets.items():
    def residual(q):
        data.qpos[indices] = q; mujoco.mj_forward(model, data)
        rotvec = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_rotvec()
        # Dropping into an opening does not require a fixed tool orientation.
        orientation_weight = 0.02 if name.startswith("above_bin") else 0.15
        return np.concatenate([(data.site_xpos[site] - target) * 20, rotvec * orientation_weight, (q-neutral) * .003])
    solution = least_squares(residual, np.clip(neutral, bounds[:,0],bounds[:,1]),
                             bounds=(bounds[:,0]+1e-6,bounds[:,1]-1e-6),max_nfev=300)
    residual(solution.x)
    error = float(np.linalg.norm(data.site_xpos[site]-target))
    results[name] = {"target": target.tolist(), "position_error_m": error,
                     "joint_angles": solution.x.tolist(), "position_reachable_15mm": error < .015,
                     "orientation_error_rad": float(np.linalg.norm(Rotation.from_matrix(data.site_xmat[site].reshape(3,3)).as_rotvec()))}
report = {"test": "kinematic_reachability_only_not_a_robot_task_success", "right_arm": results}
(run / "reachability.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
