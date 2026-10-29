"""Contract tests for training state order, camera colors and wire actions."""
from collections import defaultdict, deque
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'src/GR00T-WholeBodyControl4OpenHLM/openpi-eval'),str(root/'src/GR00T-WholeBodyControl4OpenHLM')]
import sonic_g1_env as module
from gear_sonic.data_utils.quat_processor import QuatProcessor

class Camera:
    def __init__(self,colors):
        self.image=np.concatenate([np.full((4,5,3),c,np.uint8) for c in colors],axis=1)
    def get_latest_image(self):return self.image

class MappingTest(unittest.TestCase):
    def env(self):
        e=module.SonicG1Env.__new__(module.SonicG1Env)
        e.mock=False;e.state_dim=e.action_dim=34;e._dt=1/30
        e.vision_client=Camera([(11,22,33),(44,55,66)])
        e.wrist_client=Camera([(10,20,30),(40,50,60)])
        e._latest_body_state={'body_q_measured':list(range(29)),'base_quat_measured':[1.,0.,0.,0.]}
        e._latest_hand_state={'state_hand_left':[.25],'state_hand_right':[.75]}
        e._poll_state_zmq=lambda:None;e._state_quat_processor=QuatProcessor()
        e._prev_joint_pos=None;e._accumulated_yaw=0.
        return e
    def test_missing_stream_is_infinite_age(self):
        e=self.env();e._body_received_ns=e._hand_received_ns=None
        e.vision_client=e.wrist_client=None
        self.assertTrue(all(np.isinf(x) for x in e.observation_ages().values()))
    def test_receive_age_exposes_stalled_stream(self):
        e=self.env();e._body_received_ns=0;e._hand_received_ns=900_000_000
        e.vision_client=e.wrist_client=None
        with patch.object(module.time,'monotonic_ns',return_value=1_000_000_000):
            ages=e.observation_ages()
        self.assertAlmostEqual(ages['body'],1.);self.assertAlmostEqual(ages['hands'],.1)
    def test_training_state_and_rgb_order(self):
        e=self.env();o=e.get_observation()
        np.testing.assert_array_equal(o['state'],[*range(15,22),.25,*range(22,29),.75,*range(15),0.,0.,0.])
        np.testing.assert_array_equal(o['head_image_left'][0,0],[33,22,11])
        np.testing.assert_array_equal(o['left_wrist_image'][0,0],[30,20,10])
        np.testing.assert_array_equal(o['right_wrist_image'][0,0],[60,50,40])
    def test_yaw_actual_interval_and_recorder_ema(self):
        e=self.env()
        with patch.object(module.time,'monotonic_ns',side_effect=[0,400_000_000,500_000_000]):
            e.get_observation()
            q=Rotation.from_euler('z',.4).as_quat();e._latest_body_state['base_quat_measured']=q[[3,0,1,2]]
            self.assertAlmostEqual(float(e.get_observation()['state'][33]),.1,places=5)
            self.assertAlmostEqual(float(e.get_observation()['state'][33]),.09,places=5)
    def test_yaw_wrap_and_clip(self):
        p=QuatProcessor()
        def q(x):a=Rotation.from_euler('z',x).as_quat();return a[[3,0,1,2]]
        p.process_output_yaw_vel(q(np.pi-.01),0)
        self.assertAlmostEqual(float(p.process_output_yaw_vel(q(-np.pi+.01),100_000_000)[2]),.02,places=5)
        self.assertLessEqual(abs(float(p.process_output_yaw_vel(q(0),100_100_000)[2])),2)
    def test_action_order_and_gripper_direction(self):
        e=self.env();sent=[];e._send_action_zmq=lambda **kw:sent.append(kw)
        action=np.arange(34,dtype=np.float32);action[7]=0;action[15]=1;action[31:]=0
        e.step(action);d=sent[0]
        np.testing.assert_array_equal(d['joint_pos'],np.r_[action[16:31],action[:7],action[8:15]])
        self.assertEqual(d['action_hand_left'],0);self.assertEqual(d['action_hand_right'],1)
    def test_wire_isaac_permutation_preserves_every_joint(self):
        e=self.env();e._action_frame_buffer=defaultdict(lambda:deque(maxlen=2));e._num_frames_to_send=2;e._frame_index=0
        class Socket:
            def send(self,x):self.sent=x
        e._zmq_socket=Socket();joint=np.arange(29,dtype=np.float32)
        with patch.object(module,'pack_pose_message',side_effect=lambda data,**kw:data):
            for _ in range(2):e._send_action_zmq(joint,joint,np.array([1,0,0,0]),0,1)
        wire=e._zmq_socket.sent
        expected=[0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]
        np.testing.assert_array_equal(wire['joint_pos'][0],expected)
        np.testing.assert_array_equal(wire['joint_pos'][0][np.argsort(expected)],joint)
        self.assertEqual(wire['action_hand_right'][0],1)

if __name__=='__main__':unittest.main()
