import json
from pathlib import Path
import unittest
import numpy as np
from pose_stream import interpolate_target
from arm_tracking import ArmBiasCompensator
from task_metrics import point_aabb_distance

class AlignmentTests(unittest.TestCase):
    def test_resampling_preserves_time_and_discrete_gripper(self):
        a=np.zeros(34);b=np.ones(34)
        a[7]=1;b[7]=0
        sample=interpolate_target([(0.,a),(1.,b)],.4)
        np.testing.assert_allclose(sample[[0,8,16,31,33]],.4)
        self.assertEqual(sample[7],1);self.assertEqual(sample[15],0)
        np.testing.assert_array_equal(interpolate_target([(0.,a),(1.,b)],1.),b)
    def test_resampling_holds_last_pose_when_inference_waits(self):
        a=np.arange(34,dtype=float)
        np.testing.assert_array_equal(interpolate_target([(10.,a)],15.),a)
        np.testing.assert_array_equal(interpolate_target([(10.,a)],5.),a)
    def test_neck_contact_not_rejected_by_body_center_distance(self):
        self.assertEqual(point_aabb_distance([0,0,.13],[-.03,-.03,-.1],[.03,.03,.15]),0.)
        self.assertAlmostEqual(point_aabb_distance([.08,0,.13],[-.03,-.03,-.1],[.03,.03,.15]),.05)
        self.assertGreater(point_aabb_distance([0,0,.5],[-.03,-.03,-.1],[.03,.03,.15]),.3)
    def test_arm_bias_bounded_and_preserves_other_channels(self):
        c=ArmBiasCompensator();target=np.zeros(34);measured=np.zeros(34)
        target[8]=.3;target[7]=1;target[15]=.7;target[33]=.2
        for _ in range(1000):out=c.update(target,measured,.033)
        self.assertLessEqual(abs(c.bias[8]),.35)
        np.testing.assert_array_equal(out[[7,15,16,28,33]],target[[7,15,16,28,33]])
    def test_camera_fit_rigid_orthonormal_and_training_tip_region(self):
        fit=json.loads(Path(__file__).with_name('wrist_camera_visual_fit.json').read_text())
        for c in fit['cameras'].values():
            x,y=np.array(c['xyaxes']).reshape(2,3)
            self.assertAlmostEqual(np.linalg.norm(x),1.,places=6)
            self.assertAlmostEqual(np.linalg.norm(y),1.,places=6)
            self.assertAlmostEqual(np.dot(x,y),0.,places=6)
            self.assertLess(c['landmark_rms_px'],5.)
            tip=np.array(c['projected_landmarks'])[c['fit_landmark_indices']]
            self.assertTrue(np.all(tip[:,1]>240))

if __name__=='__main__':unittest.main()
