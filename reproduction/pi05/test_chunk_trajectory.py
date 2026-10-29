import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from action_chunk_trajectory import ActionChunkTrajectory
class ChunkTimeTests(unittest.TestCase):
    def make(self):
        with patch('action_chunk_trajectory.time.monotonic',return_value=100.):return ActionChunkTrajectory(np.zeros(34))
    def test_30hz_actions_keep_their_time_on_a_50hz_preview(self):
        p=self.make();a=np.zeros((50,34));a[:,0]=np.arange(50);a[:,15]=np.r_[1.,np.zeros(49)];p.replace(a,start=100.)
        q,v,_=p.sample([100.,100.+1/60,100.+.8,101.])
        np.testing.assert_allclose(q[:,0],[0,.5,24,24],atol=1e-10)
        np.testing.assert_allclose(v[:,0],[30,30,0,0],atol=1e-10)
        self.assertEqual(q[1,15],1.)
    def test_yaw_stops_during_inference_hold(self):
        p=self.make();a=np.zeros((50,34));a[:,33]=2;p.replace(a,start=100.)
        q,_,rot=p.sample([100.5,101.2]);yaw=Rotation.from_quat(rot[:,[1,2,3,0]]).as_euler('xyz')[:,2]
        np.testing.assert_allclose(yaw,[1.,2*25/30]);self.assertEqual(q[-1,33],0.)
    def test_replanning_preserves_integrated_yaw(self):
        p=self.make();a=np.zeros((50,34));a[:,33]=1;p.replace(a,start=100.)
        a[:,33]=-1;p.replace(a,start=100.4)
        _,_,rot=p.sample([100.4,100.5]);yaw=Rotation.from_quat(rot[:,[1,2,3,0]]).as_euler('xyz')[:,2]
        np.testing.assert_allclose(yaw,[.4,.3])
if __name__=='__main__':unittest.main()
