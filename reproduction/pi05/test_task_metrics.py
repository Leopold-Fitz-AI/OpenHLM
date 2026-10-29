import unittest
from task_metrics import BottleBinScore


class TaskScoreTests(unittest.TestCase):
    def score(self):
        return BottleBinScore(.74, (.47, -.5), (.15, .15), .04, .62)

    def sample(self, s, t, low=(.44, -.53, .74), high=(.50, -.47, .98), contact=True,
               opening=.6, command=0., speed=0.):
        return s.update(t, low, high, contact, opening, command, speed, 0.)

    def test_full_contact_lift_release_sequence(self):
        s = self.score()
        self.sample(s, 0); self.sample(s, .2)
        self.assertTrue(s.grasped)
        self.sample(s, .3, low=(.44,-.53,.82), high=(.50,-.47,1.06))
        self.sample(s, .6, low=(.44,-.53,.82), high=(.50,-.47,1.06))
        self.assertTrue(s.lifted)
        for t in [1., 1.9]:
            self.sample(s,t,low=(.44,-.53,.04),high=(.50,-.47,.28),contact=False,opening=1.,command=1.)
        self.assertTrue(s.success)

    def test_prepositioned_bottle_is_not_a_success(self):
        s = self.score()
        for t in [0., 1., 2.]:
            self.sample(s,t,low=(.44,-.53,.04),high=(.50,-.47,.28),contact=False,opening=1.,command=1.)
        self.assertFalse(s.success)

    def test_single_finger_or_brief_contact_does_not_count(self):
        s = self.score()
        self.sample(s,0); self.sample(s,.1); self.sample(s,.2,contact=False)
        self.assertFalse(s.grasped)

    def test_on_rim_or_moving_bottle_is_not_success(self):
        s = self.score(); s.grasped = s.lifted = True
        for t in [0., 1.]:
            self.sample(s,t,low=(.44,-.53,.50),high=(.50,-.47,.74),contact=False,opening=1.,command=1.)
        self.assertFalse(s.success)
        for t in [2., 3.]:
            self.sample(s,t,low=(.44,-.53,.04),high=(.50,-.47,.28),contact=False,opening=1.,command=1.,speed=.4)
        self.assertFalse(s.success)

    def test_whole_object_must_be_inside(self):
        s = self.score(); s.grasped = s.lifted = True
        for t in [0., 1.]:
            self.sample(s,t,low=(.30,-.53,.04),high=(.50,-.47,.28),contact=False,opening=1.,command=1.)
        self.assertFalse(s.success)

    def test_resting_against_wall_allows_solver_penetration(self):
        s = self.score(); s.grasped = s.lifted = True
        for t in [0., 1.]:
            self.sample(s,t,low=(.44,-.53,.04),high=(.50,-.34996,.28),contact=False,opening=1.,command=1.)
        self.assertTrue(s.success)

    def test_a_gripper_still_supporting_the_bottle_is_not_release(self):
        s = self.score(); s.grasped = s.lifted = True
        for t in [0., 1.]:
            s.update(t,(.44,-.53,.04),(.50,-.47,.28),False,1.,1.,0.,0.,any_gripper_contact=True)
        self.assertFalse(s.success)


if __name__ == "__main__":
    unittest.main()
