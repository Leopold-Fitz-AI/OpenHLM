import unittest
from task_metrics import BottleBinScore
from paper_task_metrics import PaperDisposalScore
from paper_protocol import PAPER_PROMPT
import json
from pathlib import Path

class PaperRubricTests(unittest.TestCase):
    def make(self):return PaperDisposalScore(BottleBinScore(.74,(.8,.08),(.18,.18),.04,.66,bin_radius=.18),(-.65,0),(.25,0))
    def observation(self,s,t,xy=(-.65,0),force=0,lid=0,pedal=0,yaw=0):s.update(t,xy,yaw,0.,force,lid,pedal)
    def test_legacy_grasp_and_place_cannot_complete_paper_task(self):
        s=self.make();s.mechanics.grasped=s.mechanics.lifted=s.mechanics.success=True
        self.observation(s,0);self.observation(s,1)
        self.assertEqual(s.result()['task_progress'],.5);self.assertFalse(s.success)
    def test_only_left_foot_contact_with_open_lid_counts(self):
        s=self.make();self.observation(s,0,force=0,lid=1.2,pedal=-.02);self.observation(s,1,force=0,lid=1.2,pedal=-.02)
        self.assertFalse(s.pedal_opened)
        self.observation(s,2,force=30,lid=0,pedal=-.02);self.observation(s,3,force=30,lid=0,pedal=-.02)
        self.assertFalse(s.pedal_opened)
        self.observation(s,4,force=30,lid=1.2,pedal=-.02);self.observation(s,4.3,force=30,lid=1.2,pedal=-.02)
        self.assertTrue(s.pedal_opened)
    def test_home_initialization_does_not_award_navigation(self):
        s=self.make();self.observation(s,0);self.observation(s,1)
        self.assertFalse(s.returned)
        self.observation(s,2,xy=(.25,0));self.observation(s,2.5,xy=(.25,0))
        self.assertTrue(s.approached);self.assertEqual(s.points()['approach_and_return'],0)
        self.observation(s,4);self.observation(s,4.6);self.assertEqual(s.points()['approach_and_return'],1)
    def test_four_independent_points_complete_task(self):
        s=self.make();self.observation(s,0);self.observation(s,1,xy=(.25,0));self.observation(s,1.5,xy=(.25,0))
        s.mechanics.grasped=s.mechanics.lifted=s.mechanics.success=True
        self.observation(s,2,xy=(.25,0),force=30,lid=1.2,pedal=-.02);self.observation(s,2.3,xy=(.25,0),force=30,lid=1.2,pedal=-.02)
        self.observation(s,4);self.observation(s,4.6);self.assertTrue(s.success);self.assertEqual(s.result()['paper_points_total'],4)
    def test_round_bin_rejects_square_corner(self):
        s=self.make().mechanics;s.grasped=s.lifted=True
        for t in [0,1]:s.update(t,(.94,.20,.04),(.97,.23,.28),False,1,1,0,0)
        self.assertFalse(s.success)
    def test_prompt_matches_released_training_metadata(self):
        path=Path(__file__).resolve().parents[1]/'paper_alignment/evidence/full_teleop_tasks.jsonl'
        tasks=[json.loads(x) for x in path.read_text().splitlines()]
        self.assertEqual(next(x['task'] for x in tasks if x['task_index']==2),PAPER_PROMPT)
if __name__=='__main__':unittest.main()
