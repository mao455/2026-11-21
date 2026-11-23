import json
from pathlib import Path
import tempfile
import unittest

from recorded_adapter import RecordedAdapter


class Clock:
    def __init__(self): self.t=100.
    def __call__(self): return self.t
    def sleep(self, duration): self.t+=duration


class ReplayTests(unittest.TestCase):
    def fixture(self, root):
        p=Path(root);(p/'public').mkdir()
        (p/'public/initial.json').write_text(json.dumps(dict(robot_description={})))
        obs=dict(actual_q=[0]*18, image_state=dict(step=0,capture_start_wall_s=3.,capture_done_wall_s=4.,available_wall_s=5.))
        (p/'public_responses.jsonl').write_text('\n'.join(json.dumps(dict(observation=obs)) for _ in range(2)))
        (p/'agent_trace.jsonl').write_text(json.dumps(dict(op='observe'))+'\n'+json.dumps(dict(op='step',steps=6,phase='hold',target_q=[0]*18)))
        (p/'result.json').write_text(json.dumps(dict(success=True,physics=dict(hold_physics_passed=True,hold_end_step=5))))
        return p
    def test_exact_sequence_completes_after50ms_and_labels_old_clock(self):
        with tempfile.TemporaryDirectory() as root:
            clock=Clock();adapter=RecordedAdapter(self.fixture(root),clock,clock.sleep)
            try:
                obs=adapter.observe();self.assertEqual(obs['image_state']['capture_start_wall_s'],100.)
                self.assertIn('not camera exposure',obs['image_state']['current_clock_scope'])
                self.assertEqual(obs['image_state']['recorded_capture_start_wall_s'],3.)
                adapter.step([0]*18,6,'hold');result=adapter.evaluate()
                self.assertTrue(result['all_commands_equal']);self.assertTrue(result['official_success'])
                self.assertTrue(result['no_physics_executed']);self.assertFalse(result['real_hardware_measured'])
                self.assertAlmostEqual(result['stable_hold_monotonic_s'],100.05)
            finally:adapter.close()
    def test_tiny_command_difference_rejected_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            clock=Clock();adapter=RecordedAdapter(self.fixture(root),clock,clock.sleep)
            try:
                adapter.observe()
                with self.assertRaises(ValueError):adapter.step([1e-15]+[0]*17,6,'hold')
                self.assertFalse(adapter.evaluate()['official_success'])
            finally:adapter.close()
    def test_incomplete_replay_has_no_stable_time(self):
        with tempfile.TemporaryDirectory() as root:
            adapter=RecordedAdapter(self.fixture(root))
            try:
                adapter.observe();result=adapter.evaluate()
                self.assertFalse(result['official_success']);self.assertIsNone(result['stable_hold_monotonic_s'])
            finally:adapter.close()
    def test_recovery_can_observe_again_in_recorded_order(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.fixture(root);trace=(p/'agent_trace.jsonl').read_text()
            (p/'agent_trace.jsonl').write_text(trace+'\n'+json.dumps(dict(op='observe')))
            responses=(p/'public_responses.jsonl').read_text().splitlines()
            (p/'public_responses.jsonl').write_text('\n'.join(responses+[responses[-1]]))
            clock=Clock();adapter=RecordedAdapter(p,clock,clock.sleep)
            try:
                adapter.observe();adapter.step([0]*18,6,'hold')
                self.assertFalse(adapter.evaluate()['all_commands_equal'])
                adapter.observe();self.assertTrue(adapter.evaluate()['all_commands_equal'])
            finally:adapter.close()

if __name__=='__main__':unittest.main()
