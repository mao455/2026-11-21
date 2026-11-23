import tempfile
import time
import unittest
from pathlib import Path
from stage16_realtime_timing import Timeline, TimingContext, summarize


class Clock:
    def __init__(self):self.now=100.
    def __call__(self):return self.now
    def sleep(self,duration):self.now+=duration


class Adapter:
    robot_description={}
    def __init__(self,clock,block=.0):self.clock=clock;self.block=block;self.sent=[]
    def step(self,q,steps,phase):self.sent.append(self.clock());self.clock.sleep(self.block);return {'actual_q':q}
    def note(self,text):pass


class TimingTests(unittest.TestCase):
    def test_fifty_ms_period_with_blocked_transport_and_no_burst(self):
        clock=Clock();adapter=Adapter(clock,.12)
        with tempfile.TemporaryDirectory() as root:
            timeline=Timeline(Path(root)/'events.jsonl',clock);context=TimingContext(adapter,timeline,clock=clock,sleep=clock.sleep)
            context.step([0]*18);context.step([0]*18);adapter.block=0;context.step([0]*18);context.step([0]*18)
            self.assertEqual(context.overruns,2);self.assertEqual(context.missed_periods,2)
            self.assertAlmostEqual(adapter.sent[-1]-adapter.sent[-2],.05)
            self.assertTrue(all(b-a>=.05-1e-9 for a,b in zip(adapter.sent,adapter.sent[1:])))
            timeline.close()
    def test_true_blocking_sleep_is_counted(self):
        class RealAdapter:
            robot_description={}
            def step(self,q,steps,phase):time.sleep(.025);return {}
        with tempfile.TemporaryDirectory() as root:
            timeline=Timeline(Path(root)/'events.jsonl');ctx=TimingContext(RealAdapter(),timeline,period=.01)
            ctx.step([0]*18);ctx.step([0]*18)
            self.assertEqual(ctx.overruns,1);self.assertGreaterEqual(ctx.missed_periods,1);timeline.close()
    def test_missing_stable_evidence_never_passes40(self):
        rows=[dict(event='task_dispatch',monotonic_s=1),dict(event='first_command',monotonic_s=2),dict(event='evaluation_done',monotonic_s=20)]
        self.assertFalse(summarize(rows,dict(official_success=True))['official_success_within40s'])
        self.assertTrue(summarize(rows,dict(official_success=True,stable_hold_monotonic_s=19,clock='perf_counter'))['official_success_within40s'])
        with self.assertRaises(ValueError):summarize(rows,dict(official_success=True,stable_hold_monotonic_s=25,clock='perf_counter'))
    def test_invalid_commands_are_rejected_before_driver(self):
        clock=Clock();adapter=Adapter(clock)
        with tempfile.TemporaryDirectory() as root:
            timeline=Timeline(Path(root)/'events.jsonl',clock);ctx=TimingContext(adapter,timeline,clock=clock,sleep=clock.sleep)
            with self.assertRaises(ValueError):ctx.step([0]*17)
            with self.assertRaises(ValueError):ctx.step([0]*18,steps=12)
            with self.assertRaises(ValueError):ctx.step([float('nan')]*18)
            self.assertEqual(adapter.sent,[]);timeline.close()
    def test_joint_only_response_is_not_a_camera_frame(self):
        clock=Clock();adapter=Adapter(clock)
        with tempfile.TemporaryDirectory() as root:
            timeline=Timeline(Path(root)/'events.jsonl',clock);ctx=TimingContext(adapter,timeline,clock=clock,sleep=clock.sleep)
            ctx.step([0]*18)
            self.assertFalse(ctx.first_frame)
            ctx._observation({'fingereye': {'index': {'frame_id': 17}}})
            self.assertTrue(ctx.first_frame)
            self.assertEqual(sum(r['event']=='first_frame_available' for r in timeline.rows),1)
            timeline.close()

if __name__=='__main__':unittest.main()
