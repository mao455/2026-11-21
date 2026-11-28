"""Stage16 host with deferred private evaluation; original policy sandbox, physics and evaluator."""
import json
from pathlib import Path
import shutil
import time

from stage15_runtime import install

PORTABLE_CONFIGURATION = install()
import stage4_episode as original
import stage9_observation
import visual_observation

HERE = Path(__file__).resolve().parent
runtime = original.runtime
runtime.CASE_PARAMETER_RANGES['rgb_noise_std'] = (0., 20.4)
visual_observation.ObservationGuard = stage9_observation.ObservationGuard
OldPeer = runtime.Peer


class Peer(OldPeer):
    def __init__(self, command, log, env):
        if any(str(x).endswith('/workspace/worker.py') for x in command):
            for index, value in enumerate(command):
                if value == '/workspace' and index and Path(command[index - 1]).name == 'program':
                    shutil.copy2(HERE / 'initial_rgb_geometry.py', Path(command[index - 1]) / 'initial_rgb_geometry.py')
                    shutil.copy2(HERE / 'general_grasp_contacts.py', Path(command[index - 1]) / 'general_grasp_contacts.py')
                    shutil.copy2(HERE / 'mainline/hand_description.py', Path(command[index - 1]) / 'hand_description.py')
                    shutil.copy2(HERE / 'mainline/contact_regions.py', Path(command[index - 1]) / 'contact_regions.py')
                    shutil.copy2(HERE / 'mainline/contact_intent.py', Path(command[index - 1]) / 'contact_intent.py')
                    shutil.copy2(HERE / 'mainline/kinematic_candidates.py', Path(command[index - 1]) / 'kinematic_candidates.py')
                    shutil.copy2(HERE / 'mainline/joint_path.py', Path(command[index - 1]) / 'joint_path.py')
                    shutil.copy2(HERE / 'mainline/motor_feedback.py', Path(command[index - 1]) / 'motor_feedback.py')
                    shutil.copy2(HERE / 'mainline/surface_repair.py', Path(command[index - 1]) / 'surface_repair.py')
        command = [str(HERE / 'stage16_backend.py') if str(x) == str(HERE / 'round14_backend.py') else x for x in command]
        self.is_backend = any(str(x) == str(HERE / 'stage16_backend.py') for x in command)
        self.out = Path(command[command.index('--output') + 1]) if self.is_backend else None
        self.rpc, self.public_log_s = [], 0.
        self.pending_backend_request = None
        start = time.perf_counter()
        super().__init__(command, log, env)
        self.spawn_s = time.perf_counter() - start

    def receive(self, timeout=180):
        # Complex source collision meshes can take longer than five minutes
        # to cook before the first observation. The outer trial supervisor
        # still bounds the whole process tree; action RPC timeouts stay intact.
        if self.is_backend and timeout == 300:
            timeout = 1200
        return super().receive(timeout)

    def call(self, message, timeout=180):
        start = time.perf_counter()
        if self.is_backend:
            if self.pending_backend_request is not None:
                # A timed-out step may still finish rendering. Read its reply
                # before sending finish; it must never masquerade as a score.
                if message['op'] != 'finish':
                    raise RuntimeError('previous backend request still pending')
                previous = self.pending_backend_request
                recovered = self.receive(timeout)
                self.rpc.append(dict(op=previous['op'], recovered_before_finish=True,
                    reply_has_observation='observation' in recovered,
                    inclusive_s=time.perf_counter()-start))
                self.pending_backend_request = None
            # Known native render stalls exceeded180s while the simulator was
            # alive. Keep the same outer1800s budget; no physical gate changes.
            if timeout == 180:
                timeout = 360
            self.pending_backend_request = dict(message)
        answer = super().call(message, timeout)
        if self.is_backend:
            self.pending_backend_request = None
            if message['op'] == 'finish' and 'summary' not in answer:
                raise RuntimeError('backend finish response has no summary')
            self.rpc.append(dict(op=message['op'], start_wall_s=start, inclusive_s=time.perf_counter() - start))
            if 'observation' in answer:
                log_start = time.perf_counter()
                with (self.out / 'public_responses.jsonl').open('a') as stream:
                    stream.write(json.dumps(answer) + '\n')
                self.public_log_s += time.perf_counter() - log_start
            if message['op'] == 'finish':
                (self.out / 'rpc_timing.json').write_text(json.dumps(self.rpc))
        return answer

    def close(self):
        start = time.perf_counter()
        super().close()
        elapsed = time.perf_counter() - start
        if self.is_backend:
            (self.out / 'host_backend_lifecycle.json').write_text(json.dumps(dict(spawn_process_s=self.spawn_s,
                close_process_s=elapsed, public_response_log_write_s=self.public_log_s,
                note='close includes the original 15s graceful shutdown timeout if Kit does not exit; host public log is outside Peer.call inclusive')))


runtime.Peer = Peer
if __name__ == '__main__':
    raise SystemExit(runtime.main())
