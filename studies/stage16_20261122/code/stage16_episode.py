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
                    shutil.copy2(HERE / 'general_grasp_contacts.py', Path(command[index - 1]) / 'general_grasp_contacts.py')
        command = [str(HERE / 'stage16_backend.py') if str(x) == str(HERE / 'round14_backend.py') else x for x in command]
        self.is_backend = any(str(x) == str(HERE / 'stage16_backend.py') for x in command)
        self.out = Path(command[command.index('--output') + 1]) if self.is_backend else None
        self.rpc, self.public_log_s = [], 0.
        start = time.perf_counter()
        super().__init__(command, log, env)
        self.spawn_s = time.perf_counter() - start

    def call(self, message, timeout=180):
        start = time.perf_counter()
        answer = super().call(message, timeout)
        if self.is_backend:
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
