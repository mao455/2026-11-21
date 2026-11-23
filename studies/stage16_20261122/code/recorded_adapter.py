"""Paced replay of recorded simulation public inputs; no robot or physics runs.

Set FINGEREYE_REPLAY_EPISODE to an archived episode. Every target, phase and
6-step command must match exactly. A synchronous step occupies 6/120 seconds
before returning the archived observation, with ideal zero transport/capture
cost. Evaluation reports archived success, never new physical success. Historical
exposure clocks are explicitly renamed and cannot become current camera times.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np


class RecordedAdapter:
    def __init__(self, episode, clock=time.perf_counter, sleep=time.sleep):
        self.episode = Path(episode).resolve()
        self.clock, self.sleep = clock, sleep
        initial = json.loads((self.episode / 'public/initial.json').read_text())
        self.robot_description = initial['robot_description']
        self.responses = [json.loads(row)['observation'] for row in
                          (self.episode / 'public_responses.jsonl').read_text().splitlines()]
        self.operations = [row for row in map(json.loads,
                           (self.episode / 'agent_trace.jsonl').read_text().splitlines()) if row.get('op') in ('observe', 'step')]
        self.commands = [row for row in self.operations if row['op'] == 'step']
        if len(self.responses) != len(self.operations):
            raise ValueError('replay requires one recorded response per observe/step operation')
        if not self.commands or any(row['steps'] != 6 for row in self.commands):
            raise ValueError('all recorded commands must use steps=6 at 120Hz / 20Hz control')
        self.i = 0
        self.operation_index = 0
        self.observed = False
        self.failed = False
        self.last_step_completion = None
        self.notes = []
        self.command_error_max = 0.
        self.archive = zipfile.ZipFile(self.episode/'sensor_frames.zip') if (self.episode/'sensor_frames.zip').exists() else None
        self.index = json.loads((self.episode/'sensor_archive.json').read_text())['sha256'] if self.archive else {}
        self.original_load = np.load
        np.load = self.load_recorded_array

    def load_recorded_array(self, path, *args, **kwargs):
        if isinstance(path, (str, Path)) and not Path(path).exists():
            name = Path(path).name
            relocated = self.episode/'public'/name
            if relocated.exists():
                return self.original_load(relocated, *args, **kwargs)
            if self.archive and name in self.index:
                data = self.archive.read(name)
                if hashlib.sha256(data).hexdigest() != self.index[name]:
                    raise ValueError('recorded image SHA mismatch: ' + name)
                return self.original_load(io.BytesIO(data), *args, **kwargs)
        return self.original_load(path, *args, **kwargs)

    def observation(self, index):
        observation = json.loads(json.dumps(self.responses[index]))
        image = observation.get('image_state', {})
        for key in ('capture_start_wall_s', 'capture_done_wall_s', 'available_wall_s'):
            if key in image:
                image['recorded_' + key] = image.pop(key)
        image['recorded_clock_scope'] = 'archived simulation perf_counter, not current replay clock'
        publication = self.clock()
        # Frozen IO logging directly indexes these names. In this explicit
        # ideal-capture replay they denote publication, not measured exposure.
        for key in ('capture_start_wall_s', 'capture_done_wall_s', 'available_wall_s'):
            image[key] = publication
        image['current_clock_scope'] = 'zero-cost replay publication assumption; not camera exposure measurement'
        observation['replay_available_monotonic_s'] = publication
        observation['replay_scope'] = 'archived simulation RGB and measured-state inputs; no current camera or robot measurement'
        return observation

    def observe(self):
        if self.operation_index >= len(self.operations) or self.operations[self.operation_index]['op'] != 'observe':
            self.failed = True
            raise RuntimeError('observe outside the recorded operation sequence')
        self.observed = True
        observation = self.observation(self.operation_index)
        self.operation_index += 1
        return observation

    def step(self, q, steps=6, phase=''):
        if (not self.observed or self.i >= len(self.commands) or
                self.operation_index >= len(self.operations) or self.operations[self.operation_index]['op'] != 'step'):
            self.failed = True
            raise RuntimeError('command outside the recorded sequence')
        expected = self.commands[self.i]
        target = np.asarray(q, dtype=float)
        recorded = np.asarray(expected['target_q'], dtype=float)
        error = float(np.max(np.abs(target-recorded))) if target.shape == recorded.shape else None
        if error is not None:
            self.command_error_max = max(self.command_error_max, error)
        if steps != expected['steps'] or phase != expected['phase'] or not np.array_equal(target, recorded):
            self.failed = True
            raise ValueError('command mismatch at index %s: phase=%r expected=%r max_error=%r' %
                             (self.i, phase, expected['phase'], error))
        # The harness paces dispatch; this adapter also represents completion of
        # the six recorded physical steps, without advancing a physics engine.
        completion_due = self.clock() + steps/120.
        now = self.clock()
        while now < completion_due:
            self.sleep(completion_due-now)
            now = self.clock()
        self.last_step_completion = now
        self.i += 1
        observation = self.observation(self.operation_index)
        self.operation_index += 1
        return observation

    def note(self, text):
        self.notes.append(text)

    def evaluate(self):
        complete = (self.observed and not self.failed and self.i == len(self.commands)
                    and self.operation_index == len(self.operations))
        result = json.loads((self.episode/'result.json').read_text())
        physics = result['physics']
        archived_hold_complete = bool(physics.get('hold_physics_passed') and
                                     physics.get('hold_end_step') == sum(c['steps'] for c in self.commands)-1)
        return dict(clock='perf_counter', condition='paced_recorded_simulation_inputs_ideal_transport',
                    episode=str(self.episode), no_physics_executed=True, real_hardware_measured=False,
                    synchronous_step_duration_s=.05, archived_official_success=bool(result['success']),
                    camera_clock_scope='capture_* compatibility fields denote ideal zero-cost replay publication; historical times are recorded_*',
                    official_success=bool(complete and result['success']),
                    success_is_archived_verdict_not_new_physical_result=True,
                    stable_hold_monotonic_s=self.last_step_completion if complete and archived_hold_complete else None,
                    stable_time_scope='current replay timestamp of final recorded hold completion; no hardware hold was measured',
                    all_commands_equal=complete, commands_replayed=self.i, commands_expected=len(self.commands),
                    max_command_error=self.command_error_max, archived_physical_s=sum(c['steps'] for c in self.commands)/120.)

    def close(self):
        np.load = self.original_load
        if self.archive:
            self.archive.close()


def factory():
    path = os.environ.get('FINGEREYE_REPLAY_EPISODE')
    if not path:
        raise ValueError('FINGEREYE_REPLAY_EPISODE must name the archived simulation episode')
    return RecordedAdapter(path)
