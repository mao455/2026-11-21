"""Observation-only program worker. Physics and scoring live outside this process."""
from __future__ import annotations

import importlib.util
import json
import resource
import sys
import traceback


class Context:
    def __init__(self, incoming, outgoing, initial):
        self._incoming, self._outgoing = incoming, outgoing
        self._observation = initial['observation']
        self.robot_description = initial['robot_description']

    def _rpc(self, request):
        self._outgoing.write(json.dumps(request, allow_nan=False) + '\n')
        self._outgoing.flush()
        line = self._incoming.readline()
        if not line:
            raise RuntimeError('host disconnected')
        response = json.loads(line)
        if not response.get('ok', True):
            raise RuntimeError(response.get('error', 'request rejected'))
        return response

    def observe(self):
        response = self._rpc({'op': 'observe'})
        self._observation = response['observation']
        return self._observation

    def step(self, target_q, steps=1, phase='move'):
        request = {'op': 'step', 'target_q': [float(v) for v in target_q],
                   'steps': int(steps), 'phase': str(phase)}
        self._observation = self._rpc(request)['observation']
        return self._observation

    def note(self, text):
        self._rpc({'op': 'note', 'text': str(text)})


def main():
    # Prevent a generated controller from consuming unbounded CPU/memory/files.
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    incoming, outgoing = sys.stdin, sys.stdout
    sys.stdout = sys.stderr
    initial = json.loads(incoming.readline())
    ctx = Context(incoming, outgoing, initial)
    try:
        spec = importlib.util.spec_from_file_location('submitted_controller', sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(ctx)
        ctx._rpc({'op': 'complete'})
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        outgoing.write(json.dumps({'op': 'controller_error', 'error': error}) + '\n')
        outgoing.flush()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
