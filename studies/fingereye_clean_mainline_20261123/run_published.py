"""Portable publication adapter; preserves frozen policy and historical harness bytes.

Run with the Python environment containing Isaac Lab. Only static robot assets,
the interpreter installation, submitted code and public observations enter bwrap.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
LEGACY_ROBOT = '/home/CNS2025331827/docs/grasp_small_obj/FingerEyeLab/assets/xhand3camFE_left'


def sandbox_command(out):
    python = Path(sys.executable).absolute()
    command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
               '--clearenv', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
    roots = ['/usr', '/bin', '/lib', '/lib64', '/etc', sys.prefix, sys.base_prefix]
    # Never expose a checkout, output directory or home directory to the worker.
    for path in dict.fromkeys(roots):
        if Path(path).exists():
            command += ['--ro-bind', path, path]
    robot = str(LAB / 'assets/xhand3camFE_left')
    for destination in dict.fromkeys([robot, LEGACY_ROBOT]):
        command += ['--ro-bind', robot, destination]
    command += ['--ro-bind', str(out / 'program'), '/workspace',
                '--ro-bind', str(out / 'public'), str(out / 'public'),
                '--dir', '/output', '--chdir', '/workspace']
    for key, value in {'PATH': f'{python.parent}:/usr/bin:/bin', 'HOME': '/tmp',
                       'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                       'OPENBLAS_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1',
                       'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}.items():
        command += ['--setenv', key, value]
    return command + [str(python), '/workspace/worker.py', '/workspace/controller.py']


def main():
    """Use the same maintained entry in a full lab checkout."""
    from run_stage17 import main as run
    return run()


if __name__ == '__main__':
    raise SystemExit(main())
