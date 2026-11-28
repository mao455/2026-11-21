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
LAB = HERE


def sandbox_command(out):
    python = Path(sys.executable).absolute()
    command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
               '--clearenv', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
    roots = ['/usr', '/bin', '/lib', '/lib64', '/etc', sys.prefix, sys.base_prefix]
    # Never expose a checkout, output directory or home directory to the worker.
    for path in dict.fromkeys(roots):
        if Path(path).exists():
            command += ['--ro-bind', path, path]
    if os.environ.get('FINGEREYE_HAND_DESCRIPTION'):
        hand=json.loads(Path(os.environ['FINGEREYE_HAND_DESCRIPTION']).read_text())
        folders={str(Path(hand['urdf_path']).parent)}
        folders.update(str(Path(p).parent) for p in hand['source_sha256'] if Path(p).suffix.lower() in ('.stl','.obj','.dae'))
        for folder in sorted(folders):
            command += ['--ro-bind',folder,folder]
    command += ['--ro-bind', str(out / 'program'), '/workspace',
                '--ro-bind', str(out / 'public'), str(out / 'public'),
                '--dir', '/output', '--chdir', '/workspace']
    for key, value in {'PATH': f'{python.parent}:/usr/bin:/bin', 'HOME': '/tmp',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1',
                       'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}.items():
        command += ['--setenv', key, value]
    return command + [str(python), '/workspace/worker.py', '/workspace/controller.py']

