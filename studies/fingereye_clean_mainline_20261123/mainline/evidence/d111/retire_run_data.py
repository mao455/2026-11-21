"""Remove explicitly named, quiescent D111 run outputs after compact evidence.

Never follows directory links, removes Git worktrees, or touches another task.
The receipt counts allocated blocks once per inode, including subdirectories.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
RUNS = ROOT / 'FingerEyeLab/outputs/d111'


def inside(path, root):
    return path == root or root in path.parents


def active_references(paths):
    found = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        probes = [process / 'cwd']
        try:
            probes.extend((process / 'fd').iterdir())
        except (OSError, PermissionError):
            pass
        for probe in probes:
            try:
                value = os.readlink(probe)
            except OSError:
                continue
            target = Path(value.removesuffix(' (deleted)'))
            if target.is_absolute() and any(inside(target, p) for p in paths):
                found.append(dict(pid=int(process.name), probe=str(probe), target=value))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    paths = [p.absolute() for p in args.paths]
    assert len(paths) == len(set(paths))
    for p in paths:
        assert p.exists() and not p.is_symlink(), p
        assert p == p.resolve() and p != RUNS and inside(p, RUNS), p
        assert not any(other != p and inside(p, other) for other in paths), p
    worktrees = subprocess.check_output(['git', 'worktree', 'list', '--porcelain'], cwd=ROOT, text=True)
    registered = [Path(s[9:]).resolve() for s in worktrees.splitlines() if s.startswith('worktree ')]
    assert not any(inside(w, p) for w in registered for p in paths), registered
    busy = active_references(paths)
    assert not busy, busy
    rows = []
    seen = set()
    for p in paths:
        entries = [p]
        if p.is_dir():
            for parent, dirs, files in os.walk(p, followlinks=False):
                assert '.git' not in dirs + files, parent
                entries.extend(Path(parent) / name for name in dirs + files)
        allocated = 0
        directories = 0
        for entry in entries:
            st = entry.lstat()
            directories += int(entry.is_dir() and not entry.is_symlink())
            identity = (st.st_dev, st.st_ino)
            if identity not in seen:
                allocated += st.st_blocks * 512
                seen.add(identity)
        rows.append(dict(path=str(p), allocated_bytes=allocated, directories=directories))
    before = shutil.disk_usage(RUNS).free
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        assert not p.exists(), p
    receipt = dict(rows=rows, removed_directories=sum(r['directories'] for r in rows),
                   allocated_bytes_removed=sum(r['allocated_bytes'] for r in rows),
                   available_before_bytes=before, available_after_bytes=shutil.disk_usage(RUNS).free,
                   open_files_before=busy, worktrees_checked=True, nested_git_checked=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
