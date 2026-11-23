#!/usr/bin/env python3
"""Publish explicitly selected files to two isolated Git repositories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import tempfile

BASE = Path("FingerEyeLab/experiments/rax_dexbin_tinyobj")
SECRET = re.compile(
    rb"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{35,}"
    rb"|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|AKIA[A-Z0-9]{16})"
)
FORBIDDEN_PARTS = {".git", "__pycache__", ".venv", "node_modules"}
CODE_FORBIDDEN_PARTS = {"outputs", "archive", "data", "datasets", "logs"}
MEDIA = {".mp4", ".avi", ".mkv", ".npy", ".npz", ".zip", ".tar", ".gz", ".pt"}


def run(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def relative(name):
    path = PurePosixPath(name)
    if not name or str(path) == "." or path.is_absolute() or any(p in ("..", ".git") for p in path.parts):
        raise ValueError("Unsafe relative path: " + name)
    return path


def checked_file(root, name):
    path = root.joinpath(*relative(name).parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source escapes repository: " + name)
    return path.read_bytes()


def validate_files(kind, files):
    for name, blob in files.items():
        path = relative(name)
        if set(path.parts) & FORBIDDEN_PARTS or path.suffix.lower() in MEDIA:
            raise ValueError("Excluded file: " + name)
        if kind == "code":
            stage_report = path.name.startswith("STAGE") and not path.name.endswith("_OBJECTS.json")
            if set(path.parts) & CODE_FORBIDDEN_PARTS or stage_report:
                raise ValueError("Research evidence in code repository: " + name)
        elif (path.name != ".gitignore"
              and path.suffix.lower() not in (".py", ".json", ".csv", ".txt", ".md")
              and not ("figures" in path.parts and path.suffix.lower() in (".png", ".pdf"))):
            raise ValueError("Unsupported research selection: " + name)
        if len(blob) > (20 if kind == "code" else 5) * 1024 * 1024:
            raise ValueError("File exceeds publication limit: " + name)
        if path.suffix.lower() in (".json", ".csv", ".txt", ".md", ".py") and SECRET.search(blob):
            raise ValueError("Credential-like contents detected: " + name)
    if kind == "research" and sum(map(len, files.values())) > 20 * 1024 * 1024:
        raise ValueError("Research snapshot exceeds 20 MiB")


def digest(files):
    return hashlib.sha256(
        b"".join(n.encode() + b"\0" + hashlib.sha256(b).digest()
                 for n, b in sorted(files.items()))
    ).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def snapshots(root, config):
    code = {name: checked_file(root, source) for name, source in config["code"]["files"].items()}
    code["FingerEyeLab/assets/skybox/.gitkeep"] = b""
    requirements = str(BASE / "requirements-research.txt")
    lines = code[requirements].decode().splitlines()
    for dependency in config["code"].get("additional_dependencies", []):
        if dependency not in lines:
            lines.append(dependency)
    code[requirements] = ("\n".join(lines) + "\n").encode()
    source_revision = run(["git", "rev-parse", "HEAD"], root)
    for version in ("visual_stage9_early_v9", "visual_stage10_feedback_v10"):
        name = str(BASE / "versions" / version / "runtime_manifest.json")
        if name in code:
            manifest = json.loads(code[name])
            manifest["sha256"] = {
                p: hashlib.sha256(code[str(relative(posixpath.normpath("FingerEyeLab/" + p)))]).hexdigest()
                for p in manifest["sha256"]
            }
            manifest["source_runtime_revision"] = source_revision
            manifest["note"] = (
                "Original controller bytes; dependency hashes describe this code release, "
                "not the complete historical runtime."
            )
            code[name] = encoded(manifest)
    research = {
        name: checked_file(root, source)
        for name, source in config["research"]["files"].items()
    }
    validate_files("code", code)
    validate_files("research", research)
    snapshot_id = hashlib.sha256((digest(code) + digest(research)).encode()).hexdigest()
    common = dict(source_revision=source_revision, snapshot_id=snapshot_id)
    runtime_name = str(BASE / "runtime_manifest.json")
    code[runtime_name] = encoded(dict(
        **common, recommended=config.get("recommended", dict(profile="fast", fingers="auto", max_attempts=3)),
        evaluation=config.get("evaluation"),
        timing=dict(physics_hz=120, control_hz=20, camera_fps=30, video_fps=30),
        sha256={p: hashlib.sha256(b).hexdigest()
                for p, b in sorted(code.items()) if p != "README.md"},
    ))
    readme_source = config["research"].get("readme_source")
    research["README.md"] = checked_file(root, readme_source) if readme_source else (
        "# FingerEyeLab 算法与研究分析选集\n\n"
        "本仓库按明确清单发布部分算法和结构化结果。完整代码、实验源码提交与"
        "双仓库快照号见 MANIFEST.json；本地原始图像、录像和大日志不上传。\n"
    ).encode()
    research[".gitignore"] = b"__pycache__/\n*.pyc\n*.mp4\n*.npy\n.env\n"
    research["MANIFEST.json"] = encoded(dict(
        **common, complete_code_repository=config["code"]["remote"],
        complete_code_branch=config["code"]["branch"],
        files={
            name: dict(source=config["research"]["files"][name], bytes=len(blob),
                       sha256=hashlib.sha256(blob).hexdigest())
            for name, blob in sorted(research.items())
            if name in config["research"]["files"]
        },
    ))
    validate_files("code", code)
    validate_files("research", research)
    return common, {"code": code, "research": research}


def research_snapshot(root, config):
    """Explicit additive research export; never fetch private source history."""
    spec = config["research"]
    files = {name: checked_file(root, source) for name, source in spec["files"].items()}
    validate_files("research", files)
    common = dict(source_revision=run(["git", "rev-parse", "HEAD"], root),
                  snapshot_id=digest(files))
    manifest = spec.get("manifest_path", "MANIFEST.json")
    relative(manifest)
    if manifest in files:
        raise ValueError("Manifest conflicts with an explicit publication file")
    files[manifest] = encoded(dict(
        **common, publication_scope="research-only explicit files; private history/assets excluded",
        files={name: dict(source=spec["files"][name], bytes=len(blob),
                         sha256=hashlib.sha256(blob).hexdigest())
               for name, blob in sorted(files.items())},
    ))
    validate_files("research", files)
    return common, {"research": files}


def prepare(destination, spec, files, message, author=None, source_root=None, preserve_unlisted=False):
    remote, branch = spec["remote"], spec["branch"]
    exists = bool(run(["git", "ls-remote", "--heads", remote, branch]))
    if exists:
        run(["git", "clone", "--quiet", "--single-branch", "--branch", branch,
             remote, str(destination)])
    else:
        destination.mkdir()
        run(["git", "init", "-q", "-b", branch], destination)
        run(["git", "remote", "add", "origin", remote], destination)
        run(["git", "ls-remote", remote])
    if author:
        run(["git", "config", "user.name", author["name"]], destination)
        run(["git", "config", "user.email", author["email"]], destination)
    if source_root is not None:
        # Keep evaluated source commits reachable in the code release.
        # Divergent source and remote histories require reconciliation first.
        revision = run(["git", "rev-parse", "HEAD"], source_root)
        run(["git", "fetch", "--quiet", str(source_root), revision], destination)
        if not exists:
            run(["git", "checkout", "-q", "-B", branch, "FETCH_HEAD"], destination)
        elif subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", revision],
                            cwd=destination).returncode == 0:
            run(["git", "merge", "--ff-only", "--quiet", revision], destination)
        elif subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"],
                            cwd=destination).returncode != 0:
            raise RuntimeError("Code source and remote diverged; reconcile before publishing")
    for name in run(["git", "ls-files"], destination).splitlines():
        if name not in files and not preserve_unlisted:
            (destination / name).unlink()
    for name, blob in files.items():
        path = destination.joinpath(*relative(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    run(["git", "add", "-f", "-A"], destination)
    if not preserve_unlisted and set(run(["git", "ls-files"], destination).splitlines()) != set(files):
        raise RuntimeError("Staged tree differs from explicit publication selection")
    # Preserve canonical trimesh OBJ bytes (including its blank EOF). Text files
    # still receive all whitespace checks; CSV CRLF bytes remain intentional.
    run(["git", "-c", "core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol",
         "diff", "--cached", "--check", "--", ".", ":(exclude,glob)**/*.obj"], destination)
    if run(["git", "status", "--porcelain"], destination):
        run(["git", "commit", "-q", "-m", message], destination)
    return run(["git", "rev-parse", "HEAD"], destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("github_publish.json"))
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--push", action="store_true", help="Push and verify selected repositories")
    parser.add_argument("--only", choices=("both", "research"), default="both",
                        help="Research-only exports do not access or push the private code repository")
    parser.add_argument("--preserve-unlisted", action="store_true",
                        help="Keep existing public files outside the explicit research selection")
    args = parser.parse_args()
    root = args.source_root.resolve()
    config = json.loads(args.config.read_text())
    before = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    if args.preserve_unlisted and args.only != "research":
        parser.error("--preserve-unlisted requires --only research")
    common, payloads = (research_snapshot(root, config) if args.only == "research"
                        else snapshots(root, config))
    work = args.work_root or Path(tempfile.mkdtemp(prefix="fingereye-dual-publish-"))
    work.mkdir(parents=True, exist_ok=True)
    receipt_path = args.receipt or work / "receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(**common, push_requested=args.push, repositories={})
    try:
        # Prepare and validate both complete commits before either push.
        for kind in payloads:
            spec = config[kind]
            commit = prepare(
                work / kind, spec, payloads[kind],
                "Sync " + kind + " snapshot " + common["snapshot_id"][:12],
                author=config.get("author"),
                source_root=root if kind == "code" else None,
                preserve_unlisted=args.preserve_unlisted,
            )
            receipt["repositories"][kind] = dict(
                remote=spec["remote"], branch=spec["branch"], commit=commit,
                files=len(payloads[kind]), bytes=sum(map(len, payloads[kind].values())),
                pushed=False, worktree=str(work / kind),
            )
        if args.push:
            failures = []
            for kind in payloads:
                spec = config[kind]
                try:
                    run(["git", "push", "origin", "HEAD:refs/heads/" + spec["branch"]], work / kind)
                    remote_head = run(["git", "ls-remote", "--heads", spec["remote"], spec["branch"]])
                    if remote_head.split()[0] != receipt["repositories"][kind]["commit"]:
                        raise RuntimeError("Remote verification failed: " + kind)
                    receipt["repositories"][kind]["pushed"] = True
                except Exception as exc:
                    failures.append(kind)
                    receipt["repositories"][kind]["error"] = str(exc)
            if failures:
                raise RuntimeError("Not fully synchronized: " + ", ".join(failures))
    finally:
        after = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
        receipt["original_index_unchanged"] = before == after
        receipt_path.write_bytes(encoded(receipt))
        print(json.dumps(dict(receipt=str(receipt_path), **receipt), ensure_ascii=False, indent=2))
        if before != after:
            raise RuntimeError("Source index changed during publication")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
