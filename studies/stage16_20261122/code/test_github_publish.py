"""Local bare repositories exercise publication without contacting GitHub."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import publish_github as publish


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.git(["init", "-q", "-b", "main"], self.source)
        self.git(["config", "user.name", "Local Publication Test"], self.source)
        self.git(["config", "user.email", "publication-test@example.invalid"], self.source)
        (self.source / "sentinel").write_text("initial\n")
        self.git(["add", "sentinel"], self.source)
        self.git(["commit", "-qm", "initial"], self.source)
        (self.source / "sentinel").write_text("user staged work\n")
        self.git(["add", "sentinel"], self.source)
        exp = self.source / publish.BASE
        exp.mkdir(parents=True)
        (exp / "requirements-research.txt").write_text("numpy==1.26.4\n")
        (self.source / "algorithm.py").write_text("value = 1\n")
        (self.source / "results.txt").write_text("actual existing result\n")
        self.config = {
            "author": {"name": "Publication Test", "email": "test@example.invalid"},
            "code": {"branch": "main", "files": {
                str(publish.BASE / "requirements-research.txt"):
                    str(publish.BASE / "requirements-research.txt"),
                "algorithm.py": "algorithm.py",
            }},
            "research": {"branch": "main", "files": {
                "algorithms/algorithm.py": "algorithm.py",
                "reports/results.txt": "results.txt",
            }},
        }
        for kind in ("code", "research"):
            remote = self.base / (kind + ".git")
            self.git(["init", "--bare", "-q", str(remote)])
            self.config[kind]["remote"] = str(remote)
        self.config_file = self.base / "config.json"

    def git(self, args, cwd=None):
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL)

    def execute(self, suffix, push=False):
        self.config_file.write_text(json.dumps(self.config))
        work = self.base / suffix
        command = [
            sys.executable, str(Path(publish.__file__).resolve()),
            "--source-root", str(self.source), "--config", str(self.config_file),
            "--work-root", str(work),
        ]
        if push:
            command.append("--push")
        env = dict(os.environ, GIT_AUTHOR_NAME="Local Test", GIT_COMMITTER_NAME="Local Test",
                   GIT_AUTHOR_EMAIL="test@example.invalid", GIT_COMMITTER_EMAIL="test@example.invalid")
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        receipt = work / "receipt.json"
        return result, json.loads(receipt.read_text()) if receipt.exists() else None

    def test_paths_and_symlinks(self):
        for path in ("../secret", "/absolute", ".git/config", "."):
            with self.assertRaises(ValueError):
                publish.relative(path)
        outside = self.base / "outside"
        outside.write_text("not publishable")
        (self.source / "link").symlink_to(outside)
        with self.assertRaises(ValueError):
            publish.checked_file(self.source, "link")

    def test_boundaries_and_credentials(self):
        for path in ("outputs/result.json", "STAGE11_RESULTS.txt", "capture.mp4"):
            with self.assertRaises(ValueError):
                publish.validate_files("code", {path: b"example"})
        with self.assertRaises(ValueError):
            publish.validate_files("research", {"data/rgb.npy": b"example"})
        with self.assertRaises(ValueError):
            publish.validate_files("research", {"data/raw_rgb.png": b"example"})
        publish.validate_files("research", {"studies/example/figures/paired_results.png": b"derived plot"})
        with self.assertRaises(ValueError):
            publish.validate_files("research", {"data/token.txt": b"ghp_" + b"x" * 30})

    def test_dry_run_preserves_index_and_remotes(self):
        before = self.git(["ls-files", "--stage", "-z"], self.source)
        result, receipt = self.execute("dry")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(receipt["original_index_unchanged"])
        self.assertEqual(before, self.git(["ls-files", "--stage", "-z"], self.source))
        for kind in ("code", "research"):
            self.assertFalse(receipt["repositories"][kind]["pushed"])
            self.assertEqual(self.git(["ls-remote", self.config[kind]["remote"]]), b"")

    def test_both_push_and_unchanged_repeat(self):
        result, first = self.execute("first", push=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result, second = self.execute("second", push=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for kind in ("code", "research"):
            self.assertTrue(second["repositories"][kind]["pushed"])
            self.assertEqual(first["repositories"][kind]["commit"],
                             second["repositories"][kind]["commit"])
            remote_files = self.git(["--git-dir=" + self.config[kind]["remote"],
                                     "ls-tree", "-r", "--name-only", "main"]).decode()
            self.assertEqual("reports/results.txt" in remote_files, kind == "research")

    def test_code_keeps_evaluated_source_history(self):
        source_revision = self.git(["rev-parse", "HEAD"], self.source).decode().strip()
        result, receipt = self.execute("history", push=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.git(["--git-dir=" + self.config["code"]["remote"],
                  "merge-base", "--is-ancestor", source_revision,
                  receipt["repositories"]["code"]["commit"]])

    def test_diverged_code_remote_stops_before_either_push(self):
        result, first = self.execute("initial-release", push=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.source / "algorithm.py").write_text("value = 2\n")
        self.git(["add", "algorithm.py"], self.source)
        self.git(["commit", "-qm", "independent source change"], self.source)
        before = {kind: self.git(["ls-remote", self.config[kind]["remote"]])
                  for kind in ("code", "research")}
        result, receipt = self.execute("diverged", push=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("diverged", result.stderr)
        self.assertTrue(receipt["original_index_unchanged"])
        for kind in ("code", "research"):
            self.assertEqual(before[kind], self.git(["ls-remote", self.config[kind]["remote"]]))

    def test_partial_failure_is_reported_and_recoverable(self):
        hook = Path(self.config["research"]["remote"]) / "hooks/pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        result, first = self.execute("rejected", push=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(first["repositories"]["code"]["pushed"])
        self.assertFalse(first["repositories"]["research"]["pushed"])
        hook.unlink()
        result, recovered = self.execute("recovered", push=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(r["pushed"] for r in recovered["repositories"].values()))

    def test_research_only_preserves_history_and_never_accesses_private_remote(self):
        result, _ = self.execute("seed-public", push=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        old_code = self.git(["ls-remote", self.config["code"]["remote"]])
        public = self.config["research"]["remote"]
        # The new selection intentionally omits the original public report.
        self.config["research"]["files"] = {"studies/stage16/REPORT.txt": "results.txt"}
        self.config["research"]["manifest_path"] = "studies/stage16/MANIFEST.json"
        self.config["code"]["remote"] = str(self.base / "nonexistent-private.git")
        self.config_file.write_text(json.dumps(self.config))
        work = self.base / "research-only"
        result = subprocess.run([
            sys.executable, str(Path(publish.__file__).resolve()),
            "--source-root", str(self.source), "--config", str(self.config_file),
            "--work-root", str(work), "--only", "research", "--preserve-unlisted", "--push",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads((work / "receipt.json").read_text())
        self.assertEqual(set(receipt["repositories"]), {"research"})
        self.assertTrue(receipt["repositories"]["research"]["pushed"])
        self.assertEqual(old_code, self.git(["ls-remote", str(self.base / "code.git")]))
        tree = self.git(["--git-dir=" + public, "ls-tree", "-r", "--name-only", "main"]).decode()
        self.assertIn("reports/results.txt", tree)
        self.assertIn("studies/stage16/REPORT.txt", tree)
        self.assertNotIn("sentinel", tree)
        self.assertTrue(receipt["original_index_unchanged"])

    def test_missing_input_fails_before_any_push(self):
        (self.source / "results.txt").unlink()
        result, receipt = self.execute("missing", push=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(receipt)
        for kind in ("code", "research"):
            self.assertEqual(self.git(["ls-remote", self.config[kind]["remote"]]), b"")

    def test_historical_manifest_parent_dependency(self):
        name = str(publish.BASE / "versions/visual_stage9_early_v9/runtime_manifest.json")
        path = self.source / name
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"sha256": {"../algorithm.py": "old"}}))
        self.config["code"]["files"][name] = name
        _, payloads = publish.snapshots(self.source, self.config)
        manifest = json.loads(payloads["code"][name])
        self.assertEqual(manifest["sha256"]["../algorithm.py"],
                         publish.hashlib.sha256(b"value = 1\n").hexdigest())

    def test_explicit_selection_overrides_ignore_rules(self):
        (self.source / ".gitignore").write_text("algorithm.py\n")
        self.config["code"]["files"][".gitignore"] = ".gitignore"
        result, receipt = self.execute("ignored", push=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        files = self.git(["--git-dir=" + self.config["code"]["remote"],
                          "ls-tree", "-r", "--name-only", "main"]).decode()
        self.assertIn("algorithm.py", files)

    def test_canonical_obj_keeps_blank_eof_bytes(self):
        # Canonical trimesh export has a blank final line; its factory checks bytes.
        original = b"# canonical mesh\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n\n"
        mesh = self.source / "assets" / "mesh.obj"
        mesh.parent.mkdir()
        mesh.write_bytes(original)
        self.config["code"]["files"]["assets/mesh.obj"] = "assets/mesh.obj"
        result, receipt = self.execute("obj", push=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        published = self.git(["--git-dir=" + self.config["code"]["remote"],
                              "show", "main:assets/mesh.obj"])
        self.assertEqual(published, original)
        self.assertTrue(receipt["original_index_unchanged"])

    def test_text_whitespace_still_rejects_before_push(self):
        (self.source / "algorithm.py").write_bytes(b"value = 1  \n")
        result, receipt = self.execute("whitespace", push=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(receipt["original_index_unchanged"])
        for kind in ("code", "research"):
            self.assertEqual(self.git(["ls-remote", self.config[kind]["remote"]]), b"")

    def test_crlf_evidence_keeps_original_bytes(self):
        original = b"case,success\r\nwedge,True\r\n"
        (self.source / "sample.csv").write_bytes(original)
        self.config["research"]["files"]["data/sample.csv"] = "sample.csv"
        result, receipt = self.execute("csv", push=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        published = self.git(["--git-dir=" + self.config["research"]["remote"],
                              "show", "main:data/sample.csv"])
        self.assertEqual(published, original)


if __name__ == "__main__":
    unittest.main()
