"""The Windows I/O adapter must survive every actual model-worker boundary."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_helper"))
import launch_seeds as launch


class WindowsDispatchTests(unittest.TestCase):
    def test_every_windows_profile_and_arm_uses_in_process_runner(self):
        for vram in (12, 24, 48):
            for arm in (None, "proxy", "judge", "knn_static"):
                with self.subTest(vram=vram, arm=arm):
                    cmd = launch.worker_command(ROOT / "runtime", "recipe.yaml", "output", "full", 42,
                                                vram, arm, platform_name="nt")
                    self.assertTrue(cmd[2].endswith("run_profile.py"))
                    self.assertEqual(cmd[cmd.index("--vram-gb") + 1], str(vram))
                    if arm:
                        self.assertEqual(cmd[cmd.index("--arm") + 1], arm)
                    else:
                        self.assertNotIn("--arm", cmd)

    def test_linux_24_retains_original_cli_and_arm_bridge(self):
        cmd = launch.worker_command("runtime", "recipe", "output", "full", 42, platform_name="posix")
        self.assertEqual(cmd, launch.cli_command("runtime", "recipe", "output", "full", 42))
        arm = launch.worker_command("runtime", "recipe", "output", "full", 42, arm="judge", platform_name="posix")
        self.assertTrue(arm[2].endswith("run_arm.py"))
        self.assertNotIn("--vram-gb", arm)

    def test_legacy_identity_accepts_new_io_policy_without_rewriting_science(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(runtime=str(ROOT / "runtime"), recipe=str(ROOT / "configs/gsm8k-24gb.yaml"),
                output_root=str(root / "runs"), state_root=str(root / "state"), log_root=str(root / "logs"),
                gpus="0", dry_run=True, seeds=None, base_seed=42, stage="full", parallelism="auto", vram_gb=24)
            plan = launch.make_plan(args, {}, resolver=lambda *args: {"seed":42,"data_seed":42,"arms":launch.ARMS})
            root.joinpath("state").mkdir()
            saved = launch.check_manifest(plan)
            saved.pop("io_policy", None)
            launch.atomic_json(root / "state/launch_manifest.json", saved)
            resumed = launch.check_manifest(plan)
            self.assertEqual(resumed["identity"], saved["identity"])
            self.assertEqual(resumed["identity"]["version"], 1)
            if "io_policy" in plan:
                self.assertEqual(resumed["io_policy"], plan["io_policy"])


if __name__ == "__main__":
    unittest.main()
