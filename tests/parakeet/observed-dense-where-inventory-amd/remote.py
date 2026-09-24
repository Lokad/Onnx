"""Run only the unchanged inventory job against complete reference dependencies."""
import importlib.util
from pathlib import Path

loader = importlib.util.spec_from_file_location('retained_build_monitor', Path(__file__).with_name('retained_remote.py'))
retained = importlib.util.module_from_spec(loader); loader.loader.exec_module(retained)
retained.BASE = Path(__file__).resolve().parents[1]
idle, live = retained.idle, retained.live

if __name__ == '__main__': raise SystemExit(retained.main())
