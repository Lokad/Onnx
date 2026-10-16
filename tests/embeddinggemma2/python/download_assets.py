"""Acquire explicit pinned files using hf; verify before making a receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="store_true")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "assets.json").read_text(encoding="utf-8"))
    kind = "source" if args.source else "quantized" if args.quantized else "onnx"
    entry = manifest[kind]
    destination = args.root / entry["directory"]
    cli = Path(sys.executable).parent / ("hf.exe" if os.name == "nt" else "hf")
    command = [str(cli), "download", entry["repository"], *entry["files"], *entry.get("metadata", []),
               "--revision", entry["revision"], "--local-dir", str(destination), "--quiet"]
    subprocess.run(command, check=True)
    receipt = {"repository": entry["repository"], "revision": entry["revision"], "files": {}}
    for name in [*entry["files"], *entry.get("metadata", [])]:
        path = destination / name
        actual = sha256(path)
        if name in entry["files"] and actual != entry["files"][name]:
            raise ValueError(f"Hash mismatch: {name}: {actual}")
        receipt["files"][name] = {"sha256": actual, "bytes": path.stat().st_size}
    (destination / f"{kind}-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"PASS pinned {kind} assets: {len(receipt['files'])} files", flush=True)


if __name__ == "__main__":
    main()
