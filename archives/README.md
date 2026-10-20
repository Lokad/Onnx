# Historical experiment sources

Closed experiment producers and retired implementations are stored in Git commit
`f11777fb91eeff930db8d5a545ffe3481e9f517f`. Their original paths are listed in
[experiments.json](experiments.json). Reports and result data remain at their
existing locations under `tests`; they describe historical code and settings.

Restore all retired files, with their original paths and exact Git bytes, into
a new isolated directory:

```powershell
python eng/restore-experiments.py --output artifacts/experiment-history
```

For a smaller selection:

```powershell
python eng/restore-experiments.py --prefix tests/e5/softmax-zero-product/ --output artifacts/softmax-history
```

The command reads local Git objects and refuses an existing destination. A
shallow clone may need `git fetch origin f11777fb91eeff930db8d5a545ffe3481e9f517f`
first. Restoring sources does not run them, install their historical dependencies
or restore ignored models/raw traces. Reproduce a complete historical environment
from that commit when required; archived producers can contain retired absolute
paths, VM identities and runtime controls.

The maintained entrypoints are documented in
[code maintenance](../docs/code-maintenance.md). For new timing work, use
[routine benchmarks](../eng/bench.md) rather than copying an old campaign.
