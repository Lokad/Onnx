# Corrected M23 inventory

The initial `kernel-loop-build-amd` run compiled successfully but its comparison
failed because the control directory lacked Google.Protobuf. That run remains
closed as a harness failure. This successor reuses exactly its compiled product,
copies the complete pinned control dependencies and executes only the inventory.
No rebuild or numerical/performance worker is included.

Use `run.py prepare`, `stage`, `launch`, `observe`, `collect`, then `audit.py`
with Python 3.13 `-X utf8 -B` from the repository root. Fresh destinations are
`artifacts/pyannote-kernel-loop-inventory-amd-20260922` and
`/dev/shm/lokad-pyannote-kernel-loop-inventory-20260922`.

The one changed Kernel512 method, all 3,162 unchanged Core and 697 Data methods,
public API equality and original resource limits remain mandatory. Successful
inventory permits numerical qualification; it supplies no correctness or speed
claim by itself. The failed build's tools and all evidence stay unchanged.
