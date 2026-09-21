# Actual-input Parakeet projection diagnostic

This local numerical experiment separates projection arithmetic from inherited
convolution error on two retained feature inputs. It adds unknown-shape graph
outputs, reuses the qualified managed capture binary, and requires exact
original output and optimized-node controls before interpreting any arrays.

Run `prepare.py`, `run.py`, then `analyze.py` using `C:/Python313/python.exe
-X utf8 -B` from the repository root. Targets refuse overwrites. Six fresh
encoder calls are followed by two independent reference workers (NumPy/OpenBLAS
and Torch/MKL), each projecting four captured inputs in float64.

All numerical, ownership, process, resource and library checks are retained.
The diagnostic changes no production setting, timing result or tolerance.
`artifacts/parakeet-projection-20260921/closed.json` records the final audit;
the tracked results report describes its interpretation.
