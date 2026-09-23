# Build the short, wide Parakeet packing candidate

Normal SDK 10.0.204 build of isolated source 682e600d. Only the local minimum
packed-row expression changes in `Tensor<float>.RunFloatMatMulKernel` (48 for
axes >=1024, otherwise 64). Require 3,178 other Core and all 697 Data methods,
public declarations and method counts unchanged. Root inputs remain untouched.

M39's rejected complete-call screen must be closed before staging. Its candidate
is excluded. Use the current measured Core 521bae17/Data f3b9aa81 for comparison.

Run `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, and then `audit.py`. Four bounded CPU-2 jobs: SDK version,
restore, normal Release build and complete instruction inventory. This builds a
candidate; numerical and performance admission have not yet occurred.
