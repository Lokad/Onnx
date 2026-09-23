# Short, wide Parakeet packing candidate

Copy the 420 qualified root inputs. Only the local minimum packed-row expression
inside `RunFloatMatMulKernel` changes: 48 when both matrix axes are at least 1024,
otherwise the existing 64. All three uses, including the exact-three-row tail
suppression, share this expression. Existing arithmetic, dynamic AVX512 defaults,
cache budgets, scratch cap and fallback paths remain.

The rejected M39 screen exposes M51 calls slower than M106 calls. The qualified
encoder census identifies two of twenty clips, M51/M61, below the current threshold.
This motivates testing; it does not establish speed or authorize release integration.

Run `C:/Python313/python.exe -X utf8 -B tests/parakeet/short-wide-pack-source/prepare.py`.
The artifact retains the patch, full source pins, prior failure and shape provenance.
Root product files remain untouched. Numerical and timing gates follow in fresh
namespaces before any integration.
