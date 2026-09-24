# Reconcile compiler identifiers without another build

The first M64 build compiled successfully, but its scope wrapper refused the
generated-method census because adding the internal graph field renumbered
existing lambda, closure and iterator identifiers. The original worker, outputs
and failed check remain under `artifacts/parakeet-prepared-recurrence-build-amd-20260924`.
All four jobs exited zero; the supervisor exited one during its inventory review.
No inference or performance measurement occurred.

This separate review operates solely on those retained bytes. `renames.py`
reconciles explicit compiler identifiers inside ComputationalGraph. It changes
no instruction opcode, offset, operand other than those identifiers, local,
exception region, implementation flag or public interface. The two ResolveInputs
overloads require simultaneous substitution: a new ordinal collides with another
overload's previous ordinal. Every one of the 28 renamed bodies must match its
original exactly after identifier substitution. Seven additional callers differ
only by references to these identifiers. Every implementation flag stays exact.

During local review setup, an extra assertion expected a `job` field from the
later capture monitor. The retained build monitor identifies jobs by separate
log paths and has no such field. That setup attempt raised `KeyError: 'job'`
before writing either closure or starting work; the reviewer now uses the
original build protocol's resource checker and all original bounds unchanged.

The resulting inventory is passed through the original frozen `checks.py`
unchanged. Its original scope accepts exactly ten modified existing methods,
61 new internal cache/record/accessor methods, 3,179 unchanged Core methods and
all 697 unchanged Data methods. The ordered projection and existing panel kernels
stay exact; public interfaces are unchanged. This is compiled scope only.

Run `C:/Python313/python.exe -X utf8 -B` with `test_renames.py`, then `audit.py`
from the repository root. Four mutation tests use the actual inventory and
reject changed generated/caller instructions, changed flags, public-interface
drift and unrelated added helpers. `audit.py` independently checks all raw
resources, input/product identities and terminal ownership, closes the original
refusal, and writes a separate successful review closure. It never rebuilds,
reruns or edits a product, reobserves a closed worker, or changes a frozen tool.
