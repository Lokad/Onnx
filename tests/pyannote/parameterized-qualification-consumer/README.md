# Pass both expected product identities to Pyannote qualification

The retained GraphQualification consumer already receives its expected Core DLL
SHA256 as an argument, but embeds the expected Data DLL SHA256 as a constant.
Replace that constant with a fifth argument supplied by the frozen campaign
payload. Keep the actual Data assembly hashing and equality check intact.
The only other source change is argument-count validation and its usage message.
This permits the same compiled consumer to qualify both current and candidate
products, including future assembly-identity changes, without repeated builds.

source() derives the two-line edit from the exact retained source. inventory()
requires the same 96 methods, 95 unchanged, identical public surface, no additions
or removals, and only the specified Main instructions changed. It derives the
corresponding byte offsets, relative branch operands and exception ranges and
compares the whole method. It does not claim this older inspector measures method
implementation flags; source, compiler settings and project inputs remain fixed.

Seven local scope/rejection tests pass against the actual retained source and
instruction inventory. The positive instruction fixture is a specification test,
not evidence that a new consumer has compiled. A new VM build, real inventory
comparison and wrong-Core/wrong-Data rejection probes remain required before use.
No numerical tolerance, tensor comparison, public-result or ownership check changes.

The current padding application must finish and pass, followed by shared-model
correctness, before building this consumer in the Pyannote qualification lane.
Use the offline environment and existing product DLLs; do not rebuild products.
