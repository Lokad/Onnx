# Explicit successor for compiled-code review and focused qualification

The first normal-source composition built CLI, backend and tensors successfully,
then stopped before testing because the review rejected190apparent method removals.
That complete attempt remains at failure closureb52ca1fe. Product source and
binaries are unchanged by this successor.

The review proves compiler-generated symbol renames through actual metadata
signatures and unchanged method operands, requiring a bijection. It then compares
every instruction, operand, local, exception clause and stack entry. It never
normalizes user strings or numeric constants. All renamed callback bodies must
match. The actual inventory has197renamed methods, including190previously reported
removals and7overlapping metadata names. After proof,3103existing Core methods
and all697Data methods remain unchanged; exactly10existing Core methods change
and48are added. All11pure component methods remain identical after class renaming.
Public declarations are unchanged. Seven tests reject ambiguous mappings and
modified callbacks/constants and reconcile the actual inventory.

One previously unexecuted test also used the wrong API setting for runtime weight
overrides: supplying initializer-backed inputs requires useInitializers:false.
The exact correction is retained as a diff. A new focused test assembly references
the first candidate's exact Core binary and runs31cases normally and with hardware
intrinsics disabled. It does not rebuild or replace that product binary.

From root, use C:/Python313/python.exe -X utf8 -B with run.py, then audit.py.
The new artifact is artifacts/pyannote-blocked-spatial-composition-review-v2-20260922.
All original source/build/failure evidence is checked before and after. Resource
bounds remain8GiBbuild/12GiBnumerical preflight,8GiBRSS,20GiBdisk,900second workers,
CPU2inherited before runtime and monitorCPU0. Full graph/model/suite/package and
application/ORT qualification remain open after these focused checks.
