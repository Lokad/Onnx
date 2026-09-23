# Isolated register schedule for the Winograd input transform

Start from the selected root product94a550de. Prepare only one private method
change in Zzz.ConvBlockedSpatial.Winograd.cs, keeping every other source byte
and every420root build input unchanged. No product source is edited by prepare.py.

The contiguous transform currently writes and reads a512-byte stack matrix on
every channel. Compute horizontal rows in order0,2,1,3 and retire each row after
its last vertical use. Keep all sixteen original input loads, all32permutations,
24subtractions,8additions and16final stores. Preserve the exact binary operation
order, Vector256 ISA, batch8, caller guard, border path, scratch/ownership rules,
weight layout, multiplication and output transform. No arithmetic reassociation.

This is an unmeasured candidate. Before timing, require a normal product build
with only this private method differing, actual-DLL numerics in both instruction
widths, and all-tier generated-code review without transform-loop spills.
The expanded M36 numerical census and original native bounds remain mandatory.

Prospective complete-call gates: >=5% aggregate gain, <=5% regression for every
form, all18repeatability controls and strict process separation, retaining all
4,176call clocks and464preparation clocks. The smaller component threshold is
fixed for this new input-transform scope before implementation. M36 remains
rejected under its original10% threshold. Full integration still requires the
unchanged3% complete dialogue gain, crop/native/suite/package/meeting gates and
actual-root qualification. No unchanged failed timing retry.

Run C:/Python313/python.exe -X utf8 -B tests/pyannote/winograd-register-transform-source/prepare.py.
An existing artifact destination is refused. The artifact contains a complete
source copy, patch, all source pins and the prospective ExecPlan. Read PLAN.md
and .agent/m37-winograd-register-transform-20260923.md for exact next steps.
