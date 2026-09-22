# Project-based regression invocation

Run `C:/Python313/python.exe -X utf8 -B` with `run.py`, then `audit.py` after
successful termination. This explicit successor retains the original direct-DLL
invocation, which exited 1 without a TRX. Its only logged diagnostic said that
`--no-build --no-restore` were ignored; no numerical failure was reported.

The successor uses the established `dotnet test <project> -c Release --tl:off
--no-build --no-restore` command with the original isolated normal project.
Its product/source pins are checked before and after. The original suite scope
and all resource limits remain. The 31 corrected new tests retain their separate
successful closure. Neither the original archived sources nor product assemblies
are rebuilt or corrected in place.
