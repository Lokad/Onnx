# Explicit dictionary type in the corrected test helper

The normal-source v2 rebuilt Core and Data but its backend test build failed:
after removing optional parameters, target-typed `new()` resolved against the
`object` graph execution overload. This successor spells out
`new Dictionary<string, ITensor>` in that helper. It retains the failed build,
creates a fresh normal source copy, and applies all previous test corrections.
Product source, numerical assertions and suite/resource gates are unchanged.

Run `C:/Python313/python.exe -X utf8 -B` with `complete.py`, then `verify.py` after
successful termination. The v2 build/suite/instruction auditor is reused through
explicit module bindings without editing its executed files. Every 3,161 Core
and 697 Data method must match the preceding product. The combined suite expects
3,344 backend passes and 93 skips, 343 tensor passes, and 31 disabled focused
passes. It includes the unchanged optional-parameter source rule.
