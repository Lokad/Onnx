# Preserve Pad tests while meeting repository source policy

The isolated padding build includes six passing LastAxisPadTests, but its Check
helper still declares `bool reversed = false`. The existing repository-wide
SourceTree_HasNoOptionalParameters guard would reject that newly integrated file.
Prepare an integration copy with the default removed and all three omitted
arguments made explicitly false. Keep the existing explicit true call, every
assertion, test name and input case unchanged. Neither product DLL changes.

prepare_tests.py checks the exact frozen source identity, proves the four textual
edits reversible, and writes the corrected file and review patch under
artifacts/parakeet-pad-integration-tests-20260926. It does not apply the file to
root or modify a closed campaign. Run once with C:/Python313/python.exe -X utf8 -B.
The complete root suites, including the original source guard, remain required
after application/shared/Pyannote/graph qualification. No guard is weakened.
