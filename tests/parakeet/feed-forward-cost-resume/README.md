# Finish the measured feed-forward cost split

The original capture completed all 80 clock-only requests, then stopped after
33 stage traces at its 512 MiB output guard. Preserve its code 1 and failed audit.
The first 20 traces occupy 260,663,037 bytes; convolution produces hundreds of
thousands of stage records per request. RAM, elapsed-time and RSS guards passed.

Run only the incomplete stages role and previously unstarted markers role, in
separate namespaces. Use the exact reviewed Core, Data and consumer binaries.
Keep 20 clips, one warmup and three measured passes, CPU 2, one thread, 11 GiB
preflight, 12 GiB RSS, 900 seconds, and 1 GiB minimum free RAM/tmpfs. Increase
only the prospective output ceiling to 2 GiB per role, based on observed trace
volume. Do not compress, filter or aggregate events inside a timed process.
The original 512 MiB failure remains a failure. No product promotion follows.

From the repository root with Python 3.13, run `run.py prepare stages`, then
`prepare markers`. Run `retire.py original` only after verifying the terminal
original collection. This deletes only its 33 duplicate VM cost JSON files;
every local original remains. Stage and launch stages, observe until terminal,
collect once, and run `audit.py stages` once. Only then run `retire.py stages`,
stage/launch/observe/collect markers and run `audit.py markers` once. Run
`audit.py joint` once after all three complete roles have passed structural,
numerical, ownership and resource checks. Never repeat a completed owner or
discard the stopped role's partial data. Every action writes separate receipts.

The joint auditor reuses the original accounting, exact reference, repeatability
and observer-effect criteria. It includes the original successful clock and
both complete recovery roles, with no overhead subtraction. The 33 original
partial stage requests remain explicitly excluded from those complete-role
comparisons. A failed observer control makes the split unusable for selecting
an optimization. The M73 e5 repeatability failure still blocks product promotion.
