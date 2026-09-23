# Qualify the wide entry compiled with full optimization on first use

Use the actual selected release and M54 normal-build binaries. Preserve every
one of the prior66numerical groups and their assertions; append two explicit
Parallel(2) matrix calls at64/106rows and1024reduction/columns. The new cloned
entry's callback must preserve complete bits, destination identity, poisoned
repeated writes, retained allocating output, guards and input immutability.
The64row case splits32/32 and packs neither;106splits53/53 and candidate scratch
across two destination calls is16,777,216bytes, selected zero.

Each of four workers runs68groups and7,749,681initial values with1,560independent
coordinate checks, normal and AVX512off for both products. Exact existing
fixtures/project/protocol are retained. consumer_scope.py proves all prior
consumer logic unchanged except appending the two frozen ParallelRoutes cases.

Separate codegen workers keep21fixtures/80calls each. Added MatMul2D/MatMulInto
patterns capture new entries and original callers. Require first-use FullOpts
on the wide clone and its packing helper and full native review before scoring.
Packing and raw odd-row copies may inline into the helper; standalone emission
is not required. Their IL must be exact and all emitted/inlined code reviewed.
No timing from this lane establishes a performance claim.

Run consumer_scope.py, then run.py prepare, stage, launch, observe, collect,
and audit.py after all owners terminate. Freeze and commit before execution.
Nine serial jobs,12GiBavailable/3GiBtmpfs preflight,8GiBRSS,1GiBminimum available/
tmpfs,1GiBoutput/job,2GiBcampaign,900seconds/job,four hours total, all compute
threadsCPU2 and monitorCPU0, gaps<10seconds. Existing namespaces refused.
