# Complete Pyannote qualification for the Winograd product

Full product closure399dde04 admits these five jobs: consumer restore/build,
consumer instruction inventory, selected product and candidate. Selected is
Core208371f6/Datab9358370; candidate is Core521bae17/Dataf3b9aa81. The consumer
changes only its required Data identity literal; all other instructions match.
No model/application performance ratio is scored in this qualification.

Each role checks18 graph arrays/2,917,107 output values and16 complete public
requests. Original Microsoft ORT references and1e-4 scaled limits remain.
Winograd changes finite rounding: candidate-versus-selected float differences
are fully reported diagnostically. Segmentation must still be bit-identical.
Speaker assignments, intervals, exclusive timelines, embedding-presence flags,
status, duration and window counts must equal selected exactly. Only the
256-dimensional speaker centroids are numerical: the original native bound
still gates every coordinate. Candidate repeats, inputs and held outputs remain
exact. No unknown public field can be silently ignored. This contract was
specified before running the candidate in the M34 ExecPlan.

All original manifests, graph inputs/native arrays and public references are
frozen and reverified. CPU2 workers,CPU0 supervisor; SDK10.0.204/runtime10.0.8,
12GiB available and3GiB tmpfs preflight,8GiB owned RSS ceiling,1GiB live floors,
900seconds/job,1GiB outputs,2GiB artifacts. No profiling or optimization override
beyond the original consumer's designated diagnostic graph-profile pass.
Use run.py prepare,stage,launch,observe,terminal-only collect,then audit.py.
