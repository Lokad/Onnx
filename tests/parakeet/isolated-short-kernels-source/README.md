# Isolated short-projection kernels

Start from all 420 qualified selected-source inputs. Preserve MathOps.cs and the
original general dispatcher in place. Four existing call operands route through
a new guard; only 48–63 rows with both widths at least 1,024 and the established
packing budget enter the short-projection path.

The new file contains two private dispatch helpers and four internal arithmetic
copies. Those copies alone request first-use optimization. Existing arithmetic,
flags, budgets, public APIs and package dependencies remain selected. Compiled
inventory, actual-DLL numerical tests and fresh performance qualification remain
mandatory; M43's application result does not transfer to this design.

Run `C:/Python313/python.exe -X utf8 -B prepare.py` from this directory. It
refuses existing output, verifies all selected inputs and records a separate
421-file source snapshot. It does not modify the root product or BENCHMARK.md.
