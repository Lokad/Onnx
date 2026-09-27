"""Two independent portable remainder facts required in both full-suite AMD modes."""
FACTS = ['NarrowWidthsAccumulateWithoutAllocationOrOverwrite', 'RowAndPanelBoundariesPreserveAccumulationAndOwnership']
NEW_CASES = dict(backend=['Lokad.Onnx.Backend.Tests.PackedColumnRemainderTests.'+name for name in FACTS], tensors=[])
