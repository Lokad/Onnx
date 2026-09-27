"""Two unchanged portable projection facts required in both full-suite AMD modes."""
FACTS = ['BoundariesRetainBitsGuardsAndZeroAllocation', 'ExceptionalValuesRetainPayloadsAndOperandOrder']
NEW_CASES = dict(backend=['Lokad.Onnx.Backend.Tests.PreparedLstmProjectionTests.'+name for name in FACTS], tensors=[])
