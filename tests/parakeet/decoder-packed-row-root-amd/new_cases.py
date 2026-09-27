"""Four portable regression facts, required in both full-suite AMD modes."""
FACTS = ['DimensionsRowsAndBatchesRetainBitsAndOwnership',
         'OptionsMissingAndReplacedWeightsRetainBitsAndOwnership',
         'RawBoundariesRetainArithmeticGuardsAndZeroAllocation',
         'RawExceptionalValuesRetainNanPayloadsAndOwnedInputs']
NEW_CASES = dict(backend=['Lokad.Onnx.Backend.Tests.PreparedSingleRowTests.'+name for name in FACTS], tensors=[])
