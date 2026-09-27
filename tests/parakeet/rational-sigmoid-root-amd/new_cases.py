"""The eight existing arithmetic facts, unchanged in both full-suite modes."""
FACTS = ['AllTailsAndZeroLengthMatchScalar', 'SpecialValuesAtEveryVectorLane',
         'ExponentAndRoundingBoundaries', 'DenseFiniteAndBitPatternSweeps',
         'LogicalLayoutsAndOffsetsMatchTheirValues', 'OutputsAreIndependentOfInputsAndLaterCalls',
         'DoublePathRemainsExact', 'InvalidTypesAndOptionsKeepTheirContracts']
NEW_CASES = dict(backend=['Lokad.Onnx.Backend.Tests.SigmoidVectorTests.'+name for name in FACTS], tensors=[])
