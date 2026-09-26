"""Exactly the six previously qualified Pad facts, in both instruction modes."""
FACTS = ['FloatPaddingPreservesBitsAndOwnership', 'DoublePaddingPreservesBitsAndOwnership',
         'Int32PaddingPreservesBitsAndOwnership', 'Int64PaddingPreservesBitsAndOwnership',
         'SlicedAndBroadcastInputsRemainUnchanged', 'ReflectionRetainsItsExistingPath']
NEW_CASES = dict(backend=['Lokad.Onnx.Backend.Tests.LastAxisPadTests.'+name for name in FACTS], tensors=[])
