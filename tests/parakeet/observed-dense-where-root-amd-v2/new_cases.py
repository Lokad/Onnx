"""Prospective public regression census; actual AMD TRX must contain each once."""
PREFIX='Lokad.Onnx.Backend.Tests.DenseScalarWhereTests.'
FRAMES=[51,61,83,88,89,102,106,112,114,120,151,156,157,158,167,169,190,222,225]
BITS=[0,-2147483648,2143294004,2139095040]
FACTS=['MixedAttentionMaskPreservesSelectedPayloadsAndMemoryWindows',
       'MixedOuterMaskBroadcastsAcrossChannelsAndContiguousRows',
       'HigherRankConditionStillExpandsTheOutput',
       'UniformFalseMaskDoesNotBypassIncompatibleShapes',
       'DerivedTensorValueSemanticsRemainObservable']
NEW_CASES=dict(backend=sorted(
    [PREFIX+f'AttentionMaskCopiesEveryHeadWithIndependentStorage(t: {t})' for t in FRAMES]+
    [PREFIX+f'UniformNoncanonicalTrueMaskPreservesScalarBits(bits: {bits})' for bits in BITS]+
    [PREFIX+name for name in FACTS]),tensors=[])
assert len(NEW_CASES['backend'])==len(set(NEW_CASES['backend']))==28
