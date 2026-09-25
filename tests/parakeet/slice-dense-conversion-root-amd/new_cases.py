"""The26 previously qualified public slice tests; no campaign identity fixture."""
PREFIX='Lokad.Onnx.Tensors.Tests.SliceDenseConversionTests.'
FRAMES=[51, 61, 83, 88, 89, 102, 106, 112, 114, 120, 151, 156, 157, 158, 167, 169, 190, 222, 225]
FACTS=['MultiAxisCropRetainsGaps', 'FullViewStillOwnsItsCopy', 'SteppedNegativeNestedAndReducedFallbacksKeepValues', 'EmptyAndReversedStorageRemainSupported', 'DerivedParentAndViewBehaviorIsPreserved', 'MatMulMaterializesSliceWithIndependentScalarReference', 'CloneAndRepeatedDenseConversionsRemainIndependent']
NEW_CASES=dict(backend=[],tensors=sorted([PREFIX+f'PositionalSliceCopiesEveryBitAndOwnsStorage(t: {t})' for t in FRAMES]+[PREFIX+n for n in FACTS]))
assert len(NEW_CASES['tensors'])==len(set(NEW_CASES['tensors']))==26
