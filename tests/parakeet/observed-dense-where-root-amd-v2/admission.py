"""Bind each admitted comparison to the same actual selected and candidate binaries."""
def product_identities(build,selected,parakeet,pyannote,graphs):
    candidate=build['built'];current=selected['measured']
    assert set(candidate)==set(current)=={'Lokad.Onnx.dll','Lokad.Onnx.Data.dll'}
    assert parakeet['identities']==dict(current=current,candidate=candidate)
    assert pyannote['identities']==dict(selected=current,candidate=candidate)
    assert graphs['products']==dict(current={'Lokad.Onnx.dll':current['Lokad.Onnx.dll']},
                                    candidate={'Lokad.Onnx.dll':candidate['Lokad.Onnx.dll']})
    return dict(passed=True,selected=current,candidate=candidate)
