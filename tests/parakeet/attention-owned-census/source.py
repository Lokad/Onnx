"""Adapt only the existing census sizes to include square attention weights."""


def changed(raw):
    before=raw.decode().replace('\r\n','\n')
    pairs=[
        ('payload.Length == 4194304', 'payload.Length == checked(shape[0] * shape[1])'),
        ('rows.Count == 96 && rows.Count(r => !r.Cached) == 87, "Expected 96/87 weight census"',
         'rows.Count == 216 && rows.Count(r => !r.Cached) == 179, "Expected 216/179 weight census"'),
        ('"OwnedPackedWeightCount") == 87 && (long)Property(graph, "OwnedPackedWeightBytes") == 1459617792',
         '"OwnedPackedWeightCount") == 179 && (long)Property(graph, "OwnedPackedWeightBytes") == 1845493760'),
        ('"87 equal-sized replacement payloads"', '"87 feed-forward and 92 attention replacement payloads"'),
        ('v.GetType().FullName == "Lokad.Onnx.OwnedPackedTensor") == 87',
         'v.GetType().FullName == "Lokad.Onnx.OwnedPackedTensor") == 179'),
        ('owned_count = 87, owned_bytes = 1459617792L', 'owned_count = 179, owned_bytes = 1845493760L'),
        ('original_weight_count = 96', 'original_weight_count = 216'),
        ('original_dense_count = 9', 'original_dense_count = 37')]
    after=before
    for old,new in pairs:
        assert after.count(old)==1,old
        after=after.replace(old,new)
    restored=after
    for old,new in reversed(pairs):
        assert restored.count(new)==1
        restored=restored.replace(new,old)
    assert restored==before
    return (after.replace('\n','\r\n') if b'\r\n' in raw else after).encode()
