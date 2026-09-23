"""Only the two expected product hashes in SampledAudio.Main may change."""
OLD_CORE='3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
OLD_DATA='6318cf48691470b908eec4c4d09c558172e43ce3b04bca9039c68966998a684b'


def inventory(value,previous,built,product):
    assert value['inventory_complete']
    row,=value['observations']
    assert row['assembly']=='SampledAudio.dll' and row['methods']==160 and row['unchanged_methods']==159
    assert row['public_surface_equal'] and not row['added'] and not row['removed'] and row.get('compiler_rename') is None
    assert row['before_sha256']==previous['sha256'] and row['after_sha256']==built['sha256']
    assert len(row['normalized_methods'])==160
    key,=row['differences'];assert key.startswith('Program::<Main>$::') and set(row['candidate_methods'])=={key}
    expected=row['normalized_methods'][key]
    for old,new in [(OLD_CORE,product['Lokad.Onnx.dll']['sha256']),(OLD_DATA,product['Lokad.Onnx.Data.dll']['sha256'])]:
        assert expected.count(old)==1  # Actual qualified compiled census, not a source-count assumption.
        expected=expected.replace(old,new)
    assert expected==row['candidate_methods'][key]
    return dict(passed=True,methods=160,unchanged=159,changed=[key],only_product_hash_literals=True,public_surface_equal=True)
