"""The only prerequisite change is an explicit exact three-product lineage."""


def with_lineage(text):
    changes=[
        ("from semantics import compare_records", "from semantics import compare_records\nfrom lineage import qualify_lineage"),
        ("{'product','models','parakeet','shared'}", "{'product','models','parakeet','shared','parakeet-release'}"),
        ("assert reports['product']['identities'] == spec['identities']", "parakeet = qualify_lineage(base,reports,spec)"),
        ("for name in ['models','parakeet','shared']:", "for name in ['models','shared']:"),
        ("reports['parakeet']['results'][role+'-native-'+isa]", "parakeet['results'][role+'-native-'+isa]"),
        ("reports['parakeet']['results'][role+'-public-'+isa]", "parakeet['results'][role+'-public-'+isa]")]
    for before,after in changes:
        assert text.count(before)==1,before
        text=text.replace(before,after)
    return text
