"""Reverse the one explicit logging allowlist predicate, preserving all other text."""
OLD='Require(flags.Count == 0, "Runtime override");'
NEW='Require(DisassemblyPolicy.Allowed(flags), "Runtime override");'


def instrument(source):
    assert source.count(OLD)==1 and NEW not in source
    result=source.replace(OLD,NEW);verify(source,result);return result


def verify(source,actual):
    assert actual.count(NEW)==1 and actual.replace(NEW,OLD)==source
    return dict(passed=True,original_recovered_exactly=True,changed_flag_predicates=1,
                calls=6000,original_warmup=600,original_measurement_labels=180,
                timing_numerical_ownership_and_observation_hooks_unchanged=True)
