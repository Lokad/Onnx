"""Prove bijective generated-symbol renames while preserving every IL instruction."""
import collections
import json
import re

# Only compiler identifiers with declaration ordinals. Local lambda ordinals
# without a declaration prefix (b__1) and ordinary numeric constants are untouched.
TOKEN = re.compile(r'<>c__DisplayClass\d+_\d+|<>9__\d+_\d+|<[^<>]+>b__\d+_\d+|<[^<>]+>g__[^|<>]+\|\d+_\d+|<[^<>]+>d__\d+')
ORDINAL = re.compile(r'(?<=DisplayClass)\d+|(?<=b__)\d+(?=_\d+)|(?<=9__)\d+(?=_\d+)|(?<=\|)\d+(?=_\d+)|(?<=d__)\d+')


def family(key):
    for prefix in ['Lokad.Onnx.Tensor`1', 'Lokad.Onnx.ComputationalGraph']:
        if key.startswith(prefix): return prefix
    return None


def canonical(value):
    return TOKEN.sub(lambda m: ORDINAL.sub('#', m.group()), value)


def walk(value, transform):
    if isinstance(value, str): return transform(value)
    if isinstance(value, list): return [walk(v, transform) for v in value]
    if isinstance(value, dict):
        # A user string is not a metadata symbol, even if it resembles one.
        return {k: v if k == 'operand' and value.get('opcode') == 'ldstr' else walk(v, transform) for k, v in value.items()}
    return value


def parse(body): return body if body == 'NO-BODY' else json.loads(body)


def normalize(row):
    old = row['normalized_methods']; removed = set(row['removed']); changed = set(row['differences'])
    new = {k: v for k, v in old.items() if k not in removed | changed}
    assert not new.keys() & row['candidate_methods'].keys()
    new.update(row['candidate_methods'])
    forward, inverse = {}, {}

    def collect(a, b, owner):
        if owner is None: return
        assert canonical(a) == canonical(b)
        left, right = TOKEN.findall(a), TOKEN.findall(b); assert len(left) == len(right)
        for before, after in zip(left, right, strict=True):
            assert canonical(before) == canonical(after)
            source, target = (owner, before), (owner, after)
            assert forward.get(source, after) == after and inverse.get(target, before) == before, (source, target)
            forward[source] = after; inverse[target] = before

    def collect_tree(a, b, owner):
        assert type(a) is type(b)
        if isinstance(a, str): collect(a, b, owner)
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True): collect_tree(x, y, owner)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                if k == 'operand' and a.get('opcode') == 'ldstr': assert a[k] == b[k]
                else: collect_tree(a[k], b[k], owner)
        else: assert a == b

    # Identical ordinary method bodies establish symbol pairings through their
    # actual call/field/type operands, including overloaded local functions.
    for key in old.keys() & new.keys():
        owner = family(key)
        a, b = parse(old[key]), parse(new[key])
        if owner and TOKEN.search(key) is None and walk(a, canonical) == walk(b, canonical): collect_tree(a, b, owner)

    groups_old, groups_new = collections.defaultdict(list), collections.defaultdict(list)
    for key in old: groups_old[canonical(key)].append(key)
    for key in new: groups_new[canonical(key)].append(key)
    # Unique metadata signatures resolve renamed callbacks in deliberately changed
    # parent methods. Ambiguous signatures are never paired by enumeration order.
    for signature, keys in groups_old.items():
        matches = groups_new[signature]
        if len(keys) == len(matches) == 1:
            a, b = keys[0], matches[0]
            if family(a) and a != b:
                collect(a, b, family(a))
                before, after = parse(old[a]), parse(new[b])
                if walk(before, canonical) == walk(after, canonical): collect_tree(before, after, family(a))

    def restore(value, owner):
        if owner is None: return value
        return TOKEN.sub(lambda m: inverse.get((owner, m.group()), m.group()), value)

    normalized = {}; renames = []
    for key, body in new.items():
        owner = family(key)
        target = restore(key, owner)
        assert target not in normalized, ('Rename collision', target)
        normalized[target] = walk(parse(body), lambda s: restore(s, owner))
        if key != target: renames.append(dict(before=target, after=key))
    missing = sorted(old.keys() - normalized.keys())
    assert not missing, missing[:3]
    differences = sorted(k for k in old if parse(old[k]) != normalized[k])
    added = sorted(normalized.keys() - old.keys())
    # A rename cannot hide an altered generated callback body.
    for row in renames:
        assert row['before'] in old and row['before'] not in differences, row
    return dict(methods=len(old), unchanged=len(old)-len(differences), changed=differences, added=added,
        renames=renames, symbols=[dict(family=o, before=a, after=b) for (o, a), b in sorted(forward.items()) if a != b]), normalized
