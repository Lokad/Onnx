"""Constrain source changes to the new load dispatch/helper and raw geometry enumeration."""
from pathlib import Path
GUARD='        int topStart = first / tileWidth * 2 - 1;\n        int leftStart = first % tileWidth * 2 - 1;\n        if (count == WinogradBatch && first / tileWidth == (first + 7) / tileWidth\n            && topStart >= 0 && topStart + 3 < h && leftStart >= 0 && leftStart + 17 < w)\n        {\n            TransformWinogradInputContiguous(input, transformed, c, h, w, topStart, leftStart);\n            return;\n        }\n'
GEOMETRIES='    static IEnumerable<(int c, int m, int h, int w)> RawGeometries()\n    {\n        var pairs = new[] { (16,32), (16,48), (32,32), (32,64), (64,48), (64,64), (128,128), (256,256) };\n        foreach (var shapes in new[] {\n            new[] { (1,1), (1,7), (3,5), (4,8), (5,17), (6,16) },\n            new[] { (5,33), (6,34), (5,65), (6,66) } })\n        foreach (var (c, m) in pairs)\n        foreach (var (h, w) in shapes)\n            yield return (c, m, h, w);\n    }\n\n'
OLD_LOOPS='        foreach (var (c, m) in new[] { (16,32), (16,48), (32,32), (32,64), (64,48), (64,64), (128,128), (256,256) })\n        foreach (var (h, w) in new[] { (1,1), (1,7), (3,5), (4,8), (5,17), (6,16) })'

def method(text,signature):
    start=text.index(signature);opening=text.index('{',start);depth=1;end=opening+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return start,end,text[opening+1:end-1]

def check(new,old):
    a=(new/'Winograd.cs').read_text();b=(old/'Winograd.cs').read_text()
    signature='    static void TransformWinogradInput('
    sa,ea,ba=method(a,signature);sb,eb,bb=method(b,signature)
    assert a[:sa]==b[:sb] and ba.replace(GUARD,'')==bb and ba.count(GUARD)==1
    extra='    static void TransformWinogradInputContiguous('
    se,ee,_=method(a,extra)
    assert not a[ea:se].strip() and a[ee:]==b[eb:]
    driver=(new/'Driver.cs').read_text()
    assert driver.count(GEOMETRIES)==1
    driver=driver.replace(GEOMETRIES,'').replace('        foreach (var (c, m, h, w) in RawGeometries())',OLD_LOOPS).replace('rows.Count == 1920','rows.Count == 1152')
    assert driver==(old/'Driver.cs').read_text()
