"""Preserve the original numerical/reference logic while expanding shapes and DLL binding."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXTRA = '''        foreach (int c in new[] { 16, 32 })
        foreach (int m in new[] { 32, 48, 64, 80, 96, 128, 256, 512 })
        foreach (var (h, w) in new[] { (1,1), (3,17), (5,33), (6,34) })
            yield return (c, m, h, w);
'''
REFUSAL = '''    static int SingleBlockRefusals(int lanes)
    {
        const int c = 16, m = 16, h = 3, w = 17;
        var weights = new float[c * m * 9]; string weightHash = Hash(weights);
        bool refused = false;
        try { ConvBlockedSpatial.PrepareWinograd(weights,c,m,lanes); }
        catch (ArgumentException) { refused = true; }
        Require(refused && Hash(weights) == weightHash, "single-block preparation refusal");
        Require(ConvBlockedSpatial.PlanWinograd(c,m,h,w,out int ni,out int np,out int no), "refusal scratch plan");
        var input = new float[c*h*w]; var prepared = new float[16*c*m];
        var output = Enumerable.Repeat(Sentinel,no).ToArray();
        var transformed = Enumerable.Repeat(Sentinel,ni).ToArray();
        var products = Enumerable.Repeat(Sentinel,np).ToArray();
        var blocked = Enumerable.Repeat(Sentinel,no).ToArray();
        var arrays = new[] { input, prepared, output, transformed, products, blocked };
        var before = arrays.Select(Hash).ToArray(); refused = false;
        try { ConvBlockedSpatial.ExecuteWinograd(input,prepared,default,default,output,
            transformed,products,blocked,c,m,h,w,lanes,false); }
        catch (ArgumentException) { refused = true; }
        Require(refused && arrays.Select(Hash).SequenceEqual(before), "single-block execution refusal before mutation");
        return 2;
    }

'''


def generate():
    text = (ROOT/'tests/pyannote/winograd-range-prototype/Driver.cs').read_text(encoding='utf8')
    changes = [
        ('using Lokad.Onnx;', 'using ConvBlockedSpatial = KernelAccess;'),
        ('            yield return (c, m, h, w);\n    }', '            yield return (c, m, h, w);\n'+EXTRA+'    }'),
        ('Require(rows.Count == 1920, "raw census");','Require(rows.Count == 3456, "raw census");'),
        ('    static int Main(string[] args)',REFUSAL+'    static int Main(string[] args)'),
        ('Require(args.Length == 4, "mode width fixtures result");','Require(args.Length == 5, "mode width fixtures result core");\n        KernelAccess.Bind(args[4]);'),
        ('        // Keep every numerical failure', '        int singleBlockRefusals = mode == "raw" ? SingleBlockRefusals(lanes) : 0;\n        // Keep every numerical failure'),
        ('            rows, refusals, contracts, rangeChecks, noPerformanceMeasurement',
         '            core_sha256 = KernelAccess.CoreHash, rows, refusals, contracts, rangeChecks, singleBlockRefusals, noPerformanceMeasurement')]
    for old,new in changes:
        assert text.count(old) == 1,old
        text = text.replace(old,new)
    return text


if __name__ == '__main__':
    target = Path(__file__).with_name('Driver.cs'); assert not target.exists()
    target.write_text(generate(),encoding='utf8',newline='\n')
