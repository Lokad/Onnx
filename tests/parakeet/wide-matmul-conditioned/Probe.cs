using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using Lokad.Onnx;

static unsafe class Probe
{
    static void Consume(string kind, int m, int k, int n, float* a, float* b, float* packed, float* c)
    {
        int rows = m % 3 == 0 ? m : m-m%2;
        if (kind == "six")
        {
            int six = rows-rows%6;
            if (six > 0) MathOps.mm_unsafe_vectorized_intrinsics_6x4packed(six,k,n,a,packed,c);
            int rest = rows-six;
            if (rest % 3 == 0 && rest > 0) MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(rest,k,n,a+six*k,packed,c+six*n);
            else if (rest > 0) MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(rest,k,n,a+six*k,packed,c+six*n);
        }
        else if (kind == "base")
        {
            if (rows % 3 == 0) MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(rows,k,n,a,packed,c);
            else MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(rows,k,n,a,packed,c);
        }
        else Support.Require(Blocked.TryRun(rows,k,n,a,packed,c,int.Parse(kind)),"Unexpected candidate refusal");
        if (rows != m) MathOps.mm_unsafe_vectorized_intrinsics(1,k,n,a+rows*k,b,c+rows*n);
    }

    static int SelfTest()
    {
        var random = new Random(7091); int count = 0;
        foreach (int m in new[] { 2,3,4,5,6,51,64,65 })
        foreach (int k in new[] { 1,127,128,129,255,256,257,1024,4096 })
        foreach (int n in new[] { 32,64 })
        {
            float[] a = Enumerable.Range(0,m*k).Select(_ => (float)(random.NextDouble()-.5)).ToArray();
            float[] b = Enumerable.Range(0,k*n).Select(_ => (float)(random.NextDouble()-.5)).ToArray();
            float[] packed = new float[b.Length];
            float[] initial = Enumerable.Range(0,m*n).Select(_ => (float)(random.NextDouble()-.5)).ToArray();
            float[] expected = (float[])initial.Clone();
            fixed (float* ap = a, bp = b, pp = packed, ep = expected)
            {
                MathOps.PackPanelsB(k,n,bp,pp); Consume("base",m,k,n,ap,bp,pp,ep);
                foreach (string kind in new[] { "128","256","512","six" })
                {
                    float[] guarded = Enumerable.Repeat(-98765.5f,m*n+32).ToArray(); initial.CopyTo(guarded,16);
                    fixed (float* cp = guarded) Consume(kind,m,k,n,ap,bp,pp,cp+16);
                    Support.Require(MemoryMarshal.AsBytes(guarded.AsSpan(16,m*n)).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan())),"Nonzero accumulation differs");
                    Support.Require(guarded.AsSpan(0,16).IndexOfAnyExcept(-98765.5f) < 0 && guarded.AsSpan(m*n+16,16).IndexOfAnyExcept(-98765.5f) < 0,"Output canary changed");
                }
                Support.Require(!Blocked.TryRun(1,k,n,ap,pp,ep,128) && !Blocked.TryRun(m,k,31,ap,pp,ep,128),"Unsupported geometry accepted");
            }
            count++;
        }
        return count;
    }

    public static void Run(string root, string manifestPath, string output, int ordinal)
    {
        Support.Require(ordinal is >= 0 and < 4,"Worker ordinal"); int tests = SelfTest();
        var spec = Support.Read(manifestPath); var captures = Support.Read(Support.PathOf(root,spec.GetProperty("capture")));
        var fixtures = captures.GetProperty("entries").EnumerateArray().ToArray();
        var rows = new List<object>(); var checks = new List<object>();
        using var log = new StreamWriter(new FileStream(Path.Combine(output,"samples.jsonl"),FileMode.CreateNew));
        string[] roles = ["public","pack","consume-base","consume-128","consume-256","consume-512","consume-six", "combined-base","combined-128","combined-256","combined-512","combined-six"];
        int total = 0;
        for (int ci = 0; ci < fixtures.Length; ci++)
        {
            int caseIndex = (ordinal % 2 == 0 ? ci : fixtures.Length-1-ci); var f = fixtures[caseIndex];
            int m = f.GetProperty("m").GetInt32(), k = f.GetProperty("k").GetInt32(), n = f.GetProperty("n").GetInt32();
            float[] Read(string file) => MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(Support.PathOf(root,spec.GetProperty("arrays").GetProperty(file)))).ToArray();
            float[] a = Read(f.GetProperty("a").GetProperty("file").GetString()!);
            float[] b = Read(captures.GetProperty("weights").GetProperty(f.GetProperty("weight").GetString()!).GetProperty("file").GetString()!);
            float[] expected = Read(f.GetProperty("y").GetProperty("file").GetString()!);
            float[] packed = new float[b.Length], c = new float[expected.Length];
            var ta = new DenseTensor<float>(a,new[] { m,k }); var tb = new DenseTensor<float>(b,new[] { k,n }); var tc = new DenseTensor<float>(c,new[] { m,n });
            string ah = Support.Hash(MemoryMarshal.AsBytes(a.AsSpan())), bh = Support.Hash(MemoryMarshal.AsBytes(b.AsSpan()));
            fixed (float* ap = a, bp = b, pp = packed, cp = c)
            {
                MathOps.PackPanelsB(k,n,bp,pp);
                nint aa = (nint)ap, bb = (nint)bp, p = (nint)pp, cc = (nint)cp;
                string packedHash = Support.Hash(MemoryMarshal.AsBytes(packed.AsSpan()));
                void Invoke(string role)
                {
                    if (role == "public" || (role.StartsWith("combined-") && m < 64))
                        Tensor<float>.MatMul2D(ta,tb,tc,TensorExecutionOptions.Auto);
                    else if (role == "pack") MathOps.PackPanelsB(k,n,(float*)bb,(float*)p);
                    else
                    {
                        c.AsSpan().Clear(); string kind = role[(role.IndexOf('-')+1)..];
                        if (role.StartsWith("consume-")) Consume(kind,m,k,n,(float*)aa,(float*)bb,(float*)p,(float*)cc);
                        else
                        {
                            float[] rental = ArrayPool<float>.Shared.Rent(b.Length);
                            try
                            {
                                fixed (float* rp = rental) { MathOps.PackPanelsB(k,n,(float*)bb,rp); Consume(kind,m,k,n,(float*)aa,(float*)bb,rp,(float*)cc); }
                            }
                            finally { ArrayPool<float>.Shared.Return(rental); }
                        }
                    }
                }
                void Sample(string role, string phase, int repetition)
                {
                    long start = Stopwatch.GetTimestamp(); Invoke(role); long end = Stopwatch.GetTimestamp();
                    if (role == "pack") Support.Require(Support.Hash(MemoryMarshal.AsBytes(packed.AsSpan())) == packedHash,"Pack changed bytes");
                    else Support.Require(MemoryMarshal.AsBytes(c.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan())),"Actual output differs: "+role+"/"+caseIndex);
                    log.WriteLine(JsonSerializer.Serialize(new { ordinal, case_index = caseIndex, role, phase, repetition, start_ticks = start, end_ticks = end, frequency = Stopwatch.Frequency })); total++;
                }
                for (int ri = 0; ri < roles.Length; ri++)
                {
                    string role = roles[(ri+ordinal*3)%roles.Length]; long beginning = Stopwatch.GetTimestamp(); int repeat = 0;
                    do { Sample(role,"warmup",repeat++); } while (repeat < 32 || Stopwatch.GetElapsedTime(beginning).TotalSeconds < .25);
                }
                // Every ordered pair of distinct roles occurs once across these
                // twelve Williams orders. Condition the exact upcoming role to
                // reduce dependence on the preceding role's working set.
                int[] balanced = [0,1,11,2,10,3,9,4,8,5,7,6];
                for (int repeat = 0; repeat < 12; repeat++)
                    for (int ri = 0; ri < roles.Length; ri++)
                    {
                        int position = ordinal % 2 == 0 ? ri : roles.Length-1-ri;
                        string role = roles[(balanced[position]+ordinal*3+repeat)%roles.Length];
                        for (int conditioning = 0; conditioning < 3; conditioning++)
                            Sample(role,"conditioning",repeat*3+conditioning);
                        Sample(role,"measured",repeat);
                    }
                Support.Require(Support.Hash(MemoryMarshal.AsBytes(packed.AsSpan())) == packedHash,"Packed input changed");
            }
            Support.Require(Support.Hash(MemoryMarshal.AsBytes(a.AsSpan())) == ah && Support.Hash(MemoryMarshal.AsBytes(b.AsSpan())) == bh,"Caller operands changed");
            checks.Add(new { case_index = caseIndex, m,k,n, passed = true, expected_sha256 = Support.Hash(MemoryMarshal.AsBytes(expected.AsSpan())) });
            log.Flush(); Console.WriteLine("case "+caseIndex+" "+m+"x"+k+"x"+n+" passed");
        }
        Support.Write(Path.Combine(output,"result.json"),new { passed = true, identity = Support.Identity(), ordinal, tests, checks, samples = total,
            scope = "Local actual-operand kernel probe; normal runtime; combined includes pool rental, pack, clear and consume; M<64 combined preserves public fallback; no new application/native timing" });
    }
}
