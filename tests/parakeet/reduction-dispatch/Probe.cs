using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static unsafe class Probe
{
    delegate void Kernel(int m, int n, int k, float* a, float* b, float* c);
    delegate bool Admission(int m, int n, int k, float* a, float* b, float* c);
    static void Require(bool value, string message) { if (!value) throw new InvalidOperationException(message); }
    static string Hash(float[] a) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(a.AsSpan())));
    static string FileHash(string p) { using var f = File.OpenRead(p); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    static float[] Read(string p) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(p)).ToArray();
    static void Main(string[] args)
    {
        Require(args.Length == 4, "root, original runtime, result path, normal|disabled");
        string root = args[0], original = args[1]; bool disabled = args[3] == "disabled";
        Require(args[3] is "normal" or "disabled", "Unknown mode");
        Require(Environment.Version.ToString() == "10.0.12" && (long)Process.GetCurrentProcess().ProcessorAffinity == 4, "Runtime/affinity");
        Require(Fma.IsSupported != disabled, "FMA test mode");
        var admit = typeof(MathOps).GetMethod("TryPackedPartialSums", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<Admission>();
        int tests = 0, changed = 0, eligible = 0, fallback = 0, controls = 0, raw = 0, dispatch = 0;
        if (disabled)
        {
            foreach (int m in new[] { 1, 2, 3, 5 })
            {
                int n = 1025, k = 32;
                float[] a = Enumerable.Range(0, m*n).Select(i => (i % 17 - 8) * .125f).ToArray();
                float[] b = Enumerable.Range(0, n*k).Select(i => (i % 13 - 6) * .0625f).ToArray();
                float[] c = Enumerable.Repeat(123.5f, m*k).ToArray();
                fixed (float* ap = a, bp = b, cp = c) Require(!admit(m, n, k, ap, bp, cp), "Hardware-off admission");
                Require(c.All(v => v == 123.5f), "Refusal wrote output");
                float[] expected = new float[m*k];
                for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) for (int p = 0; p < n; p++) expected[i*k+j] += a[i*n+p]*b[p*k+j];
                var actual = Tensor<float>.MatMul2D(new DenseTensor<float>(a, [m,n]), new DenseTensor<float>(b, [n,k])).ToArray();
                Require(Hash(actual) == Hash(expected), "Hardware-off public result"); tests++;
            }
        }
        else
        {
            var context = new AssemblyLoadContext("original", false);
            context.Resolving += (owner, name) => owner.LoadFromAssemblyPath(Path.Combine(original, name.Name + ".dll"));
            var old = context.LoadFromAssemblyPath(Path.Combine(original, "Lokad.Onnx.dll")).GetType("Lokad.Onnx.MathOps")!;
            string[] names = ["mm_unsafe_vectorized_intrinsics_2x4packed_bump", "mm_unsafe_vectorized_intrinsics_3x4packed"];
            Require(!Avx512F.IsSupported, "This bounded local comparison is AVX2; AMD is separate");
            var prepared = typeof(Tensor<float>).GetMethod("RunPreparedPackedRows", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<Kernel>();
            var rng = new Random(652199);
            foreach (int m in new[] { 2, 3, 4, 6 })
            foreach (int n in new[] { 1, 255, 256, 257, 1023, 1024, 1025, 1279, 1280, 1281, 4095, 4096, 4097 })
            foreach (int k in new[] { 1, 7, 8, 31, 32, 33, 64, 65 })
            {
                string name = names[m % 3 == 0 ? 1 : 0];
                var baseline = old.GetMethod(name)!.CreateDelegate<Kernel>();
                var candidate = typeof(MathOps).GetMethod(name)!.CreateDelegate<Kernel>();
                float[] a = Enumerable.Range(0, m*n).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
                float[] b = Enumerable.Range(0, n*k).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
                float[] initial = Enumerable.Range(0, m*k).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
                float[] expected = (float[])initial.Clone(), originalResult = (float[])initial.Clone(), packed = new float[b.Length];
                float[] guarded = Enumerable.Repeat(-98765.5f, m*k+32).ToArray(); initial.CopyTo(guarded, 16);
                string ah = Hash(a), bh = Hash(b);
                bool selected = n >= 1024 && k >= 32 && k % 32 == 0;
                fixed (float* ap = a, bp = b, pp = packed, ep = expected, op = originalResult, gp = guarded)
                {
                    MathOps.PackPanelsB(n, k, bp, pp); string ph = Hash(packed);
                    baseline(m,n,k,ap,pp,op);
                    if (selected)
                    {
                        for (int i = 0; i < m; i++) for (int j = 0; j < k; j++)
                        for (int begin = 0; begin < n; begin += 256)
                        {
                            float partial = 0;
                            for (int p = begin; p < Math.Min(n, begin+256); p++) partial = MathF.FusedMultiplyAdd(b[p*k+j],a[i*n+p],partial);
                            expected[i*k+j] += partial;
                        }
                        eligible++;
                    }
                    else { originalResult.CopyTo(expected,0); fallback++; }
                    candidate(m,n,k,ap,pp,gp+16);
                    Require(Hash(guarded.AsSpan(16,m*k).ToArray()) == Hash(originalResult), "Raw kernel contract " + name);
                    raw++;
                    initial.CopyTo(guarded,16);
                    bool admitted = admit(m,n,k,ap,pp,gp+16);
                    Require(admitted == selected, "Helper admission");
                    if (!admitted) Require(Hash(guarded.AsSpan(16,m*k).ToArray()) == Hash(initial), "Refusal wrote output");
                    initial.CopyTo(guarded,16);
                    prepared(m,n,k,ap,pp,gp+16);
                    Require(MemoryMarshal.AsBytes(guarded.AsSpan(16,m*k)).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan())), $"Geometry {m}/{n}/{k}");
                    Require(guarded.AsSpan(0,16).IndexOfAnyExcept(-98765.5f) < 0 && guarded.AsSpan(m*k+16,16).IndexOfAnyExcept(-98765.5f) < 0, "Canary");
                    Require(Hash(packed) == ph && Hash(a) == ah && Hash(b) == bh, "Input mutation");
                    if (Hash(expected) != Hash(originalResult)) changed++;
                }
                tests++;
            }
            Require(changed == eligible && eligible > 0 && fallback > 0, "Missing positive/negative arithmetic controls");
            string prior = Path.Combine(root, "artifacts/parakeet-reduction-accuracy-20260921");
            float[] weight = Read(Path.Combine(prior, "weight.bin")), packedWeight = new float[4096*1024];
            fixed (float* bp = weight, pp = packedWeight)
            {
                MathOps.PackPanelsB(4096,1024,bp,pp);
                foreach (string route in new[] { "managed-native", "native-native", "managed-managed", "native-managed" })
                {
                    float[] a = Read(Path.Combine(root, "artifacts/parakeet-projection-20260921/outputs",route,"03.bin"));
                    float[] expected = Read(Path.Combine(prior,"output",route,"256-projection.bin")), output = new float[74*1024];
                    output = Tensor<float>.MatMul2D(new DenseTensor<float>(a,[74,4096]),new DenseTensor<float>(weight,[4096,1024])).ToArray();
                    Require(Hash(output) == Hash(expected), "Actual operand control " + route); controls++;
                }
            }
        }
        if (!disabled)
        {
            var priorContext = new AssemblyLoadContext("prior-candidate", false);
            string priorPath = Path.Combine(root,"artifacts/parakeet-reduction-model-20260921/runtime");
            priorContext.Resolving += (owner,name) => owner.LoadFromAssemblyPath(Path.Combine(priorPath,name.Name+".dll"));
            var assembly = priorContext.LoadFromAssemblyPath(Path.Combine(priorPath,"Lokad.Onnx.dll"));
            var priorTensor = assembly.GetType("Lokad.Onnx.Tensor`1")!.MakeGenericType(typeof(float));
            var priorMethod = priorTensor.GetMethod("RunFloatMatMulKernel", BindingFlags.Static | BindingFlags.NonPublic)!;
            var priorOptions = Activator.CreateInstance(assembly.GetType("Lokad.Onnx.TensorExecutionOptions")!,true,true,1)!;
            var currentMethod = typeof(Tensor<float>).GetMethod("RunFloatMatMulKernel", BindingFlags.Static | BindingFlags.NonPublic)!;
            var random = new Random(119837);
            foreach (int m in new[]{1,2,3,5,64,65,66,126}) foreach(int n in new[]{1023,1024,1025}) foreach(int k in new[]{32,33})
            {
                float[] a=Enumerable.Range(0,m*n).Select(_=>(float)(random.NextDouble()-.5)).ToArray();
                float[] b=Enumerable.Range(0,n*k).Select(_=>(float)(random.NextDouble()-.5)).ToArray();
                float[] wanted=Enumerable.Range(0,m*k).Select(_=>(float)(random.NextDouble()-.5)).ToArray();
                float[] actual=(float[])wanted.Clone();string ah=Hash(a),bh=Hash(b);
                fixed(float* ap=a,bp=b,wp=wanted,cp=actual)
                {
                    priorMethod.Invoke(null,[m,n,k,Pointer.Box(ap,typeof(float*)),Pointer.Box(bp,typeof(float*)),Pointer.Box(wp,typeof(float*)),priorOptions]);
                    currentMethod.Invoke(null,[m,n,k,Pointer.Box(ap,typeof(float*)),Pointer.Box(bp,typeof(float*)),Pointer.Box(cp,typeof(float*)),TensorExecutionOptions.Intrinsics]);
                }
                Require(Hash(actual)==Hash(wanted),$"Dynamic dispatch {m}/{n}/{k}");
                Require(Hash(a)==ah && Hash(b)==bh,"Dispatch input ownership");dispatch++;
            }
        }
        using var stream = new FileStream(args[2], FileMode.CreateNew);
        JsonSerializer.Serialize(stream, new { passed = true, mode = args[3], tests, eligible, fallback, changed, raw_contracts = raw, dynamic_dispatch_cases = dispatch, actual_operand_controls = controls,
            core_sha256 = FileHash(typeof(MathOps).Assembly.Location), original_core_sha256 = FileHash(Path.Combine(original,"Lokad.Onnx.dll")),
            runtime = Environment.Version.ToString(), affinity = 4, processor_count = Environment.ProcessorCount }, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine($"{args[3]}: {tests} geometry tests, {controls} actual operand controls passed");
    }
}
