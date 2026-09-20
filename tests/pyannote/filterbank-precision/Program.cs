using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using PrecisionProbe;

if (!OperatingSystem.IsWindows() || args.Length != 1 || !BitConverter.IsLittleEndian)
    throw new PlatformNotSupportedException("Expected Windows and artifact path.");
using var process = Process.GetCurrentProcess();
if (process.ProcessorAffinity != (IntPtr)1 || Environment.ProcessorCount != 1) throw new InvalidOperationException("CPU0 required.");
string root = Path.GetFullPath(args[0]);
using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(root,"manifest.json")));
Tables.Load(Path.Combine(root,"inputs"));
string output = Path.Combine(root,"managed");
if (Directory.Exists(output)) throw new IOException("Output exists.");
Directory.CreateDirectory(output);
var records = new List<object>();
var variants = new (string Name, Run Function)[] { ("Original",Original.LogMelFilterbank), ("Frame",Frame.LogMelFilterbank), ("Spectrum",Spectrum.LogMelFilterbank), ("Output",Output.LogMelFilterbank), ("All",All.LogMelFilterbank) };
foreach (var item in manifest.RootElement.GetProperty("cases").EnumerateArray())
{
    string name = item.GetProperty("name").GetString()!;
    var pcm = Tables.Floats(Path.Combine(root,"inputs",name+".f32"));
    string pcmHash = Evidence.Hash(MemoryMarshal.AsBytes(pcm.AsSpan()));
    var rows = new List<object>();
    foreach (var variant in variants)
    {
        Capture.Start(1+(pcm.Length-400)/160);
        var result = variant.Function(pcm,16000,CancellationToken.None);
        if (!result.Dimensions.SequenceEqual(new[]{1,1+(pcm.Length-400)/160,80})) throw new InvalidDataException("Shape");
        if (Evidence.Hash(MemoryMarshal.AsBytes(pcm.AsSpan()))!=pcmHash) throw new InvalidDataException("Input mutation");
        Tables.Verify();
        string directory=Path.Combine(output,name,variant.Name);Directory.CreateDirectory(directory);
        var stages=Capture.Save(directory);
        string file=Path.Combine(directory,"features.f32");
        using(var stream=new FileStream(file,FileMode.CreateNew)) stream.Write(MemoryMarshal.AsBytes(result.Buffer.Span));
        rows.Add(new { variant=variant.Name,stages,features=Evidence.Pin(file),input_unchanged=true,coefficients_unchanged=true });
    }
    records.Add(new { name, rows, pcm=pcmHash });
    Console.WriteLine(name);
}
bool native=process.Modules.Cast<ProcessModule>().Any(m=>m.FileName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase));
if(native) throw new InvalidOperationException("Unexpected ORT load");
using(var stream=new FileStream(Path.Combine(output,"result.json"),FileMode.CreateNew))
    JsonSerializer.Serialize(stream,new { complete=true,manifest=Evidence.Pin(Path.Combine(root,"manifest.json")),records,
        runtime=new {pid=process.Id,framework=Environment.Version.ToString(),affinity=process.ProcessorAffinity.ToInt64(),processor_count=Environment.ProcessorCount,native_ort_loaded=native},
        core=Evidence.Pin(typeof(DenseTensor<float>).Assembly.Location),producer=Evidence.Pin(typeof(Capture).Assembly.Location) },new JsonSerializerOptions {WriteIndented=true});

delegate DenseTensor<float> Run(ReadOnlySpan<float> pcm,int sampleRate,CancellationToken token);

namespace PrecisionProbe
{
    internal static class Evidence
    {
        public static string Hash(ReadOnlySpan<byte> bytes)=>Convert.ToHexStringLower(SHA256.HashData(bytes));
        public static object Pin(string path) { using var f=File.OpenRead(path);return new {bytes=f.Length,sha256=Convert.ToHexStringLower(SHA256.HashData(f))}; }
    }
    internal static class Tables
    {
        public static float[] Window=Array.Empty<float>(),Mel=Array.Empty<float>();
        private static string windowHash="",melHash="";
        public static float[] Floats(string path)=>MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(path)).ToArray();
        public static void Load(string path)
        {
            Window=Floats(Path.Combine(path,"window.f32"));Mel=Floats(Path.Combine(path,"mel.f32"));
            if(Window.Length!=400||Mel.Length!=20480||!Window.All(float.IsFinite)||!Mel.All(float.IsFinite))throw new InvalidDataException("Coefficient shape/finiteness");
            windowHash=Evidence.Hash(MemoryMarshal.AsBytes(Window.AsSpan()));melHash=Evidence.Hash(MemoryMarshal.AsBytes(Mel.AsSpan()));
        }
        public static void Verify()
        {
            if(windowHash!=Evidence.Hash(MemoryMarshal.AsBytes(Window.AsSpan()))||melHash!=Evidence.Hash(MemoryMarshal.AsBytes(Mel.AsSpan())))throw new InvalidDataException("Coefficient mutation");
        }
    }
    internal static class Capture
    {
        private static Dictionary<string,double[]> values=new();
        private static int frames;
        public static void Start(int count)
        {
            frames=count;values=new Dictionary<string,double[]>();
            foreach(var (name,width) in new[]{("windowed",512),("real",257),("imaginary",257),("power",257),("energy",80),("raw",80),("features",80)})values.Add(name,new double[count*width]);
        }
        public static void Window(int frame,Complex[] array) { for(int i=0;i<512;i++)values["windowed"][frame*512+i]=array[i].Real; }
        public static void Fourier(int frame,Complex[] array) { for(int i=0;i<257;i++){values["real"][frame*257+i]=array[i].Real;values["imaginary"][frame*257+i]=array[i].Imaginary;} }
        public static void Power(int frame,float[] array) { for(int i=0;i<257;i++)values["power"][frame*257+i]=array[i]; }
        public static void Power(int frame,double[] array) { for(int i=0;i<257;i++)values["power"][frame*257+i]=array[i]; }
        public static void Energy(int frame,int mel,double value)=>values["energy"][frame*80+mel]=value;
        public static void Log(int frame,int mel,double value)=>values["raw"][frame*80+mel]=value;
        public static void Features(ReadOnlySpan<float> array) { for(int i=0;i<array.Length;i++)values["features"][i]=array[i]; }
        public static Dictionary<string,object> Save(string path)
        {
            var result=new Dictionary<string,object>();
            foreach(var (name,array) in values)
            {
                if(!array.All(double.IsFinite))throw new InvalidDataException("Nonfinite stage");
                string file=Path.Combine(path,name+".f64");
                using(var stream=new FileStream(file,FileMode.CreateNew))stream.Write(MemoryMarshal.AsBytes(array.AsSpan()));
                int[] shape=name=="features"?new[]{1,frames,80}:new[]{frames,array.Length/frames};
                result.Add(name,new {shape,pin=Evidence.Pin(file)});
            }
            return result;
        }
    }
}
