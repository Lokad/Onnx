using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static void Require(bool ok,string why){if(!ok)throw new InvalidDataException(why);}
static string Hash(ReadOnlySpan<float> data)=>Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(data)));
static string FileHash(string file){using var s=File.OpenRead(file);return Convert.ToHexStringLower(SHA256.HashData(s));}
string capture=Path.GetFullPath(args[0]),output=Path.GetFullPath(args[1]);Require(!File.Exists(output),"Output exists");
Require(Vector<float>.Count==8 && Fma.IsSupported,"Actual eight-lane FMA hardware required");
string core=FileHash(typeof(ComputationalGraph).Assembly.Location);Require(core=="187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4","Wrong core");
using var receiptDocument=JsonDocument.Parse(File.ReadAllBytes(Path.Combine(capture,"closed.json")));
Require(FileHash(Path.Combine(capture,"closed.json"))=="9d23fb9f8b38b43ecbdb333707a7259698d0477533a279613d3bfaffc23a650b","Wrong closed capture");
foreach(var p in receiptDocument.RootElement.GetProperty("files").EnumerateObject())
{
    string file=Path.Combine(capture,p.Name);Require(new FileInfo(file).Length==p.Value.GetProperty("bytes").GetInt64() && FileHash(file)==p.Value.GetProperty("sha256").GetString(),"Changed closed file "+p.Name);
}
var variants=new (string Name,Kernel Run)[]{("CopyA",Kernels.CopyA),("CopyB",Kernels.CopyB),("Conditional",Kernels.Conditional)};
long compared=0;int cases=0;var records=new List<object>();
void Check(float[] input,float[] bias,string name)
{
    var backing=Enumerable.Repeat(-12345.5f,input.Length+11).ToArray();input.CopyTo(backing,3);var x=backing.AsSpan(3,input.Length);
    var biasBacking=Enumerable.Repeat(-12345.5f,bias.Length+9).ToArray();bias.CopyTo(biasBacking,5);var b=biasBacking.AsSpan(5,bias.Length);
    string xHash=Hash(backing),bHash=Hash(biasBacking);var expected=new float[input.Length];Kernels.Product(x,b,expected);
    foreach(var variant in variants)
    {
        var destination=Enumerable.Repeat(-12345.5f,input.Length+13).ToArray();var y=destination.AsSpan(7,input.Length);
        variant.Run(x,b,y);Require(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(y)),variant.Name+" bits: "+name);
        Require(destination.Take(7).Concat(destination.Skip(7+input.Length)).All(v=>v==-12345.5f),"Sentinel changed");
        var inplace=input.ToArray();variant.Run(inplace,b,inplace);Require(Hash(inplace)==Hash(expected),variant.Name+" in-place: "+name);
        Require(Hash(backing)==xHash && Hash(biasBacking)==bHash,"Input/bias changed");compared+=2L*input.Length;
    }
    cases++;
}
var random=new Random(904201);
foreach(int width in new[]{1,7,8,9,15,16,17,31,32,33,384,1536})
foreach(int rows in new[]{0,1,2,7})foreach(int tail in new[]{0,1,7})
    Check(Enumerable.Range(0,width*rows+tail).Select(_=>random.NextSingle()*16-8).ToArray(),Enumerable.Range(0,width).Select(_=>random.NextSingle()-.5f).ToArray(),$"shape {rows}x{width}+{tail}");
float[] special={0f,-0f,float.Epsilon,-float.Epsilon,float.PositiveInfinity,float.NegativeInfinity,float.NaN,BitConverter.Int32BitsToSingle(0x7fa12345),BitConverter.Int32BitsToSingle(unchecked((int)0xffa54321)),float.MaxValue,-float.MaxValue};
foreach(float value in special)
{
    Check(Enumerable.Repeat(value,128).ToArray(),new float[32],"uniform exceptional");
    Check(Enumerable.Range(0,128).Select(i=>special[i%special.Length]).ToArray(),Enumerable.Repeat(value,32).ToArray(),"mixed exceptional");
}
foreach(float center in new[]{0f,.921875f,-.921875f,3.925f,-3.925f})for(int delta=-32;delta<=32;delta++)
{
    float value=center;for(int i=0;i<Math.Abs(delta);i++)value=delta<0?float.BitDecrement(value):float.BitIncrement(value);
    Check(Enumerable.Repeat(value/.7071067811865476f,128).ToArray(),new float[32],"boundary");
}
uint state=9471923;
for(int batch=0;batch<1024;batch++)
{
    var x=new float[2048];for(int i=0;i<x.Length;i++){state^=state<<13;state^=state>>17;state^=state<<5;x[i]=BitConverter.Int32BitsToSingle(unchecked((int)state));}
    Check(x,new float[32],"random bits "+batch);
}
foreach(string name in new[]{"e5-8tok","e5-30tok","e5-30pad128","e5-128tok","e5-512tok"})
for(int layer=0;layer<12;layer++)
{
    string folder=Path.Combine(capture,"capture",name),prefix=layer.ToString("D2");
    var x=MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(Path.Combine(folder,prefix+"-x.f32"))).ToArray();
    var b=MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(Path.Combine(folder,prefix+"-bias.f32"))).ToArray();
    Check(x,b,name+" layer "+layer);records.Add(new{name,layer,values=x.Length,input_sha256=Hash(x),bias_sha256=Hash(b)});
}
using(var stream=new FileStream(output,FileMode.CreateNew))JsonSerializer.Serialize(stream,new{passed=true,cases,compared,core_sha256=core,probe_sha256=FileHash(typeof(Program).Assembly.Location),runtime=RuntimeInformation.FrameworkDescription,width=Vector<float>.Count,fma=Fma.IsSupported,avx512=Avx512F.IsSupported,captures=records},new JsonSerializerOptions{WriteIndented=true});
Console.WriteLine($"Exact proof passed: {cases} cases, {compared} compared values, 60 model captures.");
