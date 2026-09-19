using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using SoftmaxReduction;

if(args.Length is <1 or >2 || (args.Length==2&&args[1]!="--condition"))throw new ArgumentException("Probe <new-correctness.json> [--condition]");
string output=Path.GetFullPath(args[0]);if(File.Exists(output))throw new IOException("Existing output");
static void Require(bool value,string message){if(!value)throw new InvalidDataException(message);}
static string Sha(string path){using var f=File.OpenRead(path);return Convert.ToHexStringLower(SHA256.HashData(f));}
static string Hash(ReadOnlySpan<float> values)=>Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values)));
static bool Bits(ReadOnlySpan<float> left,ReadOnlySpan<float> right)=>MemoryMarshal.AsBytes(left).SequenceEqual(MemoryMarshal.AsBytes(right));
Require(Sha(typeof(Tensor<float>).Assembly.Location)=="187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4","Frozen core differs");
var actualMax=typeof(Tensor<float>).GetMethod("SoftmaxContiguousMaxMasked",BindingFlags.NonPublic|BindingFlags.Static)!.CreateDelegate<Maximum>();
var actual=typeof(Tensor<float>).GetMethod("SoftmaxMaskedFloatSpanPtr",BindingFlags.NonPublic|BindingFlags.Static)!.CreateDelegate<Kernel>();
Maximum[] maxima={actualMax,Kernels.OriginalMax,Kernels.ReducedMax};
Kernel[] kernels={actual,Kernels.Original,Kernels.Reduced};
bool conditioned=args.Length==2;
if(conditioned)
{
    // Exercise normal tiering; these calls are not timing observations.
    var input=Enumerable.Range(0,4*128).Select(i=>(i%31-15)*0.1f).ToArray();
    var mask=new float[128];var destination=new float[input.Length];
    foreach(var method in maxima)
    {
        var watch=Stopwatch.StartNew();float sink=0;int calls=0;
        do
        {
            sink+=method(input,0,mask,128,true);
            if((++calls&1023)==0)Thread.Sleep(1);
        }while(watch.Elapsed.TotalSeconds<1);
        Require(float.IsFinite(sink),"Conditioning maximum differs");
    }
    foreach(var method in kernels)
    {
        var watch=Stopwatch.StartNew();int calls=0;
        do
        {
            method(input,mask,destination,4,128,true);
            if((++calls&1023)==0)Thread.Sleep(1);
        }while(watch.Elapsed.TotalSeconds<1);
        Require(destination.All(float.IsFinite),"Conditioning kernel differs");
    }
}
int maximumCases=0,tensorCases=0,inPlaceCases=0,refusals=0;double independentMaximumError=0;
uint state=0x71A32B91;
uint Next(){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
void CompareMaximum(float[] input,float[] mask,int start,int count,bool simd)
{
    string inputHash=Hash(input),maskHash=Hash(mask);
    int expected=BitConverter.SingleToInt32Bits(actualMax(input,start,mask,count,simd));
    foreach(var method in maxima.Skip(1))
        Require(BitConverter.SingleToInt32Bits(method(input,start,mask,count,simd))==expected,$"Maximum differs {start}/{count}/{simd}/{method.Method.Name}");
    Require(Hash(input)==inputHash&&Hash(mask)==maskHash,"Maximum input changed");maximumCases++;
}
foreach(int count in new[]{0,1,7,8,9,15,16,17,30,31,32,33,127,128,129,511,512,513})
foreach(int start in new[]{0,1,5})foreach(bool simd in new[]{false,true})
{
    var input=new float[start+count+5];var mask=new float[count+3];
    for(int attempt=0;attempt<64;attempt++)
    {
        for(int i=0;i<input.Length;i++)input[i]=BitConverter.Int32BitsToSingle((int)Next());
        for(int i=0;i<mask.Length;i++)mask[i]=attempt%2==0?0:BitConverter.Int32BitsToSingle((int)Next());
        CompareMaximum(input,mask,start,count,simd);
    }
    foreach(float exceptional in new[]{0f,BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)),float.PositiveInfinity,float.NegativeInfinity,
        BitConverter.Int32BitsToSingle(0x7FA12345),BitConverter.Int32BitsToSingle(unchecked((int)0xFFC12345)),float.MaxValue,float.MinValue})
    {
        Array.Fill(input,-1);Array.Clear(mask);
        for(int lane=0;lane<count;lane++)
        {
            input[start+lane]=exceptional;CompareMaximum(input,mask,start,count,simd);input[start+lane]=-1;
        }
        Array.Fill(input,exceptional);CompareMaximum(input,mask,start,count,simd);
    }
}
foreach(int columns in new[]{0,1,7,8,9,15,16,17,30,31,32,33,127,128,129,511,512,513})
foreach(int rows in new[]{0,1,2,3})foreach(int pattern in Enumerable.Range(0,12))foreach(bool simd in new[]{false,true})
{
    var random=new Random(rows*columns+pattern);var input=Enumerable.Range(0,rows*columns).Select(_=>random.NextSingle()*160-80).ToArray();var mask=new float[columns];
    if(pattern==1)for(int j=columns/2;j<columns;j++)mask[j]=float.MinValue;
    if(pattern==2)for(int j=0;j<columns;j+=2)mask[j]=-10000;
    if(pattern==3)for(int j=0;j<columns;j++)mask[j]=-4*random.NextSingle();
    if(pattern==4&&input.Length>0)input[0]=float.PositiveInfinity;
    if(pattern==5)Array.Fill(input,float.NegativeInfinity);
    if(pattern==6&&input.Length>0)input[^1]=BitConverter.Int32BitsToSingle(0x7FA12345);
    if(pattern==7){Array.Fill(input,float.MaxValue);Array.Fill(mask,float.MaxValue);}
    if(pattern==8)for(int j=0;j<input.Length;j++)input[j]=j%2==0?0f:BitConverter.Int32BitsToSingle(unchecked((int)0x80000000));
    if(pattern==9){Array.Fill(input,10001f);Array.Fill(mask,-10000f);}
    if(pattern==10){Array.Fill(input,1f);Array.Fill(mask,float.MinValue);}
    if(pattern==11)for(int j=0;j<mask.Length;j++)mask[j]=j%2==0?BitConverter.Int32BitsToSingle(0x7FA12345):0;
    var expected=new float[input.Length];actual(input,mask,expected,rows,columns,simd);
    foreach(var method in kernels.Skip(1))
    {
        var backing=Enumerable.Repeat(-12345.5f,input.Length+10).ToArray();var destination=backing.AsSpan(5,input.Length);
        var guardedInput=Enumerable.Repeat(123f,input.Length+10).ToArray();input.CopyTo(guardedInput,5);
        var guardedMask=Enumerable.Repeat(456f,mask.Length+6).ToArray();mask.CopyTo(guardedMask,3);
        string inputHash=Hash(guardedInput),maskHash=Hash(guardedMask);
        method(guardedInput.AsSpan(5,input.Length),guardedMask.AsSpan(3,mask.Length),destination,rows,columns,simd);
        Require(Bits(expected,destination),$"Kernel bits differ {rows}/{columns}/{pattern}/{simd}/{method.Method.Name}");
        Require(Hash(guardedInput)==inputHash&&Hash(guardedMask)==maskHash,"Kernel input/mask changed");
        Require(backing.Take(5).Concat(backing.Skip(input.Length+5)).All(v=>v==-12345.5f),"Destination guard changed");
        var inPlace=(float[])input.Clone();method(inPlace,mask,inPlace,rows,columns,simd);
        Require(Bits(expected,inPlace),"In-place output differs");inPlaceCases++;
    }
    if(pattern<4&&columns>0)
    {
        for(int row=0;row<rows;row++)
        {
            var values=new double[columns];for(int j=0;j<columns;j++)values[j]=(float)(input[row*columns+j]+mask[j]);
            double maximum=values.Max(),sum=values.Sum(v=>Math.Exp(v-maximum));
            for(int j=0;j<columns;j++)
            {
                double error=Math.Abs(expected[row*columns+j]-Math.Exp(values[j]-maximum)/sum);
                independentMaximumError=Math.Max(independentMaximumError,error);Require(error<=1e-6,"Independent normalization differs");
            }
        }
    }
    tensorCases++;
}
foreach(var method in kernels)
{
    var destination=Enumerable.Repeat(123f,8).ToArray();
    try{method(new float[8],new float[7],destination,1,8,true);throw new InvalidDataException("Short mask accepted");}
    catch(ArgumentException){Require(destination.All(v=>v==123f),"Refusal wrote output");refusals++;}
}
long affinity=OperatingSystem.IsWindows()||OperatingSystem.IsLinux()?Process.GetCurrentProcess().ProcessorAffinity.ToInt64():0;
using(var stream=new FileStream(output,FileMode.CreateNew))JsonSerializer.Serialize(stream,new{schema=1,scope="Standalone exactness only; no timing claim",conditioned,maximumCases,tensorCases,inPlaceCases,refusals,independentMaximumError,
    width=Vector<float>.Count,affinity,runtime=RuntimeInformation.FrameworkDescription,core_sha256=Sha(typeof(Tensor<float>).Assembly.Location),probe_sha256=Sha(Assembly.GetExecutingAssembly().Location),
    flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_")||k.StartsWith("DOTNET_")||k.StartsWith("COMPlus_")).ToDictionary(k=>k,Environment.GetEnvironmentVariable)},new JsonSerializerOptions{WriteIndented=true});
Console.WriteLine($"Exact: {maximumCases} maxima; {tensorCases} tensors; {inPlaceCases} in-place checks; {refusals} refusals; double error {independentMaximumError:R}");

delegate float Maximum(Span<float> input,int start,Span<float> mask,int count,bool simd);
delegate void Kernel(Span<float> input,Span<float> mask,Span<float> output,int rows,int columns,bool simd);
