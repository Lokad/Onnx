// Correctness cases retained from the closed 1ab186d maximum-reduction proof.
using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using SoftmaxReduction;

if(args.Length!=3)throw new ArgumentException("Probe <new-output.json> <check|control|compare> <order0..7>");
string phase=args[1];bool checkOnly=phase=="check";int order=int.Parse(args[2]);
Require(phase is "check" or "control" or "compare","Phase");Require(order>=0&&order<8,"Order");
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
int maximumCases=0,tensorCases=0,inPlaceCases=0,refusals=0;double maximumDoubleError=0;
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
                maximumDoubleError=Math.Max(maximumDoubleError,error);Require(error<=1e-6,"Independent normalization differs");
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
var records=new List<object>(24);
int[] sequence=Enumerable.Range(0,4).Select(i=>(i+order%4)%4).ToArray();if(order>=4)Array.Reverse(sequence);
if(!checkOnly)
{
    Require(OperatingSystem.IsLinux(),"Timing requires Linux CPU clocks");
    Kernel copied=Kernels.Original;
    Kernel[] timing={actual,copied,copied,phase=="control"?copied:Kernels.Reduced};
    string[] names={"actual","copied","duplicate","probe"};
    var specs=new[]{("8",8,8), ("30",30,30), ("pad128",128,30), ("128",128,128), ("512",512,512), ("pad512",512,30)};
    var sets=specs.Select(s=>new Data(s.Item1,s.Item2,s.Item3)).ToArray();
    var samples=new Measurement[sets.Length,4,9];
    var conditioning=new long[sets.Length,4];
    var initialInput=sets.Select(d=>Hash(d.Input)).ToArray();var initialMask=sets.Select(d=>Hash(d.Mask)).ToArray();
    // Condition the same measurement call site and complete batches used below.
    // All conditioning finishes before any retained batch; no serialization or hashes between batches.
    for(int shape=0;shape<sets.Length;shape++)foreach(int mode in sequence)
    {
        var d=sets[shape];long stop=Stopwatch.GetTimestamp()+Stopwatch.Frequency;
        do{Measurements.Run(timing[mode],d);conditioning[shape,mode]+=d.Iterations;}while(Stopwatch.GetTimestamp()<stop);
    }
    for(int shape=0;shape<sets.Length;shape++)
        for(int sample=0;sample<9;sample++)foreach(int mode in sequence)
            samples[shape,mode,sample]=Measurements.Run(timing[mode],sets[shape]);
    for(int shape=0;shape<sets.Length;shape++)
    {
        var d=sets[shape];var expected=new float[d.Input.Length];actual(d.Input,d.Mask,expected,d.Rows,d.Columns,true);
        Require(Hash(d.Input)==initialInput[shape]&&Hash(d.Mask)==initialMask[shape],"Measured input changed");
        foreach(int mode in Enumerable.Range(0,4))
        {
            timing[mode](d.Input,d.Mask,d.Output,d.Rows,d.Columns,true);Require(Bits(expected,d.Output),"Measured output differs");
            records.Add(new{name=d.Name,mode=names[mode],implementation=timing[mode].Method.DeclaringType!.FullName+"."+timing[mode].Method.Name,
                rows=d.Rows,columns=d.Columns,iterations=d.Iterations,conditioning_calls=conditioning[shape,mode],
                samples=Enumerable.Range(0,9).Select(s=>new{ticks=samples[shape,mode,s].Ticks,thread_ns=samples[shape,mode,s].Thread,
                    process_ns=samples[shape,mode,s].Process,gc=new[]{samples[shape,mode,s].G0,samples[shape,mode,s].G1,samples[shape,mode,s].G2}}).ToArray(),
                input_sha256=Hash(d.Input),mask_sha256=Hash(d.Mask),output_sha256=Hash(d.Output)});
        }
    }
}
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
long affinity=OperatingSystem.IsWindows()||OperatingSystem.IsLinux()?Process.GetCurrentProcess().ProcessorAffinity.ToInt64():0;
using(var stream=new FileStream(output,FileMode.CreateNew))JsonSerializer.Serialize(stream,new{schema=2,protocol="softmax-reduction-batches-v1",phase,order,sequence,checkOnly,width=Vector<float>.Count,maximumCases,tensorCases,inPlaceCases,refusals,maximumDoubleError,
    affinity,core_sha256=Sha(typeof(Tensor<float>).Assembly.Location),probe_sha256=Sha(Assembly.GetExecutingAssembly().Location),runtime=RuntimeInformation.FrameworkDescription,frequency=Stopwatch.Frequency,records},new JsonSerializerOptions{WriteIndented=true});
Console.WriteLine($"Exact: {maximumCases} maxima; {tensorCases} tensors; {inPlaceCases} in-place; {refusals} refusals; {records.Count} timing records.");

delegate float Maximum(Span<float> input,int start,Span<float> mask,int count,bool simd);
delegate void Kernel(Span<float> input,Span<float> mask,Span<float> output,int rows,int columns,bool simd);
sealed class Data
{
    internal readonly string Name;internal readonly int Rows,Columns,Iterations;internal readonly float[] Input,Mask,Output;
    internal Data(string name,int columns,int active)
    {
        Name=name;Columns=columns;Rows=12*columns;Iterations=columns==8?32768:columns==30?4096:columns==128?256:32;
        var random=new Random(91+columns);Input=Enumerable.Range(0,Rows*columns).Select(_=>random.NextSingle()*20-10).ToArray();Mask=new float[columns];Output=new float[Input.Length];
        for(int j=active;j<columns;j++)Mask[j]=float.MinValue;
    }
}
static class Clock
{
    [StructLayout(LayoutKind.Sequential)]struct Timespec{public long Seconds,Nanoseconds;}
    [DllImport("libc",EntryPoint="clock_gettime")]static extern int ReadClock(int id,out Timespec value);
    internal static long Read(int id){if(ReadClock(id,out var value)!=0)throw new InvalidOperationException("clock_gettime failed");return value.Seconds*1_000_000_000+value.Nanoseconds;}
}

readonly record struct Measurement(long Ticks,long Thread,long Process,int G0,int G1,int G2);
static class Measurements
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    internal static Measurement Run(Kernel run,Data d)
    {
        int g0=GC.CollectionCount(0),g1=GC.CollectionCount(1),g2=GC.CollectionCount(2);
        long pc=Clock.Read(2),tc=Clock.Read(3),start=Stopwatch.GetTimestamp();
        for(int i=0;i<d.Iterations;i++)run(d.Input,d.Mask,d.Output,d.Rows,d.Columns,true);
        long ticks=Stopwatch.GetTimestamp()-start,thread=Clock.Read(3)-tc,process=Clock.Read(2)-pc;
        return new Measurement(ticks,thread,process,GC.CollectionCount(0)-g0,GC.CollectionCount(1)-g1,GC.CollectionCount(2)-g2);
    }
}
