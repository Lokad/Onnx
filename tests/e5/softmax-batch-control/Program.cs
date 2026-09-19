using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using ZeroBlocks;

if(args.Length!=3) throw new ArgumentException("Probe <new-output.json> <check|control|compare> <order0..7>");
string output=Path.GetFullPath(args[0]);if(File.Exists(output))throw new IOException("Existing output");
string phase=args[1];bool checkOnly=phase=="check";int order=int.Parse(args[2]);Require(order>=0&&order<8,"Order");
Require(phase is "check" or "control" or "compare","Phase");
static void Require(bool value,string message){if(!value)throw new InvalidDataException(message);}
static string Sha(string path){using var f=File.OpenRead(path);return Convert.ToHexStringLower(SHA256.HashData(f));}
static string Hash(float[] values)=>Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
static bool Bits(ReadOnlySpan<float> a,ReadOnlySpan<float> b)=>MemoryMarshal.AsBytes(a).SequenceEqual(MemoryMarshal.AsBytes(b));
var actual=typeof(Tensor<float>).GetMethod("SoftmaxMaskedFloatSpanPtr",BindingFlags.NonPublic|BindingFlags.Static)!.CreateDelegate<Kernel>();
Kernel[] modes={actual,Kernels.Original,Kernels.Candidate,Kernels.Adaptive};
Require(Sha(typeof(Tensor<float>).Assembly.Location)=="7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9","Frozen core differs");
int width=Vector<float>.Count,expValues=0,tensorCases=0,refusals=0;double maximumDoubleError=0;
var vector=new float[width];var reference=new float[width];var candidate=new float[width];
void Exp()
{
    var x=new Vector<float>(vector);Kernels.OriginalExp(x).CopyTo(reference);Kernels.CandidateExp(x).CopyTo(candidate);
    Require(Bits(reference,candidate),"Exponential bits differ");expValues+=width;
}
uint state=123456789;
for(int i=0;i<2_000_000;i+=width)
{
    for(int j=0;j<width;j++){state^=state<<13;state^=state>>17;state^=state<<5;vector[j]=BitConverter.Int32BitsToSingle((int)(state|0x80000000));}Exp();
}
for(int i=0;i<500_000;i+=width){for(int j=0;j<width;j++)vector[j]=-(i+j)*.0002f;Exp();}
foreach(float center in new[]{0f,-0f,-88.722839f,-87.33655f,-.34657359f,-.69314718f,float.NegativeInfinity,float.NaN})
for(int shift=-16;shift<=16;shift++)
{
    float value=center;for(int i=0;i<Math.Abs(shift);i++)value=shift<0?float.BitDecrement(value):float.BitIncrement(value);
    if(value>0)continue;Array.Fill(vector,value);Exp();vector[width-1]=0;Exp();
}
foreach(int columns in new[]{1,7,8,9,15,16,17,30,31,32,33,127,128,129,511,512,513})
foreach(int rows in new[]{1,2,3})foreach(int pattern in Enumerable.Range(0,9))foreach(bool simd in new[]{false,true})
{
    var random=new Random(rows*columns+pattern);var input=Enumerable.Range(0,rows*columns).Select(_=>random.NextSingle()*160-80).ToArray();var mask=new float[columns];
    if(pattern==1)for(int j=columns/2;j<columns;j++)mask[j]=float.MinValue;
    if(pattern==2)for(int j=0;j<columns;j+=2)mask[j]=-10000;
    if(pattern==3)for(int j=0;j<columns;j++)mask[j]=-4*random.NextSingle();
    if(pattern==4)input[0]=float.PositiveInfinity;
    if(pattern==5)Array.Fill(input,float.NegativeInfinity);
    if(pattern==6)input[^1]=BitConverter.Int32BitsToSingle(0x7fa12345);
    if(pattern==7){Array.Fill(input,float.MaxValue);Array.Fill(mask,float.MaxValue);}
    if(pattern==8){for(int j=0;j<input.Length;j++)input[j]=j%columns==0?0:float.BitDecrement(-88.722839f);}
    string inputHash=Hash(input),maskHash=Hash(mask);var expected=new float[input.Length];actual(input,mask,expected,rows,columns,simd);
    foreach(var mode in modes.Skip(1))
    {
        var backing=Enumerable.Repeat(-12345.5f,input.Length+10).ToArray();var result=backing.AsSpan(5,input.Length);
        mode(input,mask,result,rows,columns,simd);Require(Bits(expected,result),$"Softmax bits differ: {rows}/{columns}/{pattern}/{simd}/{mode.Method.Name}");
        Require(backing.Take(5).Concat(backing.Skip(input.Length+5)).All(v=>v==-12345.5f),"Output guard changed");
        Require(Hash(input)==inputHash&&Hash(mask)==maskHash,"Input changed");
    }
    if(pattern<4)
    {
        for(int row=0;row<rows;row++)
        {
            var values=new double[columns];for(int j=0;j<columns;j++)values[j]=(float)(input[row*columns+j]+mask[j]);
            double maximum=values.Max(),sum=values.Sum(v=>Math.Exp(v-maximum));
            for(int j=0;j<columns;j++){double error=Math.Abs(expected[row*columns+j]-Math.Exp(values[j]-maximum)/sum);maximumDoubleError=Math.Max(maximumDoubleError,error);Require(error<=1e-6,"Independent softmax error");}
        }
    }
    tensorCases++;
}
foreach(var mode in modes)
{
    try{mode(new float[8],new float[7],new float[8],1,8,true);throw new InvalidDataException("Short mask accepted");}
    catch(ArgumentException){refusals++;}
}
var records=new List<object>(24);
int[] sequence=Enumerable.Range(0,4).Select(i=>(i+order%4)%4).ToArray();if(order>=4)Array.Reverse(sequence);
if(!checkOnly)
{
    Require(OperatingSystem.IsLinux(),"Timing requires Linux CPU clocks");
    Kernel copied=Kernels.Original;
    Kernel[] timing={actual,copied,copied,phase=="control"?copied:Kernels.Adaptive};
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
using(var f=new FileStream(output,FileMode.CreateNew))JsonSerializer.Serialize(f,new{schema=2,protocol="softmax-long-batches-v1",phase,order,sequence,checkOnly,width,expValues,tensorCases,refusals,maximumDoubleError,
    core_sha256=Sha(typeof(Tensor<float>).Assembly.Location),probe_sha256=Sha(Assembly.GetExecutingAssembly().Location),runtime=RuntimeInformation.FrameworkDescription,
    frequency=Stopwatch.Frequency,records},new JsonSerializerOptions{WriteIndented=true});
Console.WriteLine($"Passed {expValues} exponential lanes, {tensorCases} tensors, {refusals} refusals; {records.Count} timing records.");

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
