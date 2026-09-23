"""Preserve the fixed timing loops while replacing both legs with actual Winograd DLLs."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
CURRENT='521bae1702849ca23dda586515e7cbabaac2d1eabdff04dc90a7ba76059e93fb'
CANDIDATE='90b2164bbb93419bda814d76d51dca448d5204d3893294564ff07cb7580c8108'


def generate():
    source=(ROOT/'tests/pyannote/winograd-range-screen/Screen.cs').read_text(encoding='utf8')
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    replace('using System.Runtime.InteropServices;','using System.Runtime.InteropServices;\nusing System.Runtime.Loader;')
    replace('    const string ComponentHash = "ffedd38708b086c2620ac27f8ab540880e59d7d66b8a65f050447a80b2f88377";',
            f'    const string CurrentHash = "{CURRENT}";\n    const string CandidateHash = "{CANDIDATE}";')
    replace('''    delegate bool Direct(ReadOnlySpan<float> input,ReadOnlySpan<float> weights,ReadOnlySpan<float> bias,
        ReadOnlySpan<float> residual,Span<float> destination,Span<float> packedInput,Span<float> packedOutput,
        int c,int m,int h,int w,int stride,int lanes,bool relu);
''','')
    replace('CompleteCall(Case call,float[] prepared,bool candidate,Direct direct,Winograd winograd,Plan plan,',
            'CompleteCall(Case call,float[] prepared,Winograd winograd,Plan plan,')
    replace('''        if (candidate)
            Require(plan(call.C,call.M,call.H,call.W,out ni,out np,out no),"candidate scratch budget");
        else { ni=checked(call.C*(call.H+2)*(call.W+2));np=outputCount;no=0; }''',
            '''        Require(plan(call.C,call.M,call.H,call.W,out ni,out np,out no),"Winograd scratch budget");''')
    replace('            if (candidate) d=ArrayPool<float>.Shared.Rent(no);','            d=ArrayPool<float>.Shared.Rent(no);')
    replace('''            bool ok=candidate
                ? winograd(call.Input,prepared,call.Bias,call.Residual,output,a,b,d!,call.C,call.M,call.H,call.W,16,call.Relu)
                : direct(call.Input,prepared,call.Bias,call.Residual,output,a,b,call.C,call.M,call.H,call.W,1,16,call.Relu);''',
            '''            bool ok=winograd(call.Input,prepared,call.Bias,call.Residual,output,a,b,d!,call.C,call.M,call.H,call.W,16,call.Relu);''')
    replace('x.GetProperty("selectedOutput").GetString()!,x.GetProperty("output").GetString()!',
            'x.GetProperty("output").GetString()!,x.GetProperty("output").GetString()!')
    replace('Require(args.Length==5,"component fixtures reference result role");','Require(args.Length==5,"runtimes fixtures reference result role");')
    replace('''        Require(FileHash(args[0])==ComponentHash,"qualified component identity");
        var assembly=Assembly.LoadFrom(args[0]);var type=assembly.GetType("Lokad.Onnx.ConvBlockedSpatial",true)!;
        var prepareDirect=Bind<Prepare>(type,"Prepare");var prepareWinograd=Bind<Prepare>(type,"PrepareWinograd");
        var direct=Bind<Direct>(type,"Execute");var winograd=Bind<Winograd>(type,"ExecuteWinograd");var plan=Bind<Plan>(type,"PlanWinograd");''',
            '''        Require(FileHash(Path.Combine(args[0],"current","Lokad.Onnx.dll"))==CurrentHash,"current product identity");
        Require(FileHash(Path.Combine(args[0],"candidate","Lokad.Onnx.dll"))==CandidateHash,"candidate product identity");
        var loaded=new Dictionary<string,object>();''')
    replace('''            string leg=candidate?"candidate":"current";var prepare=candidate?prepareWinograd:prepareDirect;''',
            '''            string leg=candidate?"candidate":"current";
            string path=Path.GetFullPath(Path.Combine(args[0],leg,"Lokad.Onnx.dll"));
            var context=new AssemblyLoadContext("Winograd-"+leg,isCollectible:false);
            var assembly=context.LoadFromAssemblyPath(path);
            Require(assembly.Location==path,"loaded product path");
            string actualHash=FileHash(assembly.Location);
            Require(actualHash==(candidate?CandidateHash:CurrentHash),"loaded product hash");
            loaded.Add(leg,new {sha256=actualHash,location=assembly.Location});
            var type=assembly.GetType("Lokad.Onnx.ConvBlockedSpatial",true)!;
            var prepare=Bind<Prepare>(type,"PrepareWinograd");
            var winograd=Bind<Winograd>(type,"ExecuteWinograd");var plan=Bind<Plan>(type,"PlanWinograd");''')
    replace('CompleteCall(call,packed[call.Index],candidate,direct,winograd,plan,out long requested,out long rented)',
            'CompleteCall(call,packed[call.Index],winograd,plan,out long requested,out long rented)')
    replace('component=ComponentHash,','products=new {current=CurrentHash,candidate=CandidateHash},loaded,')
    assert source.count('Stopwatch.GetTimestamp()')==4
    return source


if __name__=='__main__':
    target=Path(__file__).with_name('Screen.cs');assert not target.exists()
    target.write_text(generate(),encoding='utf8',newline='\n')
