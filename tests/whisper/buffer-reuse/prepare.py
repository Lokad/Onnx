"""Build and contract-test a private source prototype; leave production source unchanged."""
from pathlib import Path
import difflib,hashlib,json,shutil,subprocess,tarfile

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/whisper-buffer-reuse-20260920'


def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def run(command,log):
    with log.open('x',encoding='utf-8') as stream:r=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    assert r.returncode==0,str(log)


def main():
    assert not BASE.exists();BASE.mkdir();source=BASE/'source';source.mkdir()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=['src/Lokad.Onnx','src/Lokad.Onnx.Data','tests/Lokad.Onnx.Backend.Tests','tests/Shared',
           'global.json','README.md','LICENSE.txt','icon.png','CHANGELOG.md']
    archive=BASE/'original-source.tar'
    subprocess.run(['git','archive','--format=tar','--output',str(archive),revision,*paths],cwd=ROOT,check=True)
    with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
    original={p.relative_to(source).as_posix():pin(p) for p in sorted(source.rglob('*')) if p.is_file()}
    changes={}
    def patch(name,transform):
        path=source/name;old=path.read_text(encoding='utf-8');new=transform(old);assert old!=new
        path.write_text(new,encoding='utf-8');changes[name]=dict(before=original[name],after=pin(path))
        (BASE/(path.stem+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/'+name,tofile='prototype/'+name)),encoding='utf-8')
    def replace_once(text,a,b):
        assert text.count(a)==1,a;return text.replace(a,b)
    overload='''    /// <summary>Creates an isolated context with an explicit budget for already-released arrays.</summary>
    /// <remarks>The limit bounds cached payload, not process memory. Zero disables between-run
    /// retention. Existing outputs, inputs and live aliases are never adopted into this cache.</remarks>
    public GraphExecution CreateExecution(ExecutionOptions? options, long maximumReleasedBufferBytes)
    {
        if (maximumReleasedBufferBytes < 0) throw new ArgumentOutOfRangeException(nameof(maximumReleasedBufferBytes));
        var execution = CreateExecution(options);
        execution.ReleasedBuffers = new ReleasedBufferCache(maximumReleasedBufferBytes, ReleasedBufferCache.DefaultCountLimit);
        return execution;
    }

'''
    patch('src/Lokad.Onnx/ComputationalGraph.cs',lambda s:replace_once(s,'    protected void EnsurePrepared()\n',overload+'    protected void EnsurePrepared()\n'))
    def whisper(s):
        s=replace_once(s,'memory. Each request has independent execution contexts and attention caches.','memory. Execution contexts are reused; each request creates independent attention-cache state.')
        s=replace_once(s,'    readonly WhisperGeneration generation;','    readonly WhisperGeneration generation;\n    readonly GraphExecution encodingExecution, firstExecution, pastExecution;')
        s=replace_once(s,'        RequireInputs(pastDecoder, pastNames);','''        RequireInputs(pastDecoder, pastNames);
        encodingExecution = encoder.CreateExecution(ExecutionOptions.Memory, 512L * 1024 * 1024);
        firstExecution = firstDecoder.CreateExecution(ExecutionOptions.Memory, 128L * 1024 * 1024);
        pastExecution = pastDecoder.CreateExecution(ExecutionOptions.Memory, 128L * 1024 * 1024);''')
        s=replace_once(s,'        var encoding = encoder.CreateExecution(ExecutionOptions.Memory);\n        var first = firstDecoder.CreateExecution(ExecutionOptions.Memory);\n        var past = pastDecoder.CreateExecution(ExecutionOptions.Memory);',
            '        var encoding = encodingExecution;\n        var first = firstExecution;\n        var past = pastExecution;')
        return s
    patch('src/Lokad.Onnx.Data/WhisperTranscriber.cs',whisper)
    tests='''
    [Theory]
    [InlineData(0, false)]
    [InlineData(31, false)]
    [InlineData(32, true)]
    [InlineData(64, true)]
    public void ExplicitContextBudgetControlsRetentionWithoutChangingOwnedOutputs(long budget, bool reuse)
    {
        var context = Chain(true, false).CreateExecution(ExecutionOptions.Memory, budget);
        var first = Feed(8, 1);
        Assert.True(context.Execute(first, false));
        long cold = context.LastPoolAllocatedNewBytes;
        var held = (Tensor<float>)context.Outputs["y"];
        var expected = held.ToArray();
        context.Reset();
        var next = Feed(8, 20);
        Assert.True(context.Execute(next, false));
        Assert.Equal(cold - (reuse ? 32 : 0), context.LastPoolAllocatedNewBytes);
        Assert.Equal(expected, held.ToArray());
        Assert.Equal(Enumerable.Range(20, 8).Select(x => 4f*x), ((Tensor<float>)context.Outputs["y"]).ToArray());
        Assert.Equal(Enumerable.Range(20, 8).Select(x => (float)x), ((Tensor<float>)next["x"]).ToArray());
        Assert.InRange(context.ReleasedBuffers!.Bytes, 0, budget);
        Assert.InRange(context.ReleasedBuffers.Count, 0, ReleasedBufferCache.DefaultCountLimit);
    }

    [Fact]
    public void ExplicitContextBudgetsStayIndependentAndSurviveInvalidation()
    {
        var graph = Chain(true, false);
        var one = graph.CreateExecution(ExecutionOptions.Memory, 32);
        var two = graph.CreateExecution(ExecutionOptions.Memory, 0);
        foreach (var context in new[] { one, two })
        {
            Assert.True(context.Execute(Feed(8, 1), false));
            context.Reset();
            Assert.True(context.Execute(Feed(8, 2), false));
        }
        Assert.NotSame(one.ReleasedBuffers, two.ReleasedBuffers);
        Assert.Equal(32, one.ReleasedBuffers!.Bytes);
        Assert.Equal(0, two.ReleasedBuffers!.Bytes);
        var cache = one.ReleasedBuffers;
        one.Reset();one.RefreshLifetimeAnalysis();
        Assert.Same(cache, one.ReleasedBuffers);Assert.Equal(0, cache.Bytes);
        Assert.True(one.Execute(Feed(13, 1), false)); // 52-byte buffers exceed this context's budget.
        Assert.Equal(0, cache.Bytes);
        one.Reset();Assert.True(one.Execute(Feed(8, 3), false));Assert.Equal(32, cache.Bytes);
        var noPool = ExecutionOptions.Memory with { Tensor = ExecutionOptions.Memory.Tensor with { DisableBufferPool = true } };
        one.Reset();Assert.True(one.Execute(Feed(8, 4), false, ExecutionProvider.CPU, noPool));Assert.Equal(0, cache.Bytes);
    }

    [Fact]
    public void NegativeExplicitContextBudgetIsRejected()
    {
        var graph = Chain(true, false);
        Assert.Throws<ArgumentOutOfRangeException>(() => graph.CreateExecution(null, -1));
        Assert.Null(graph.ReleasedBuffers);
    }
'''
    def add_tests(s):
        i=s.rfind('}');assert not s[i+1:].strip();return s[:i]+tests+s[i:]
    patch('tests/Lokad.Onnx.Backend.Tests/ReleasedBufferCacheTests.cs',add_tests)
    shutil.copyfile(ROOT/'.agent/m4-whisper-buffer-reuse-prototype-20260920.md',BASE/'prospective-plan.md')
    source_files={p.relative_to(source).as_posix():pin(p) for p in sorted(source.rglob('*')) if p.is_file()}
    with (BASE/'source.json').open('x',encoding='utf-8') as f:json.dump(dict(revision=revision,archive=pin(archive),original=original,changes=changes,files=source_files),f,indent=2)
    project=source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
    run(['dotnet','build',str(project),'--tl:off','--nologo','-v','minimal','-c','Release'],BASE/'build.log')
    run(['dotnet','test',str(project),'--tl:off','--nologo','-v','minimal','-c','Release','--no-build','--logger','trx;LogFileName=backend.trx','--results-directory',str(BASE/'test-results')],BASE/'tests.log')
    product=BASE/'product-bin';product.mkdir()
    built=source/'src/Lokad.Onnx.Data/bin/Release/net10.0'
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Google.Protobuf.dll','FastBertTokenizer.dll','Lokad.Tokenizers.dll','SixLabors.ImageSharp.dll']:shutil.copyfile(built/name,product/name)
    with (BASE/'built.json').open('x',encoding='utf-8') as f:json.dump(dict(built=True,tests_passed=True,source=pin(BASE/'source.json'),
        files={p.relative_to(BASE).as_posix():pin(p) for p in [BASE/'build.log',BASE/'tests.log',*sorted(product.iterdir()),*sorted((BASE/'test-results').rglob('*.trx'))]}),f,indent=2)
    print(json.dumps(dict(built=True,tests_passed=True,source_revision=revision,changed_files=list(changes),receipt=pin(BASE/'built.json'))))


if __name__=='__main__':main()
