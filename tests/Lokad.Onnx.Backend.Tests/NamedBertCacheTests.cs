using System.Reflection;
using FastBertTokenizer;

namespace Lokad.Onnx.Backend.Tests;

public class NamedBertCacheTests
{
    static object Load(string name, Func<string, BertTokenizer> loader) =>
        typeof(Text).GetMethod("GetSharedBertTokenizer", BindingFlags.Static | BindingFlags.NonPublic)!
            .Invoke(null, new object[] { name, loader })!;

    [Fact]
    public async Task BlockedColdKey_DoesNotBlockCachedOrOtherKeys()
    {
        string prefix = Guid.NewGuid().ToString("N");
        var ready = new BertTokenizer();
        Load(prefix + "-cached", _ => ready);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var slow = Task.Run(() => Load(prefix + "-slow", _ =>
        {
            entered.Set();
            if (!release.Wait(TimeSpan.FromSeconds(15))) throw new TimeoutException();
            return new BertTokenizer();
        }));
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
            Assert.Same(ready, await Task.Run(() => Text.GetOrLoadBertTokenizer(prefix + "-cached")).WaitAsync(TimeSpan.FromSeconds(10)));
            await Task.Run(() => Load(prefix + "-other", _ => new BertTokenizer())).WaitAsync(TimeSpan.FromSeconds(10));
        }
        finally { release.Set(); }
        await slow;
    }

    [Fact]
    public async Task SameKey_LoadsOnce()
    {
        string name = Guid.NewGuid().ToString("N");
        int loads = 0;
        var results = await Task.WhenAll(Enumerable.Range(0, 8).Select(_ => Task.Run(() => Load(name, _ =>
        {
            Interlocked.Increment(ref loads);
            return new BertTokenizer();
        }))));
        Assert.Equal(1, loads);
        Assert.All(results, result => Assert.Same(results[0], result));
    }

    [Fact]
    public void FailedLoad_IsRetryable()
    {
        string name = Guid.NewGuid().ToString("N");
        var error = Assert.Throws<TargetInvocationException>(() => Load(name, _ => throw new IOException("fixture failure")));
        Assert.IsType<IOException>(error.InnerException);
        var tokenizer = new BertTokenizer();
        Load(name, _ => tokenizer);
        Assert.Same(tokenizer, Text.GetOrLoadBertTokenizer(name));
    }
}
