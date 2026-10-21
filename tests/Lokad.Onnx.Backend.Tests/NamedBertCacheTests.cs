using System.Reflection;
using FastBertTokenizer;

namespace Lokad.Onnx.Backend.Tests;

public class NamedBertCacheTests
{
    static object Load(string name, Func<string, BertTokenizer> loader) =>
        typeof(Text).GetMethod("GetSharedBertTokenizer", BindingFlags.Static | BindingFlags.NonPublic)!
            .Invoke(null, new object[] { name, loader })!;

    static Task<T> OnDedicatedThread<T>(Func<T> operation) =>
        Task.Factory.StartNew(operation, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

    [Fact]
    public async Task BlockedColdKey_DoesNotBlockCachedOrOtherKeys()
    {
        string prefix = Guid.NewGuid().ToString("N");
        var ready = new BertTokenizer();
        Load(prefix + "-cached", _ => ready);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        // Test cache contention independently of the runner's shared thread pool.
        var slow = OnDedicatedThread(() => Load(prefix + "-slow", _ =>
        {
            entered.Set();
            release.Wait(); // Released by finally, after the probes have completed or timed out.
            return new BertTokenizer();
        }));
        var workers = new List<Task> { slow };
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
            var cached = OnDedicatedThread(() => Text.GetOrLoadBertTokenizer(prefix + "-cached"));
            var other = OnDedicatedThread(() => Load(prefix + "-other", _ => new BertTokenizer()));
            workers.Add(cached);
            workers.Add(other);
            await Task.WhenAll(cached, other).WaitAsync(TimeSpan.FromSeconds(10));
            Assert.False(slow.IsCompleted);
            Assert.Same(ready, await cached);
        }
        finally
        {
            release.Set();
            await Task.WhenAll(workers);
        }
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
