using System.Reflection;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx.Backend.Tests;

[Collection("NonParallelAllocation")]
public class ParakeetRecordingCancellationTests
{
    static readonly FieldInfo GateField = typeof(ParakeetTranscriber)
        .GetField("gate", BindingFlags.Instance | BindingFlags.NonPublic)!;

    // Cancellation and empty recordings never read the model fields. Supply
    // only the gate to exercise the real API without loading large assets.
    static (ParakeetTranscriber Transcriber, object Gate) Fixture()
    {
        var transcriber = (ParakeetTranscriber)RuntimeHelpers.GetUninitializedObject(typeof(ParakeetTranscriber));
        var gate = new object();
        GateField.SetValue(transcriber, gate);
        return (transcriber, gate);
    }

    static ParakeetRecording Record(ParakeetTranscriber transcriber, float[] pcm, CancellationToken cancellation) =>
        transcriber.TranscribeRecording(pcm, ParakeetTranscriber.SampleRate, ParakeetRecordingOptions.Default, cancellation);

    static void AssertEmpty(ParakeetRecording recording)
    {
        Assert.Empty(recording.Text);
        Assert.Empty(recording.Windows);
        Assert.Equal(ParakeetRecordingStopReason.Completed, recording.StopReason);
        Assert.Equal(0, recording.DurationSeconds);
        Assert.Equal(0, recording.ProcessedSeconds);
    }

    [Fact]
    public void EmptyRecordingCompletesWithoutInference()
    {
        var (transcriber, _) = Fixture();
        AssertEmpty(Record(transcriber, Array.Empty<float>(), CancellationToken.None));
    }

    [Fact]
    public void AlreadyCancelledRecordingDoesNotWaitForTheGate()
    {
        var (transcriber, gate) = Fixture();
        var cancellation = new CancellationToken(true);
        using var finished = new ManualResetEventSlim();
        OperationCanceledException? canceled = null;
        Exception? error = null;
        var caller = new Thread(() =>
        {
            try { Record(transcriber, Array.Empty<float>(), cancellation); }
            catch (OperationCanceledException ex) { canceled = ex; }
            catch (Exception ex) { error = ex; }
            finally { finished.Set(); }
        });
        Monitor.Enter(gate);
        try
        {
            caller.Start();
            Assert.True(finished.Wait(TimeSpan.FromSeconds(10)), "Pre-cancelled caller waited for the gate.");
        }
        finally
        {
            Monitor.Exit(gate);
            Assert.True(caller.Join(10_000), "Owned caller did not finish.");
        }
        Assert.Null(error);
        Assert.NotNull(canceled);
        Assert.Equal(cancellation, canceled.CancellationToken);
    }

    [Theory]
    [InlineData(4_096, 1)]
    [InlineData(4_096, 3)]
    [InlineData(1_048_576, 1)]
    [InlineData(1_048_576, 3)]
    public void CancelledQueuedRecordingsDoNotAllocateAudioCopies(int length, int callers)
    {
        var (transcriber, gate) = Fixture();
        var pcm = Enumerable.Repeat(.125f, length).ToArray();
        var original = pcm.ToArray();
        using var cancellation = new CancellationTokenSource();
        using var ready = new CountdownEvent(callers);
        var threads = new Thread[callers];
        var allocated = new long[callers];
        var canceled = new OperationCanceledException?[callers];
        var errors = new Exception?[callers];
        Monitor.Enter(gate);
        try
        {
            for (int i = 0; i < callers; i++)
            {
                int index = i;
                threads[i] = new Thread(() =>
                {
                    long before = -1;
                    bool signaled = false;
                    try
                    {
                        var (warmup, _) = Fixture();
                        var preCancelled = new CancellationToken(true);
                        for (int warm = 0; warm < 256; warm++)
                        {
                            Record(warmup, Array.Empty<float>(), CancellationToken.None);
                            try { Record(warmup, Array.Empty<float>(), preCancelled); }
                            catch (OperationCanceledException) { }
                        }
                        ready.Signal();
                        signaled = true;
                        before = GC.GetAllocatedBytesForCurrentThread();
                        Record(transcriber, pcm, cancellation.Token);
                    }
                    catch (OperationCanceledException ex) { canceled[index] = ex; }
                    catch (Exception ex) { errors[index] = ex; }
                    finally
                    {
                        if (before >= 0) allocated[index] = GC.GetAllocatedBytesForCurrentThread() - before;
                        if (!signaled) ready.Signal();
                    }
                });
                threads[i].Start();
            }
            Assert.True(ready.Wait(TimeSpan.FromSeconds(10)), "Callers did not finish warmup.");
            Assert.All(errors, error => Assert.Null(error));
            Assert.True(SpinWait.SpinUntil(() => threads.All(t => (t.ThreadState & ThreadState.WaitSleepJoin) != 0), 10_000),
                "Callers did not reach the held serialization gate.");
            cancellation.Cancel();
        }
        finally
        {
            cancellation.Cancel();
            Monitor.Exit(gate);
            bool joined = true;
            foreach (var thread in threads)
                if (thread is not null) joined &= thread.Join(10_000);
            Assert.True(joined, "Owned caller did not finish.");
        }
        Assert.All(errors, error => Assert.Null(error));
        Assert.All(canceled, error =>
        {
            Assert.NotNull(error);
            Assert.Equal(cancellation.Token, error.CancellationToken);
        });
        Assert.Equal(original, pcm);
        AssertEmpty(Record(transcriber, Array.Empty<float>(), CancellationToken.None));
        Assert.All(allocated, bytes => Assert.InRange(bytes, 0, 8_192));
    }
}
