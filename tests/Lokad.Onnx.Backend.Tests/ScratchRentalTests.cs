using System;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.Linq;
using System.Threading;

namespace Lokad.Onnx.Backend.Tests;

public class ScratchRentalTests
{
    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void DirectConvolutionReturnsScratchWhenReporterThrows(bool doubles, bool fatal)
    {
        ITensor input = doubles ? DenseTensor<double>.Ones([1, 2, 8, 8]) : DenseTensor<float>.Ones([1, 2, 8, 8]);
        ITensor weight = doubles ? DenseTensor<double>.Ones([3, 2, 3, 3]) : DenseTensor<float>.Ones([3, 2, 3, 3]);
        Exception failure = fatal ? new OutOfMemoryException("reporter failure") : new InvalidOperationException("reporter failure");
        using var events = new Rentals();
        ITensor? output = null;
        foreach (var error in new Exception?[] { null, failure, null })
        {
            var reporter = new Reporter(error);
            var observed = events.Capture(() =>
            {
                var options = TensorExecutionOptions.Scalar with { ScratchReporter = reporter };
                output = doubles
                    ? Tensor<double>.Conv2D((Tensor<double>)input, (Tensor<double>)weight, 1, [0, 0, 0, 0], null, null, [1, 1], null, options)
                    : Tensor<float>.Conv2D((Tensor<float>)input, (Tensor<float>)weight, 1, [0, 0, 0, 0], null, null, [1, 1], null, options);
            });
            Assert.Same(error, observed.Error);
            Assert.Equal(1, reporter.Calls);
            Assert.Equal(648L * (doubles ? sizeof(double) : sizeof(float)), reporter.TotalScratchBytes);
            if (error is null) CheckOutput(output!);
            Balanced(observed.Events);
        }
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void GraphReturnsScratchAndRetainsReporterFailurePolicy(bool explicitContext, bool fatal)
    {
        var graph = Model.Load(new OnnxModel
        {
            Opset = { [""] = 18 },
            Inputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [1, 2, 8, 8] },
                       new OnnxValueInfo { Name = "w", ElementType = TensorElementType.Float, Dims = [3, 2, 3, 3] } },
            Outputs = { new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = [1, 3, 6, 6] } },
            Nodes = { new OnnxNode { Name = "conv", OpType = "Conv", Inputs = ["x", "w"], Outputs = ["y"] } }
        });
        var feed = new Dictionary<string, ITensor>
        {
            ["x"] = DenseTensor<float>.Ones([1, 2, 8, 8]),
            ["w"] = DenseTensor<float>.Ones([3, 2, 3, 3])
        };
        ComputationalGraph execution = explicitContext ? graph.CreateExecution(ExecutionOptions.Scalar) : graph;
        Exception failure = fatal ? new OutOfMemoryException("reporter failure") : new InvalidOperationException("reporter failure");
        using var events = new Rentals();
        foreach (var error in new Exception?[] { null, failure, null })
        {
            var reporter = new Reporter(error);
            bool success = false;
            var observed = events.Capture(() => success = execution.Execute(feed, true, ExecutionProvider.CPU,
                ExecutionOptions.Scalar with { Tensor = TensorExecutionOptions.Scalar with { ScratchReporter = reporter } }));
            Assert.Equal(1, reporter.Calls);
            Assert.Equal(648L * sizeof(float), reporter.TotalScratchBytes);
            if (error is null)
            {
                Assert.Null(observed.Error);
                Assert.True(success, execution.LastErrorMessage);
                Assert.Equal(648L * sizeof(float), execution.LastScratchBytes);
                CheckOutput(execution.Outputs["y"]);
            }
            else if (fatal) Assert.Same(failure, observed.Error);
            else
            {
                Assert.Null(observed.Error);
                Assert.False(success);
                Assert.Same(failure, execution.LastErrorCause);
                Assert.Empty(execution.Outputs);
                Assert.Equal(648L * sizeof(float), execution.LastScratchBytes);
            }
            Balanced(observed.Events);
        }
    }

    static void CheckOutput(ITensor output)
    {
        Assert.Equal(new[] { 1, 3, 6, 6 }, output.Dims);
        Assert.All(output.ToArray().Cast<object>(), value => Assert.Equal(18d, Convert.ToDouble(value)));
    }

    static void Balanced(Rental[] events)
    {
        Assert.Contains(events, e => e.Rented && e.Length >= 648);
        foreach (var buffer in events.GroupBy(e => (e.Pool, e.Buffer)))
            Assert.Equal(buffer.Count(e => e.Rented), buffer.Count(e => !e.Rented));
    }

    sealed class Reporter(Exception? error) : IScratchAccountant
    {
        int calls;
        long bytes;
        public int Calls => Volatile.Read(ref calls);
        public long TotalScratchBytes => Interlocked.Read(ref bytes);
        public void AddScratchBytes(long count)
        {
            Interlocked.Increment(ref calls);
            Interlocked.Add(ref bytes, count);
            if (error is not null) throw error;
        }
    }

    readonly record struct Rental(bool Rented, int Pool, int Buffer, int Length);

    // ArrayPool events are process-wide. Capture only this synchronous call's
    // thread and disarm before assertions so other tests and diagnostics cannot
    // contribute their rentals. Match identities, including logging's buffers.
    sealed class Rentals : EventListener
    {
        readonly List<Rental> records = new();
        bool armed;
        int thread;
        protected override void OnEventSourceCreated(EventSource source)
        {
            if (source.Name == "System.Buffers.ArrayPoolEventSource") EnableEvents(source, EventLevel.Verbose);
        }
        protected override void OnEventWritten(EventWrittenEventArgs data)
        {
            if (!armed || Environment.CurrentManagedThreadId != thread || data.EventName is not ("BufferRented" or "BufferReturned")) return;
            records.Add(new Rental(data.EventName == "BufferRented", (int)data.Payload![2]!, (int)data.Payload[0]!, (int)data.Payload[1]!));
        }
        public (Exception? Error, Rental[] Events) Capture(Action action)
        {
            records.Clear();
            thread = Environment.CurrentManagedThreadId;
            Exception? error = null;
            armed = true;
            try { action(); }
            catch (Exception ex) { error = ex; }
            finally { armed = false; }
            return (error, records.ToArray());
        }
    }
}
