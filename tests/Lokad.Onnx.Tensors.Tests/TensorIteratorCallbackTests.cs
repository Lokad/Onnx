using System.Collections;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorIteratorCallbackTests
{
    [Theory]
    [InlineData(new int[] { 2, 2 }, 4)]
    [InlineData(new int[] { }, 1)]
    [InlineData(new int[] { 0, 3 }, 0)]
    public void Completion_NotifiesOncePerTraversal(int[] dimensions, int expectedCount)
    {
        int calls = 0;
        var cursor = new TensorDimensionsIterator(dimensions, (ref TensorDimensionsIterator ended) =>
        {
            calls++;
            Assert.Null(ended.Next());
        });
        int visited = 0;
        while (cursor.MoveNext()) visited++;
        Assert.Equal(expectedCount, visited);
        Assert.Equal(1, calls);
        Assert.False(cursor.MoveNext());
        Assert.Null(cursor.Next());
        Assert.Equal(1, calls);
        cursor.Reset();
        while (cursor.MoveNext()) { }
        Assert.Equal(2, calls);
    }

    [Fact]
    public void ManualNext_NotifiesAtItsExistingEndAndResetReenablesIt()
    {
        int calls = 0;
        var cursor = new TensorDimensionsIterator(new[] { 2 }, (ref TensorDimensionsIterator _) => calls++);
        Assert.Equal(new[] { 1 }, cursor.Next());
        Assert.Equal(0, calls);
        Assert.Null(cursor.Next());
        Assert.Null(cursor.Next());
        Assert.Equal(1, calls);
        cursor.Reset();
        Assert.NotNull(cursor.Next());
        Assert.Null(cursor.Next());
        Assert.Equal(2, calls);
    }

    [Fact]
    public void FreshForeachAndAppend_RetainTheCallback()
    {
        int calls = 0;
        var configuration = new TensorDimensionsIterator(new[] { 2 }, (ref TensorDimensionsIterator _) => calls++);
        foreach (int expectedCount in new[] { 2, 2 })
        {
            int count = 0;
            foreach (var _ in configuration) count++;
            Assert.Equal(expectedCount, count);
        }
        Assert.Equal(2, calls);
        int appendedCount = 0;
        foreach (var _ in configuration.Append(2)) appendedCount++;
        Assert.Equal(4, appendedCount);
        Assert.Equal(3, calls);
    }

    [Fact]
    public void InterfaceEnumeration_RetainsTheCallback()
    {
        int calls = 0;
        var configuration = new TensorDimensionsIterator(new[] { 2 }, (ref TensorDimensionsIterator _) => calls++);
        using (var cursor = ((IEnumerable<int[]>)configuration).GetEnumerator())
            while (cursor.MoveNext()) { }
        Assert.Equal(1, calls);
        foreach (var _ in (IEnumerable)configuration) { }
        Assert.Equal(2, calls);
    }

    [Fact]
    public void EarlyResetOrDisposal_DoesNotNotify()
    {
        int calls = 0;
        var cursor = new TensorDimensionsIterator(new[] { 2 }, (ref TensorDimensionsIterator _) => calls++);
        Assert.True(cursor.MoveNext());
        cursor.Reset();
        Assert.Equal(0, calls);
        using (var enumerator = ((IEnumerable<int[]>)cursor).GetEnumerator()) Assert.True(enumerator.MoveNext());
        Assert.Equal(0, calls);
    }

    [Fact]
    public void DefaultCursor_RemainsInert()
    {
        var cursor = default(TensorDimensionsIterator);
        Assert.False(cursor.MoveNext());
        Assert.Null(cursor.Next());
        cursor.Reset();
        ((IDisposable)cursor).Dispose();
        Assert.False(cursor.GetEnumerator().MoveNext());
    }

    [Fact]
    public void CallbackCanResetCursor_WhileTheAdvancingCallStillEnds()
    {
        int calls = 0;
        var cursor = new TensorDimensionsIterator(new[] { 1 }, (ref TensorDimensionsIterator ended) =>
        {
            if (++calls == 1) ended.Reset();
        });
        Assert.True(cursor.MoveNext());
        Assert.False(cursor.MoveNext());
        Assert.Equal(1, calls);
        Assert.True(cursor.MoveNext());
        Assert.False(cursor.MoveNext());
        Assert.False(cursor.MoveNext());
        Assert.Equal(2, calls);
    }

    [Fact]
    public void ThrowingCallback_LeavesTheCursorExhausted()
    {
        int calls = 0;
        var cursor = new TensorDimensionsIterator(new[] { 1 }, (ref TensorDimensionsIterator _) =>
        {
            calls++;
            throw new InvalidOperationException("callback failed");
        });
        Assert.True(cursor.MoveNext());
        Assert.Throws<InvalidOperationException>(() => cursor.MoveNext());
        Assert.False(cursor.MoveNext());
        Assert.Equal(1, calls);
    }
}
