namespace Lokad.Onnx.Tensors.Tests;

using System;
using System.Buffers;
using System.Runtime.InteropServices;
public class UnsafeFixedSizeListTests
{
    [Fact]
    public unsafe void CanCreate()
    {
        const int Seed = 20260904; var rnd = new Random(Seed);
        var l0 = rnd.Next();
        var l1 = rnd.Next();
        var l4 = rnd.Next();
        var l9 = rnd.Next();
        int* x = stackalloc int[10];
        UnsafeFixedSizeList<int> l = new UnsafeFixedSizeList<int>(x, 10, 0);
        l.Add(l0);
        Assert.Equal(l0, l[0]);
        l.Insert(0, l1);
        Assert.Equal(l1, l[0]);
        Assert.Equal(l0, l[1]);
        l.Insert(1, l4);
        Assert.Equal(l4, l[1]);
        Assert.Equal(l0, l[2]);
        l.Add(l9);
        Assert.Equal(l9, l[3]);
        l.RemoveAt(2);
        Assert.Equal(l9, l[2]);
    }

    [Fact]
    public unsafe void CanStackAllocCreate()
    {
        const int Seed = 20260904; var rnd = new Random(Seed);
        var l0 = rnd.Next();
        var l1 = rnd.Next();
        var l4 = rnd.Next();
        var l9 = rnd.Next();
        int* x = stackalloc int[10];
        UnsafeFixedSizeList<int> l = new UnsafeFixedSizeList<int>(x, 10, 0);
        l.Add(l0);
        Assert.Equal(l0, l[0]);
        l.Insert(0, l1);
        Assert.Equal(l1, l[0]);
        Assert.Equal(l0, l[1]);
        l.Insert(1, l4);
        Assert.Equal(l4, l[1]);
        Assert.Equal(l0, l[2]);
        l.Add(l9);
        Assert.Equal(l9, l[3]);
        l.RemoveAt(2);
        Assert.Equal(l9, l[2]);
    }


    // Overflow tests deliberately advertise a smaller size than the real
    // buffer: exceeding the advertised capacity must throw while every
    // executed write stays inside allocated memory on both old and new code.
    [Fact]
    public unsafe void AddPastCapacity_Throws()
    {
        int* x = stackalloc int[4];
        UnsafeFixedSizeList<int> l = new UnsafeFixedSizeList<int>(x, 2, 0);
        l.Add(1);
        l.Add(2);
        Assert.Throws<ArgumentOutOfRangeException>(() => l.Add(3));
        Assert.Equal(2, l.Count);
    }

    [Fact]
    public unsafe void InsertPastCapacity_Throws()
    {
        int* x = stackalloc int[4];
        UnsafeFixedSizeList<int> l = new UnsafeFixedSizeList<int>(x, 1, 0);
        l.Add(1);
        Assert.Throws<ArgumentOutOfRangeException>(() => l.Insert(0, 2));
        Assert.Equal(1, l.Count);
    }

    [Fact]
    public unsafe void AddRangePastCapacity_Throws()
    {
        int* x = stackalloc int[4];
        UnsafeFixedSizeList<int> l = new UnsafeFixedSizeList<int>(x, 2, 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => l.AddRange(new int[] { 1, 2, 3 }));
        Assert.Equal(0, l.Count);
    }

    [Fact]
    public unsafe void IndexerOutOfRange_Throws()
    {
        int* x = stackalloc int[4];
        UnsafeFixedSizeList<int> l = new UnsafeFixedSizeList<int>(x, 2, 0);
        l.Add(7);
        Assert.Throws<ArgumentOutOfRangeException>(() => l[1]);
        Assert.Throws<ArgumentOutOfRangeException>(() => l[-1]);
        Assert.Throws<ArgumentOutOfRangeException>(() => l[1] = 1);
        l[0] = 9;
        Assert.Equal(9, l[0]);
    }
}
