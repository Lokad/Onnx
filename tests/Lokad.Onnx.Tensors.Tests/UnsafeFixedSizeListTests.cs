namespace Lokad.Onnx.Tensors.Tests;

using System;
using System.Buffers;
using System.Runtime.InteropServices;
public class UnsafeFixedSizeListTests
{
    [Fact]
    public unsafe void CanCreate()
    {
        var rnd = new Random();
        var l0 = rnd.Next();
        var l1 = rnd.Next();
        var l4 = rnd.Next();
        var l9 = rnd.Next();
        int* x = stackalloc int[10];
        UnsafeFixedSizeList<int> l = new UnsafeFixedSizeList<int>(x, 10);
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
        var rnd = new Random();
        var l0 = rnd.Next();
        var l1 = rnd.Next();
        var l4 = rnd.Next();
        var l9 = rnd.Next();
        int* x = stackalloc int[10];
        UnsafeFixedSizeList<int> l = new UnsafeFixedSizeList<int>(x, 10);
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
}

