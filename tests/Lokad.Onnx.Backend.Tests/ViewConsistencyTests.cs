using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins that sliced and broadcast views, which share storage with exotic
/// strides, agree with dense computation: every kernel densifies defensively
/// today, and these tests catch any future fast path that misreads a view.
/// Expectations are hand-computed; the differential corpus cannot express
/// view inputs by its dense-only format.
/// </summary>
public class ViewConsistencyTests
{
    static DenseTensor<float> Base() =>
        DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });

    static Tensor<float> MiddleCols() =>
        Base().Slice(new SliceIndex(0, 2), new SliceIndex(1, 3));

    static Tensor<float> BroadcastRow() =>
        DenseTensor<float>.OfValues(new float[] { 5f, 6f }).PadLeft().BroadcastDim(0, 2);

    [Fact]
    public void Slice_ReduceMean_MatchesDense()
    {
        var axes = new int[] { 1 }.ToTensor<int>();
        var got = Tensor<float>.ReduceMean(MiddleCols(), axes, false, false);
        Assert.Equal(new float[] { 2.5f, 5.5f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.ReduceMean(dense, axes, false, false).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Softmax_MatchesDense()
    {
        var got = Tensor<float>.Softmax(MiddleCols(), -1, null, 13);
        var values = got.ToArray();
        Assert.Equal(0.26894f, values[0], 5);
        Assert.Equal(0.73106f, values[1], 5);
        Assert.Equal(0.26894f, values[2], 5);
        Assert.Equal(0.73106f, values[3], 5);
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Softmax(dense, -1, null, 13).ToArray(), values);
    }

    [Fact]
    public void Broadcast_MatMul_MatchesDense()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var expected = new float[] { 15f, 18f, 35f, 42f };
        var got = Tensor<float>.MatMul(a, BroadcastRow(), TensorExecutionOptions.Scalar);
        Assert.Equal(new int[] { 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(expected, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.MatMul(a, dense, TensorExecutionOptions.Scalar).ToArray(), got.ToArray());
    }

    [Fact]
    public void Broadcast_Add_MatchesDense()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 10f, 20f }, { 30f, 40f } });
        var expected = new float[] { 15f, 26f, 35f, 46f };
        var got = Tensor<float>.Add(BroadcastRow(), a);
        Assert.Equal(expected, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Add(dense, a).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Conv2D_MatchesDense()
    {
        // ORT 1.29: [18, 22, 34, 38].
        var parent = DenseTensor<float>.OfValues(new float[1, 1, 3, 4] { { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f } } } });
        var view = parent.Slice(new SliceIndex(0, 1), new SliceIndex(0, 1), new SliceIndex(0, 3), new SliceIndex(1, 4));
        Assert.IsType<TensorSlice<float>>(view);
        var w = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        var got = Tensor<float>.Conv2D(view, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(new float[] { 18f, 22f, 34f, 38f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[1, 1, 3, 3] { { { { 2f, 3f, 4f }, { 6f, 7f, 8f }, { 10f, 11f, 12f } } } });
        Assert.Equal(Tensor<float>.Conv2D(dense, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_LayerNorm_MatchesDense()
    {
        // ORT 1.29: rows of [-1.2247356, 0, 1.2247356].
        var parent = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        var view = parent.Slice(new SliceIndex(0, 2), new SliceIndex(0, 3));
        Assert.IsType<TensorSlice<float>>(view);
        var scale = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f });
        var bias = DenseTensor<float>.OfValues(new float[] { 0f, 0f, 0f });
        var got = Tensor<float>.LayerNormalization(view, scale, bias, -1, 1e-5f);
        var expected = new float[] { -1.2247357f, 0f, 1.2247357f, -1.2247357f, 0f, 1.2247357f };
        Assert.Equal(expected.Length, got.ToArray().Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], got.ToArray()[i], 6);
        var dense = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 5f, 6f, 7f } });
        Assert.Equal(Tensor<float>.LayerNormalization(dense, scale, bias, -1, 1e-5f).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Gather_MatchesDense()
    {
        // Gathering rows out of a strided view exercises the generic
        // index path (the span fast path requires standard strides).
        var idx = DenseTensor<int>.OfValues(new int[] { 1, 0 });
        var got = Tensor<float>.Gather(MiddleCols(), idx, 0);
        Assert.Equal(new int[] { 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f, 6f, 2f, 3f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Gather(dense, idx, 0).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Concat_MatchesDense()
    {
        // Concatenating two views stacked on the same storage walks the
        // generic iterator (the chunk copier needs dense inputs).
        var view = MiddleCols();
        var got = Tensor<float>.Concat(view, view, 0);
        Assert.Equal(new int[] { 4, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 2f, 3f, 5f, 6f, 2f, 3f, 5f, 6f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Concat(dense, dense, 0).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Transpose_MatchesDense()
    {
        // Permuting a strided view must read through the view strides,
        // not the parent storage order.
        var got = Tensor<float>.Transpose(MiddleCols(), new int[] { 1, 0 });
        Assert.Equal(new int[] { 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 2f, 5f, 3f, 6f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Transpose(dense, new int[] { 1, 0 }).ToArray(), got.ToArray());
    }

    [Fact]
    public void Transpose_MatMul_MatchesDense()
    {
        // A transposed view fed to MatMul must read through the view
        // strides on every kernel path (odd 3x2 exercises tails).
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } });
        var t = Tensor<float>.Transpose(a, new int[] { 1, 0 });
        var b = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f }, { 1f, 1f } });
        var got = Tensor<float>.MatMul(t, b, TensorExecutionOptions.Scalar);
        Assert.Equal(new int[] { 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 6f, 8f, 8f, 10f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 1f, 3f, 5f }, { 2f, 4f, 6f } });
        Assert.Equal(Tensor<float>.MatMul(dense, b, TensorExecutionOptions.Scalar).ToArray(), got.ToArray());
    }

    [Fact]
    public void Transpose_Tile_MatchesDense()
    {
        // Tiling a transposed view replicates logical values across the
        // permuted strides (verified differentially tri-mode via OpDump).
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var t = Tensor<float>.Transpose(a, new int[] { 1, 0 });
        var got = Tensor<float>.Tile(t, new int[] { 2, 1 });
        Assert.Equal(new int[] { 6, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 4f, 2f, 5f, 3f, 6f, 1f, 4f, 2f, 5f, 3f, 6f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 1f, 4f }, { 2f, 5f }, { 3f, 6f } });
        Assert.Equal(Tensor<float>.Tile(dense, new int[] { 2, 1 }).ToArray(), got.ToArray());
    }

    [Fact]
    public void Transpose_Concat_MatchesDense()
    {
        // Concatenating a transposed view with a dense tensor walks the
        // generic iterator over permuted strides.
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } });
        var t = Tensor<float>.Transpose(a, new int[] { 1, 0 });
        var b = DenseTensor<float>.OfValues(new float[,] { { 7f }, { 8f } });
        var got = Tensor<float>.Concat(t, b, 1);
        Assert.Equal(new int[] { 2, 4 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 3f, 5f, 7f, 2f, 4f, 6f, 8f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 1f, 3f, 5f }, { 2f, 4f, 6f } });
        Assert.Equal(Tensor<float>.Concat(dense, b, 1).ToArray(), got.ToArray());
    }

    [Fact]
    public void NestedSlice_MatchesDense()
    {
        // A slice of a slice chains two indirections; values must match
        // the equivalent dense window.
        var nested = MiddleCols().Slice(new SliceIndex(1, 2), new SliceIndex(0, 2));
        Assert.Equal(new int[] { 1, 2 }, nested.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f, 6f }, nested.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f } });
        Assert.Equal(dense.ToArray(), nested.ToArray());
    }

    [Fact]
    public void Slice_ReduceMax_MatchesDense()
    {
        // Row-wise maxima over a strided view must read through the
        // view, not the parent rows.
        var axes = new int[] { 1 }.ToTensor<int>();
        var got = Tensor<float>.ReduceMax(MiddleCols(), axes, false, false);
        Assert.Equal(new float[] { 3f, 6f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.ReduceMax(dense, axes, false, false).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_ReduceSum_MatchesDense()
    {
        // Column-wise sums over a strided view skip the sliced-off column.
        var axes = new int[] { 0 }.ToTensor<int>();
        var got = Tensor<float>.ReduceSum(MiddleCols(), axes, false, false);
        Assert.Equal(new float[] { 7f, 9f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.ReduceSum(dense, axes, false, false).ToArray(), got.ToArray());
    }

    [Fact]
    public void Broadcast_Tile_MatchesDense()
    {
        // Tiling a broadcast view replicates the logical values, not the
        // single stored row.
        var got = Tensor<float>.Tile(BroadcastRow(), new int[] { 1, 2 });
        Assert.Equal(new int[] { 2, 4 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f, 6f, 5f, 6f, 5f, 6f, 5f, 6f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Tile(dense, new int[] { 1, 2 }).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Expand_MatchesDense()
    {
        // Expanding a strided view with a new leading axis replicates the
        // logical window on both planes.
        var got = Tensor<float>.Expand(MiddleCols(), new int[] { 2, 2, 2 });
        Assert.Equal(new int[] { 2, 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 2f, 3f, 5f, 6f, 2f, 3f, 5f, 6f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Expand(dense, new int[] { 2, 2, 2 }).ToArray(), got.ToArray());
    }

    [Fact]
    public void ReversedSlice_ReduceMax_MatchesDense()
    {
        // A negative-step slice walks storage backwards; row-wise maxima
        // must follow the logical order [[4,5,6],[1,2,3]].
        var rev = Base().Slice(new SliceIndex(null, null, -1), new SliceIndex(0, 3));
        Assert.IsType<TensorSlice<float>>(rev);
        var axes = new int[] { 1 }.ToTensor<int>();
        var got = Tensor<float>.ReduceMax(rev, axes, false, false);
        Assert.Equal(new float[] { 6f, 3f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 4f, 5f, 6f }, { 1f, 2f, 3f } });
        Assert.Equal(Tensor<float>.ReduceMax(dense, axes, false, false).ToArray(), got.ToArray());
    }

    [Fact]
    public void ReversedSlice_Transpose_MatchesDense()
    {
        // Permuting a backwards-walking view must transpose logical
        // values, not storage order.
        var rev = Base().Slice(new SliceIndex(null, null, -1), new SliceIndex(0, 3));
        var got = Tensor<float>.Transpose(rev, new int[] { 1, 0 });
        Assert.Equal(new int[] { 3, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 4f, 1f, 5f, 2f, 6f, 3f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 4f, 5f, 6f }, { 1f, 2f, 3f } });
        Assert.Equal(Tensor<float>.Transpose(dense, new int[] { 1, 0 }).ToArray(), got.ToArray());
    }

    [Fact]
    public void ReversedSlice_Where_MatchesDense()
    {
        // Selection from a backwards-walking payload follows logical rows.
        var rev = Base().Slice(new SliceIndex(null, null, -1), new SliceIndex(0, 3));
        var cond = DenseTensor<bool>.OfValues(new bool[,] { { true }, { false } });
        var zeros = DenseTensor<float>.OfValues(new float[,] { { 0f, 0f, 0f }, { 0f, 0f, 0f } });
        var got = Tensor<float>.Where(cond, rev, zeros);
        Assert.Equal(new float[] { 4f, 5f, 6f, 0f, 0f, 0f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 4f, 5f, 6f }, { 1f, 2f, 3f } });
        Assert.Equal(Tensor<float>.Where(cond, dense, zeros).ToArray(), got.ToArray());
    }
    [Fact]
    public void ReversedSlice_MatMul_MatchesDense()
    {
        // A backwards-walking MatMul operand multiplies logical rows:
        // [[4,5,6],[1,2,3]] times [[1,0],[0,1],[1,1]] is [[10,11],[4,5]].
        var rev = Base().Slice(new SliceIndex(null, null, -1), new SliceIndex(0, 3));
        Assert.IsType<TensorSlice<float>>(rev);
        var b = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f }, { 1f, 1f } });
        var got = Tensor<float>.MatMul(rev, b, TensorExecutionOptions.Scalar);
        Assert.Equal(new int[] { 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 10f, 11f, 4f, 5f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 4f, 5f, 6f }, { 1f, 2f, 3f } });
        Assert.Equal(Tensor<float>.MatMul(dense, b, TensorExecutionOptions.Scalar).ToArray(), got.ToArray());
    }

    [Fact]
    public void ReversedSlice_ReduceSum_MatchesDense()
    {
        // Row-wise sums follow the logical order [[4,5,6],[1,2,3]]: 15 and 6.
        var rev = Base().Slice(new SliceIndex(null, null, -1), new SliceIndex(0, 3));
        Assert.IsType<TensorSlice<float>>(rev);
        var axes = new int[] { 1 }.ToTensor<int>();
        var got = Tensor<float>.ReduceSum(rev, axes, false, false);
        Assert.Equal(new float[] { 15f, 6f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 4f, 5f, 6f }, { 1f, 2f, 3f } });
        Assert.Equal(Tensor<float>.ReduceSum(dense, axes, false, false).ToArray(), got.ToArray());
    }
    [Fact]
    public void BroadcastAddView_MatchesDense()
    {
        // A broadcast row view plus a sliced view add exactly like dense.
        var v = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } })
            .Slice(new SliceIndex(0, 2), new SliceIndex(1, 3));
        var brow = DenseTensor<float>.OfValues(new float[] { 10f, 20f }).PadLeft().BroadcastDim(0, 2);
        var got = Tensor<float>.Add(v, brow);
        Assert.Equal(new float[] { 12f, 23f, 15f, 26f }, got.ToArray());
    }

    [Fact]
    public void SliceOfSlice_MatchesDense()
    {
        // Nested views compose: slicing a slice reads the right elements.
        var outer = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f } })
            .Slice(new SliceIndex(0, 3), new SliceIndex(1, 4));
        var inner = outer.Slice(new SliceIndex(1, 3), new SliceIndex(0, 2));
        var dense = DenseTensor<float>.OfValues(new float[,] { { 6f, 7f }, { 10f, 11f } });
        Assert.Equal(dense.ToArray(), ((Tensor<float>)inner).ToArray());
    }

    [Fact]
    public void GatherViewData_MatchesDense()
    {
        // Gathering from a strided view reads logical values, not storage order.
        var v = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } })
            .Slice(new SliceIndex(0, 2), new SliceIndex(1, 3));
        var r = CPU.Gather(v, DenseTensor<long>.OfValues(new long[] { 1L, 0L }), 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 5f, 6f, 2f, 3f }, ((Tensor<float>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void WhereViewCond_MatchesDense()
    {
        // A sliced bool condition selects exactly like its dense twin.
        var c = DenseTensor<bool>.OfValues(new bool[,] { { true, false, true }, { false, true, false } })
            .Slice(new SliceIndex(0, 2), new SliceIndex(0, 2));
        var x = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        var z = DenseTensor<float>.OfValues(new float[,] { { 0f, 0f }, { 0f, 0f } });
        var r = CPU.Where(c, x, z, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 2f, 0f, 0f, 6f }, ((Tensor<float>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void ConvViewData_MatchesDense()
    {
        // Conv reads strided view inputs logically (1x2 kernel over the
        // middle columns).
        var v = DenseTensor<float>.OfValues(new float[1, 1, 2, 4] { { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } } } })
            .Slice(new SliceIndex(0, 1), new SliceIndex(0, 1), new SliceIndex(0, 2), new SliceIndex(1, 4));
        var w = DenseTensor<float>.OfValues(new float[1, 1, 1, 2] { { { { 1f, 1f } } } });
        var y = Tensor<float>.Conv2D(v, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(new int[] { 1, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f, 7f, 13f, 15f }, y.ToArray());
    }

    [Fact]
    public void ResizeNearestView_MatchesDense()
    {
        // Nearest resampling of a strided view doubles each logical element.
        var v = DenseTensor<float>.OfValues(new float[1, 1, 2, 4] { { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } } } })
            .Slice(new SliceIndex(0, 1), new SliceIndex(0, 1), new SliceIndex(0, 2), new SliceIndex(1, 3));
        var sc = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 2f, 2f });
        var r = CPU.Resize(v, null, sc, null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 2f, 2f, 3f, 3f, 2f, 2f, 3f, 3f, 6f, 6f, 7f, 7f, 6f, 6f, 7f, 7f },
            ((Tensor<float>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void ReduceSumView_MatchesDense()
    {
        // Row sums over a strided view equal dense row sums.
        var v = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } })
            .Slice(new SliceIndex(0, 2), new SliceIndex(1, 3));
        var r = CPU.ReduceSum(v, DenseTensor<int>.OfValues(new int[] { 1 }), 0, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 5f, 11f }, ((Tensor<float>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void ExpandViewData_MatchesDense()
    {
        // Expanding a strided single-column view broadcasts logical values.
        var v = DenseTensor<float>.OfValues(new float[,] { { 1f, 9f }, { 2f, 9f } })
            .Slice(new SliceIndex(0, 2), new SliceIndex(0, 1));
        var r = CPU.Expand(v, DenseTensor<long>.OfValues(new long[] { 2L, 3L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 1f, 1f, 1f, 2f, 2f, 2f }, ((Tensor<float>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void TileViewData_MatchesDense()
    {
        // Tiling a strided view replicates logical values.
        var v = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 9f }, { 3f, 4f, 9f } })
            .Slice(new SliceIndex(0, 2), new SliceIndex(0, 2));
        var r = CPU.Tile(v, DenseTensor<long>.OfValues(new long[] { 1L, 2L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f }, ((Tensor<float>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void SplitViewData_MatchesDense()
    {
        // Splitting a strided view partitions logical values.
        var v = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } })
            .Slice(new SliceIndex(0, 2), new SliceIndex(1, 3));
        var r = CPU.Split(v, DenseTensor<long>.OfValues(new long[] { 1L, 1L }), 1, null, null, null, 2);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 2f, 5f }, ((Tensor<float>)r.Outputs![0]).ToArray());
        Assert.Equal(new float[] { 3f, 6f }, ((Tensor<float>)r.Outputs![1]).ToArray());
    }

    [Fact]
    public void TransposeView_MatchesDense()
    {
        // Transposing a strided view permutes logical values.
        var v = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } })
            .Slice(new SliceIndex(0, 2), new SliceIndex(1, 3));
        Assert.Equal(new float[] { 2f, 5f, 3f, 6f }, Tensor<float>.Transpose(v, new int[] { 1, 0 }).ToArray());
    }

    [Fact]
    public void ErfView_StaysInApproximationClass()
    {
        // Strided inputs take the scalar Erf path while contiguous inputs
        // vectorize, so views may differ by the documented scalar-versus-
        // vector approximation (bounded 4.8e-07), never more.
        var v = DenseTensor<float>.OfValues(new float[,] { { 0.5f, 1f, 1.5f }, { 2f, 2.5f, 3f } })
            .Slice(new SliceIndex(0, 2), new SliceIndex(0, 3));
        var dense = DenseTensor<float>.OfValues(new float[,] { { 0.5f, 1f, 1.5f }, { 2f, 2.5f, 3f } });
        var a = Tensor<float>.Erf(v).ToArray();
        var b = Tensor<float>.Erf(dense).ToArray();
        double worst = 0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs((double)a[i] - b[i]));
        Assert.True(worst <= 4.8e-07, $"erf view drift {worst:E3} exceeds the documented class");
    }

}
