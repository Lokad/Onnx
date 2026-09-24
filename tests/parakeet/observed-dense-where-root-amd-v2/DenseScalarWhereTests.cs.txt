using System.Runtime.InteropServices;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

// Public selection and ownership contracts for the measured mask geometries.
public class DenseScalarWhereTests
{
    public static IEnumerable<object[]> AttentionFrames => new[]
        { 51,61,83,88,89,102,106,112,114,120,151,156,157,158,167,169,190,222,225 }
        .Select(t => new object[] { t });

    [Theory]
    [MemberData(nameof(AttentionFrames))]
    public void AttentionMaskCopiesEveryHeadWithIndependentStorage(int t)
    {
        int length = checked(8*t*t), offset = 7;
        var backing = Values(length+17);
        var before = Bits(backing);
        var y = new DenseTensor<float>(backing.AsMemory(offset,length),new[] {1,8,t,t});
        var condition = DenseTensor<bool>.OfShape(1,1,t,t);
        var scalar = Scalar(BitConverter.Int32BitsToSingle(0x7fc0abcd));
        var result = Select(condition,scalar,y);
        var expected = before.Skip(offset).Take(length).ToArray();
        Assert.Equal(new[] {1,8,t,t},result.Dimensions.ToArray());
        Assert.Equal(expected,Bits(result.ToArray()));
        Assert.Equal(before,Bits(backing));
        backing[offset] = 17f;
        scalar.SetValue(0,19f);
        condition.Buffer.Span.Fill(true);
        var later = Select(condition,scalar,y);
        Assert.All(later.ToArray(),v => Assert.Equal(19f,v));
        Assert.Equal(expected,Bits(result.ToArray()));
        result.SetValue(0,23f);
        Assert.Equal(17f,backing[offset]);
        Assert.Equal(before.Take(offset),Bits(backing).Take(offset));
        Assert.Equal(before.Skip(offset+length),Bits(backing).Skip(offset+length));
    }

    [Fact]
    public void MixedAttentionMaskPreservesSelectedPayloadsAndMemoryWindows()
    {
        const int rows=32,columns=33,heads=8,batches=2;
        var maskBacking = new bool[rows*columns+13];
        var condition = new DenseTensor<bool>(maskBacking.AsMemory(5,rows*columns),new[] {1,1,rows,columns});
        var bytes = MemoryMarshal.AsBytes(condition.Buffer.Span);
        for (int i=0;i<bytes.Length;i++) bytes[i]=(byte)(i%3==0 ? 0 : i%3==1 ? 2 : 255);
        var maskBefore = MemoryMarshal.AsBytes(maskBacking.AsSpan()).ToArray();
        var backing = Values(batches*heads*rows*columns+19);
        var before = Bits(backing);
        var y = new DenseTensor<float>(backing.AsMemory(9,batches*heads*rows*columns),new[] {batches,heads,rows,columns});
        var scalarBacking = new[] {13f,BitConverter.Int32BitsToSingle(0x7fc01234),17f};
        var scalar = new DenseTensor<float>(scalarBacking.AsMemory(1,1),Array.Empty<int>());
        var result = Select(condition,scalar,y);
        var expected = new List<int>();
        for (int batch=0;batch<batches;batch++)
        for (int head=0;head<heads;head++)
        for (int row=0;row<rows;row++)
        for (int column=0;column<columns;column++)
            expected.Add((row*columns+column)%3==0
                ? before[9+((batch*heads+head)*rows+row)*columns+column] : 0x7fc01234);
        Assert.Equal(expected,Bits(result.ToArray()));
        Assert.Equal(before,Bits(backing));
        Assert.Equal(maskBefore,MemoryMarshal.AsBytes(maskBacking.AsSpan()).ToArray());
        Assert.Equal(new[] {13f,BitConverter.Int32BitsToSingle(0x7fc01234),17f}.Select(BitConverter.SingleToInt32Bits),Bits(scalarBacking));
    }

    [Fact]
    public void MixedOuterMaskBroadcastsAcrossChannelsAndContiguousRows()
    {
        var condition = DenseTensor<bool>.OfShape(2,1,4,1);
        var bytes = MemoryMarshal.AsBytes(condition.Buffer.Span);
        for (int i=0;i<bytes.Length;i++) bytes[i]=(byte)(i%3==0 ? 0 : 255);
        var y = new DenseTensor<float>(Values(2*3*4*1024),new[] {2,3,4,1024});
        var before = Bits(y.ToArray());
        var result = Select(condition,Scalar(-10000f),y);
        var expected = new List<int>();
        for (int batch=0;batch<2;batch++)
        for (int channel=0;channel<3;channel++)
        for (int row=0;row<4;row++)
        for (int column=0;column<1024;column++)
            expected.Add((batch*4+row)%3==0
                ? before[((batch*3+channel)*4+row)*1024+column]
                : BitConverter.SingleToInt32Bits(-10000f));
        Assert.Equal(expected,Bits(result.ToArray()));
        Assert.Equal(before,Bits(y.ToArray()));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(int.MinValue)]
    [InlineData(0x7fc01234)]
    [InlineData(0x7f800000)]
    public void UniformNoncanonicalTrueMaskPreservesScalarBits(int bits)
    {
        var condition = DenseTensor<bool>.OfShape(1,1,8);
        var bytes = MemoryMarshal.AsBytes(condition.Buffer.Span);
        for (int i=0;i<bytes.Length;i++) bytes[i]=(byte)(i%2==0 ? 2 : 255);
        var y = new DenseTensor<float>(Values(8*1024),new[] {1,1024,8});
        var before = Bits(y.ToArray());
        var scalar = Scalar(BitConverter.Int32BitsToSingle(bits));
        var result = Select(condition,scalar,y);
        Assert.All(Bits(result.ToArray()),actual => Assert.Equal(bits,actual));
        Assert.Equal(before,Bits(y.ToArray()));
        scalar.SetValue(0,31f);
        y.SetValue(0,37f);
        Assert.All(Bits(result.ToArray()),actual => Assert.Equal(bits,actual));
    }

    [Fact]
    public void HigherRankConditionStillExpandsTheOutput()
    {
        var condition = new DenseTensor<bool>(new[] {true,false},new[] {1,2,1});
        var y = new DenseTensor<float>(Values(4096),new[] {4096});
        var result = Select(condition,Scalar(-0f),y);
        Assert.Equal(new[] {1,2,4096},result.Dimensions.ToArray());
        Assert.Equal(Enumerable.Repeat(int.MinValue,4096).Concat(Bits(y.ToArray())),Bits(result.ToArray()));
    }

    [Fact]
    public void UniformFalseMaskDoesNotBypassIncompatibleShapes()
    {
        var y = new DenseTensor<float>(Values(8*32*32),new[] {1,8,32,32});
        var before = Bits(y.ToArray());
        Assert.Throws<ArgumentException>(() => CPU.Where(DenseTensor<bool>.OfShape(1,7,1,1),Scalar(5f),y,null));
        Assert.Equal(before,Bits(y.ToArray()));
    }

    [Fact]
    public void DerivedTensorValueSemanticsRemainObservable()
    {
        var y = new AdjustedDense();
        var result = Select(DenseTensor<bool>.OfShape(),Scalar(-1f),y);
        Assert.Equal(Enumerable.Range(0,4096).Select(i => (float)(i+1000)),result.ToArray());
    }

    static Tensor<float> Select(Tensor<bool> condition,Tensor<float> x,Tensor<float> y)
    {
        var result=CPU.Where(condition,x,y,null);
        Assert.Equal(OpStatus.Success,result.Status);
        return (Tensor<float>)result.Outputs![0];
    }

    static DenseTensor<float> Scalar(float value)
    {
        var result=DenseTensor<float>.OfShape();result.SetValue(0,value);return result;
    }

    static float[] Values(int count)
    {
        int[] bits={0,int.MinValue,0x7f800000,unchecked((int)0xff800000),0x7fc01234,unchecked((int)0xffc05678),0x3f800000};
        return Enumerable.Range(0,count).Select(i => BitConverter.Int32BitsToSingle(bits[i%bits.Length])).ToArray();
    }

    static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

    sealed class AdjustedDense : DenseTensor<float>
    {
        public AdjustedDense() : base((ReadOnlySpan<int>)new[] {4096}) { }
        public override float GetValue(int index) => index+1000f;
    }
}
