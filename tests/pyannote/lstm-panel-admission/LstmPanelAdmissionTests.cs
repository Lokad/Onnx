using System;
using System.Reflection;

namespace Lokad.Onnx.Backend.Tests;

public class LstmPanelAdmissionTests
{
    static readonly Func<int, int, int> StorageLength = typeof(CPUExecutionProvider)
        .GetNestedType("LstmProjectionPanels", BindingFlags.NonPublic)!
        .GetMethod("StorageLength", BindingFlags.Static | BindingFlags.NonPublic)!
        .CreateDelegate<Func<int, int, int>>();

    [Theory]
    [InlineData(0, 0, 0)]
    [InlineData(0, 1024, 1024)]
    [InlineData(61440, 131072, 192512)]
    [InlineData(262144, 131072, 393216)]
    [InlineData(int.MaxValue, 1, 0)]
    [InlineData(int.MaxValue, int.MaxValue, 0)]
    [InlineData(-1, 32, 0)]
    [InlineData(32, -1, 0)]
    public void ActualPyannoteSizesAndOverflowRefusal(int input, int recurrent, int expected)
    {
        Assert.Equal(expected, StorageLength(input, recurrent));
    }

    [Fact]
    public void OptionalCombinedArrayHonorsItsOwnRepresentationLimit()
    {
        const int recurrent = 131072;
        int inputAtLimit = Array.MaxLength - recurrent;
        Assert.Equal(Array.MaxLength, StorageLength(inputAtLimit, recurrent));
        Assert.Equal(0, StorageLength(inputAtLimit + 1, recurrent));
        Assert.Equal(0, StorageLength(Array.MaxLength, recurrent));
    }
}
