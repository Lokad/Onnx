namespace Lokad.Onnx.Backend.Tests;

public class CollectionJoinTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Joins_EnumerateOnce(bool spaces)
    {
        int enumerations = 0;
        IEnumerable<string> Words()
        {
            Assert.Equal(1, ++enumerations);
            yield return "one";
            yield return "two";
        }
        Assert.Equal(spaces ? "one two" : "one,two",
            spaces ? Words().JoinWithSpaces() : Words().JoinWith(","));
    }

    [Fact]
    public void Joins_PreserveEmptyAndNullElements()
    {
        Assert.Equal("", Array.Empty<string>().JoinWith(","));
        Assert.Equal("", new string[] { null! }.JoinWith(","));
        Assert.Equal(",value,", new[] { null!, "value", "" }.JoinWith(","));
        Assert.Equal("value", new[] { "val", "ue" }.JoinWith(null!));
    }
}
