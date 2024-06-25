namespace Lokad.Onnx.Backend.Tests;

using static Lokad.Onnx.Text;
public class DataTests
{
    [Fact]
    public void CanPad()
    {
        string s1 = "Hello world";
        var t1 = GetTextTensors(s1, "me5s");
        string[] t = { "Hello world", "This is a test", "of padding" };
        var tensors = GetTextTensors(t, "me5s");
        Assert.NotNull(tensors);    
    }
}
