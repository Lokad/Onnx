namespace Lokad.Onnx
{
    public class RuntimeNotInitializedException : Exception
    {
        public RuntimeNotInitializedException() : base($"This runtime object is not initialized.") { }
    }
}
