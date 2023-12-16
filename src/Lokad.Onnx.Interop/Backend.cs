namespace Lokad.Onnx.Interop;

using System;
using Python.Runtime;

[PyExport(true)]
public enum DeviceType { CPU = 0, CUDA = 1 }

[PyExport(true)]
public class Device
{
    public Device(string device)
    {
        var options = device.Split(':');
        type = (DeviceType) Enum.Parse(typeof(DeviceType), options[0]);  
        if (options.Length > 1) 
        {
            device_id = int.Parse(options[1]);
        }
    }
    public DeviceType type;
    public int device_id = 0;
}


[PyExport(true)]    
public class Backend
{
    

}
