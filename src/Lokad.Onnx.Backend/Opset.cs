using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx.Backend
{
    public struct Opset
    {
        public string Domain = "ai";
        public int Version;

        public Opset(string domain, int version)
        {
            Domain = domain;
            Version = version;
        }

        public Opset(int version)
        { 
            Version = version; 
        }
    }
}
