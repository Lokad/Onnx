using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public enum OP_TYPE
    {
        MatAdd
    }
    
    public struct Node
    {
        public long ID;
        public string Name;
        public string Description;
        public Satsuma.Node WeightedGraphNode;
        public OP_TYPE Op;
        public string[] Inputs;
        public string[] Outputs;
    }
}
