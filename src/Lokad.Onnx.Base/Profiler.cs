using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public enum OpStage
    {
        MATH,
        COPY,
        CALCULATE_INDICES
    }
    
    public record OpProfile { OpType Op; OpStage Stage; public TimeSpan Time;  }

    public class OpsProfiler
    {
        public static Dictionary<int, List<OpProfile>> Profile = new Dictionary<int, List<OpProfile>>();

        public static bool Running => timer.IsRunning;

        public static void StartOpProfile(int id, OpType op, OpStage stage)
        {
            if (Running)
            {
                Profile.Last().Value.Last().Time = timer.Elapsed;
                timer.Reset();
                Profile.Add(id, new List<OpProfile>());
                
            }
        }

        private static Stopwatch timer = new Stopwatch();
    }
}
