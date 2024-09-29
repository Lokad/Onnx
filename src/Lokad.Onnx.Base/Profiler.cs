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
    
    public record OpProfile { public OpStage Stage; public TimeSpan Time;  }

    public record NodeProfile { public int NodeId; public OpType Op; public List<OpProfile> OpsProfile = new List<OpProfile>(); }

    public class Profiler
    {
        #region Fields
        private static Stopwatch timer = new Stopwatch();

        public static List<NodeProfile> Profile = new List<NodeProfile>();
        #endregion

        #region Properties
        public static NodeProfile CurrentNodeProfile => Profile.Last();
        
        public static OpProfile CurrentOpProfile => CurrentNodeProfile.OpsProfile.Last();

        public static bool Running => timer.IsRunning;
        #endregion

        #region Methods
        protected static void AddTimeIfTimerRunning()
        {
            if (Running)
            {
                timer.Stop();
                CurrentOpProfile.Time = timer.Elapsed;
                timer.Reset();
            }
        }

        public static void StartNodeProfile(int id, OpType op)
        {
            AddTimeIfTimerRunning();
            Profile.Add(new NodeProfile() { NodeId = id, Op = op });
        }

        public static void StopNodeProfile() => AddTimeIfTimerRunning();
        

        public static void StartOpStage(OpStage stage)
        {
            AddTimeIfTimerRunning();
            CurrentNodeProfile.OpsProfile.Add(new OpProfile() { Stage = stage, Time = TimeSpan.Zero });
            timer.Start();  
        }

        public static void StopOpStage() => AddTimeIfTimerRunning();
        #endregion

    }
}
