using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public enum OpStage
    {
        Math,
        Copy,
        Broadcast,
        ValidateArguments,
        CalculateIndices,
        Cast,
        GraphOrchestration
    }
    
    public record OpProfile { public OpStage Stage; public TimeSpan Time;  }

    public record NodeProfile { public long NodeId; public OpType Op; public Stack<OpProfile> OpsProfile = new Stack<OpProfile>(); }

    public class Profiler
    {
        #region Fields
        private static Stopwatch timer = new Stopwatch();

        public static bool Enabled = false;

        public static Stack<NodeProfile> Profile = new Stack<NodeProfile>();
        #endregion

        #region Properties
        public static NodeProfile CurrentNodeProfile => Profile.Peek();
        
        public static OpProfile CurrentOpProfile => CurrentNodeProfile.OpsProfile.Peek();

        public static bool Running => timer.IsRunning;
        #endregion

        #region Methods
        protected static void AddTimeIfTimerRunning()
        {
            if (!Enabled) return;

            if (Running)
            {
                timer.Stop();
                CurrentOpProfile.Time = timer.Elapsed;
                timer.Reset();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]  
        public static void StartNodeProfile(long id, OpType op)
        {
            if (!Enabled) return;
            
            AddTimeIfTimerRunning();
            Profile.Push(new NodeProfile() { NodeId = id, Op = op });
            CurrentNodeProfile.OpsProfile.Push(new OpProfile() { Stage = OpStage.GraphOrchestration, Time = TimeSpan.Zero });
            timer.Start();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StopNodeProfile() => AddTimeIfTimerRunning();


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StartOpStage(OpStage stage)
        {
            if (!Enabled) return;

            AddTimeIfTimerRunning();
            CurrentNodeProfile.OpsProfile.Push(new OpProfile() { Stage = stage, Time = TimeSpan.Zero });
            timer.Start();  
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StopOpStage() => AddTimeIfTimerRunning();
        #endregion

    }
}
