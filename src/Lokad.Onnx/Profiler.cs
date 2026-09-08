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
    
    public record struct OpProfile { public OpStage Stage; public TimeSpan Time; }

    public record NodeProfile { public long NodeId; public OpType Op; public Stack<OpProfile> OpsProfile = new Stack<OpProfile>(); }

    public sealed class ProfilerContext : IDisposable
    {
        public bool Enabled;
        public readonly Stack<NodeProfile> Profile = new Stack<NodeProfile>();
        private readonly Stopwatch timer = new Stopwatch();
        private readonly object sync = new object();
        private readonly ProfilerContext? previous;
        private readonly bool shared;

        internal ProfilerContext(bool enabled, ProfilerContext? previous, bool shared)
        {
            Enabled = enabled;
            this.previous = previous;
            this.shared = shared;
        }

        public void Dispose()
        {
            if (shared) return;
            Profiler.Restore(previous);
        }

        void AddTimeLocked()
        {
            if (timer.IsRunning)
            {
                timer.Stop();
                var top = Profile.Peek().OpsProfile.Pop();
                top.Time = timer.Elapsed;
                Profile.Peek().OpsProfile.Push(top);
                timer.Reset();
            }
        }

        void AddTimeIfTimerRunning()
        {
            if (!Enabled) return;
            lock (sync) { AddTimeLocked(); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void StartNodeProfile(long id, OpType op)
        {
            if (!Enabled) return;

            lock (sync)
            {
                AddTimeLocked();
                Profile.Push(new NodeProfile() { NodeId = id, Op = op });
                CurrentNodeProfile.OpsProfile.Push(new OpProfile() { Stage = OpStage.GraphOrchestration, Time = TimeSpan.Zero });
                timer.Start();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void StopNodeProfile() => AddTimeIfTimerRunning();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void StartOpStage(OpStage stage)
        {
            if (!Enabled) return;

            lock (sync)
            {
                AddTimeLocked();
                CurrentNodeProfile.OpsProfile.Push(new OpProfile() { Stage = stage, Time = TimeSpan.Zero });
                timer.Start();
            }
        }

        NodeProfile CurrentNodeProfile => Profile.Peek();

        internal bool Running => timer.IsRunning;
    }

    public class Profiler
    {
        private static readonly ProfilerContext shared = new ProfilerContext(false, null, shared: true);
        private static readonly System.Threading.AsyncLocal<ProfilerContext?> ambient = new System.Threading.AsyncLocal<ProfilerContext?>();

        static ProfilerContext Current => ambient.Value ?? shared;

        public static ProfilerContext BeginExecution() => BeginExecution(Current.Enabled);

        public static ProfilerContext BeginExecution(bool enabled)
        {
            var current = ambient.Value;
            if (!enabled && (current is null || !current.Enabled))
                return shared;
            var ctx = new ProfilerContext(enabled, current, false);
            ambient.Value = ctx;
            return ctx;
        }

        internal static void Restore(ProfilerContext? previous) => ambient.Value = previous;

        #region Methods
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StartNodeProfile(long id, OpType op) => Current.StartNodeProfile(id, op);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StopNodeProfile() => Current.StopNodeProfile();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StartOpStage(OpStage stage) => Current.StartOpStage(stage);


        public static string StageDescription(OpStage stage) => stage switch
        {
            OpStage.GraphOrchestration => "Graph Orchestration",
            OpStage.Math => "Math",
            OpStage.Copy => "Copy",
            OpStage.Broadcast => "Broadcast",
            OpStage.ValidateArguments => "Validate Arguments",
            OpStage.CalculateIndices => "Calculate Indices",
            OpStage.Cast => "Cast",
            _ =>  throw new NotImplementedException()

        };
        #endregion

    }
}
