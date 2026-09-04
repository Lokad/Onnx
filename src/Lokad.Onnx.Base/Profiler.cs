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

    public sealed class ProfilerContext : IDisposable
    {
        public bool Enabled;
        public readonly Stack<NodeProfile> Profile = new Stack<NodeProfile>();
        private readonly Stopwatch timer = new Stopwatch();
        private readonly object sync = new object();
        private readonly ProfilerContext? previous;

        internal ProfilerContext(bool enabled, ProfilerContext? previous)
        {
            Enabled = enabled;
            this.previous = previous;
        }

        public void Dispose() => Profiler.Restore(previous);

        void AddTimeLocked()
        {
            if (timer.IsRunning)
            {
                timer.Stop();
                CurrentOpProfile.Time = timer.Elapsed;
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
                AddTimeIfTimerRunning();
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
                AddTimeIfTimerRunning();
                CurrentNodeProfile.OpsProfile.Push(new OpProfile() { Stage = stage, Time = TimeSpan.Zero });
                timer.Start();
            }
        }

        NodeProfile CurrentNodeProfile => Profile.Peek();

        OpProfile CurrentOpProfile => CurrentNodeProfile.OpsProfile.Peek();

        internal bool Running => timer.IsRunning;
    }

    public class Profiler
    {
        private static readonly ProfilerContext shared = new ProfilerContext(false, null);
        private static readonly System.Threading.AsyncLocal<ProfilerContext?> ambient = new System.Threading.AsyncLocal<ProfilerContext?>();

        static ProfilerContext Current => ambient.Value ?? shared;

        public static bool Enabled { get => Current.Enabled; set => Current.Enabled = value; }

        public static Stack<NodeProfile> Profile => Current.Profile;

        public static NodeProfile CurrentNodeProfile => Current.Profile.Peek();

        public static OpProfile CurrentOpProfile => CurrentNodeProfile.OpsProfile.Peek();

        public static bool Running => Current.Running;

        public static ProfilerContext BeginExecution() => BeginExecution(Current.Enabled);

        public static ProfilerContext BeginExecution(bool enabled)
        {
            var ctx = new ProfilerContext(enabled, ambient.Value);
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

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        //public static void StopOpStage() => AddTimeIfTimerRunning();

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
