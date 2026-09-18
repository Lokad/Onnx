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
        CopyX,
        CopyY,
        Broadcast,
        ValidateArguments,
        CalculateIndices,
        Cast,
        GraphOrchestration
    }
    
    public record struct OpProfile { public OpStage Stage; public TimeSpan Time; }

    public record NodeProfile { public long NodeId; public OpType Op; public string Detail = ""; public Stack<OpProfile> OpsProfile = new Stack<OpProfile>(); }

    public record struct WallNode { public long NodeId; public OpType Op; public long StartTicks; public long EndTicks; }

    public sealed class ProfilerContext : IDisposable
    {
        public bool Enabled;
        public readonly Stack<NodeProfile> Profile = new Stack<NodeProfile>();
        private readonly Stopwatch timer = new Stopwatch();
        private readonly object sync = new object();
        private readonly ProfilerContext? previous;
        private readonly bool shared;
        private readonly bool wallOnly;
        internal bool WallOnly => wallOnly;
        public List<WallNode> Wall { get; } = new List<WallNode>();
        private int wallOpen = -1;

        internal ProfilerContext(bool enabled, ProfilerContext? previous, bool shared, bool wallOnly)
        {
            Enabled = enabled;
            this.previous = previous;
            this.shared = shared;
            this.wallOnly = wallOnly;
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
        public void StartNodeProfile(long id, OpType op) => StartNodeProfile(id, op, null);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void StartNodeProfile(long id, OpType op, Func<string>? detail)
        {
            if (!Enabled) return;

            lock (sync)
            {
                if (wallOnly)
                {
                    if (wallOpen >= 0)
                    {
                        var stale = Wall[wallOpen];
                        stale.EndTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                        Wall[wallOpen] = stale;
                    }
                    Wall.Add(new WallNode() { NodeId = id, Op = op, StartTicks = System.Diagnostics.Stopwatch.GetTimestamp() });
                    wallOpen = Wall.Count - 1;
                    return;
                }
                AddTimeLocked();
                Profile.Push(new NodeProfile() { NodeId = id, Op = op, Detail = detail is null ? "" : detail() });
                CurrentNodeProfile.OpsProfile.Push(new OpProfile() { Stage = OpStage.GraphOrchestration, Time = TimeSpan.Zero });
                timer.Start();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void StopNodeProfile()
        {
            if (!Enabled) return;
            lock (sync)
            {
                if (wallOnly)
                {
                    if (wallOpen >= 0)
                    {
                        var open = Wall[wallOpen];
                        open.EndTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                        Wall[wallOpen] = open;
                        wallOpen = -1;
                    }
                    return;
                }
                AddTimeLocked();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void StartOpStage(OpStage stage)
        {
            if (!Enabled || wallOnly) return;

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
        private static readonly ProfilerContext shared = new ProfilerContext(false, null, shared: true, wallOnly: false);
        private static readonly System.Threading.AsyncLocal<ProfilerContext?> ambient = new System.Threading.AsyncLocal<ProfilerContext?>();

        static ProfilerContext Current => ambient.Value ?? shared;

        public static ProfilerContext BeginExecution() => BeginExecution(Current.Enabled);

        public static ProfilerContext BeginWallExecution()
        {
            var wall = new ProfilerContext(true, ambient.Value, false, true);
            ambient.Value = wall;
            return wall;
        }

        public static ProfilerContext BeginExecution(bool enabled)
        {
            var current = ambient.Value;
            if (!enabled && (current is null || !current.Enabled))
                return shared;
            var ctx = new ProfilerContext(enabled, current, false, current is not null && current.WallOnly);
            ambient.Value = ctx;
            return ctx;
        }

        internal static void Restore(ProfilerContext? previous) => ambient.Value = previous;

        #region Methods
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StartNodeProfile(long id, OpType op) => Current.StartNodeProfile(id, op, null);

        public static void StartNodeProfile(long id, OpType op, Func<string>? detail) => Current.StartNodeProfile(id, op, detail);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StopNodeProfile() => Current.StopNodeProfile();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StartOpStage(OpStage stage) => Current.StartOpStage(stage);


        public static string StageDescription(OpStage stage) => stage switch
        {
            OpStage.GraphOrchestration => "Graph Orchestration",
            OpStage.Math => "Math",
            OpStage.Copy => "Copy",
            OpStage.CopyX => "CopyX",
            OpStage.CopyY => "CopyY",
            OpStage.Broadcast => "Broadcast",
            OpStage.ValidateArguments => "Validate Arguments",
            OpStage.CalculateIndices => "Calculate Indices",
            OpStage.Cast => "Cast",
            _ =>  throw new NotImplementedException()

        };
        #endregion

    }
}
