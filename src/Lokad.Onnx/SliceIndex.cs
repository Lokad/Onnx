namespace Lokad.Onnx
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.CompilerServices;

    /// <summary>
    /// One dimension of a Python-style slice selection: an optional inclusive
    /// start, an optional exclusive stop, and a step, plus markers for single
    /// index, ellipsis, and new-axis selections.
    /// </summary>
    public class SliceIndex
    {
        public static readonly SliceIndex All = new SliceIndex(null, null);
        public static readonly SliceIndex None = new SliceIndex(0, 0, 1);
        public static readonly SliceIndex Ellipsis = new SliceIndex(0, 0, 1) { IsEllipsis = true };
        public static readonly SliceIndex NewAxis = new SliceIndex(0, 0, 1) { IsNewAxis = true };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static SliceIndex Index(int index) => new SliceIndex(index, index + 1) { IsIndex = true };

        public int? Start;
        public int? Stop;
        public int Step;
        public bool IsIndex;
        public bool IsEllipsis;
        public bool IsNewAxis;

        public int? Length => Stop - Start;

        public SliceIndex(int? start) : this(start, null, 1) { }
        public SliceIndex(int? start, int? stop) : this(start, stop, 1) { }
        public SliceIndex(int? start, int? stop, int step)
        {
            Start = start;
            Stop = stop;
            Step = step;
        }

        public SliceIndex(string slice_notation)
        {
            Step = 1;
            Parse(slice_notation);
        }

        public static SliceIndex[] ParseSlices(string multi_slice_notation)
        {
            if (multi_slice_notation is null) throw new ArgumentNullException(nameof(multi_slice_notation));
            var tokens = multi_slice_notation.Split(',');
            var slices = new List<SliceIndex>();
            foreach (var token in tokens)
            {
                if (!string.IsNullOrWhiteSpace(token)) slices.Add(new SliceIndex(token));
            }
            return slices.ToArray();
        }

        public static string FormatSlices(params SliceIndex[] slices)
        {
            var texts = new string[slices.Length];
            for (int i = 0; i < slices.Length; i++) texts[i] = slices[i].ToString();
            return string.Join(",", texts);
        }

        static bool LooksLikeInteger(string text)
        {
            int i = 0;
            if (i < text.Length && (text[i] == '+' || text[i] == '-')) i++;
            int digits = 0;
            while (i < text.Length && text[i] >= '0' && text[i] <= '9')
            {
                i++;
                digits++;
            }
            return digits > 0 && i == text.Length;
        }

        static bool StrictShape(string text)
        {
            int i = 0;
            if (i < text.Length && (text[i] == '+' || text[i] == '-')) i++;
            while (i < text.Length && text[i] == ' ') i++;
            int digits = 0;
            while (i < text.Length && text[i] >= '0' && text[i] <= '9')
            {
                i++;
                digits++;
            }
            return digits > 0 && i == text.Length;
        }

        static string SqueezeSpaces(string text)
        {
            int gaps = 0;
            foreach (char c in text) if (c == ' ' || c == '\t') gaps++;
            if (gaps == 0) return text;
            char[] kept = new char[text.Length - gaps];
            int n = 0;
            foreach (char c in text)
            {
                if (c != ' ' && c != '\t') kept[n++] = c;
            }
            return new string(kept);
        }

        void Parse(string slice_notation)
        {
            if (string.IsNullOrEmpty(slice_notation))
                throw new ArgumentException("Slice notation expected, got empty string or null");
            string text = slice_notation.Trim();
            if (text == "...")
            {
                Start = 0;
                Stop = 0;
                Step = 1;
                IsEllipsis = true;
                return;
            }
            if (text == "newaxis" || text == "np.newaxis")
            {
                Start = 0;
                Stop = 0;
                Step = 1;
                IsNewAxis = true;
                return;
            }
            int firstColon = text.IndexOf(':');
            if (firstColon < 0)
            {
                if (!StrictShape(text))
                    throw new ArgumentException($"Invalid slice notation: '{slice_notation}'");
                string digits = SqueezeSpaces(text);
                if (!int.TryParse(digits, out var index))
                    throw new ArgumentException($"Invalid value for index: '{digits}'");
                Start = index;
                Stop = index + 1;
                Step = 1;
                IsIndex = true;
                return;
            }
            int secondColon = text.IndexOf(':', firstColon + 1);
            if (secondColon >= 0 && text.IndexOf(':', secondColon + 1) >= 0)
                throw new ArgumentException($"Invalid slice notation: '{slice_notation}'");
            string startPart = text.Substring(0, firstColon).Trim();
            string stopPart = secondColon < 0 ? text.Substring(firstColon + 1).Trim() : text.Substring(firstColon + 1, secondColon - firstColon - 1).Trim();
            string stepPart = secondColon < 0 ? "" : text.Substring(secondColon + 1).Trim();
            Start = ParseBound(startPart, "start", slice_notation);
            Stop = ParseBound(stopPart, "stop", slice_notation);
            if (stepPart.Length == 0)
            {
                Step = 1;
                return;
            }
            string squeezed = SqueezeSpaces(stepPart);
            if (!LooksLikeInteger(squeezed))
                throw new ArgumentException($"Invalid slice notation: '{slice_notation}'");
            if (!int.TryParse(squeezed, out var step))
                throw new ArgumentException($"Invalid value for step: '{squeezed}'");
            Step = step;
        }

        static int? ParseBound(string part, string role, string original)
        {
            if (part.Length == 0) return null;
            string squeezed = SqueezeSpaces(part);
            if (!LooksLikeInteger(squeezed))
                throw new ArgumentException($"Invalid slice notation: '{original}'");
            if (!int.TryParse(squeezed, out var value))
                throw new ArgumentException($"Invalid value for {role}: '{squeezed}'");
            return value;
        }

        public static bool operator ==(SliceIndex a, SliceIndex b)
        {
            if (ReferenceEquals(a, b)) return true;
            if (a is null || b is null) return false;
            return a.Start == b.Start && a.Stop == b.Stop && a.Step == b.Step;
        }

        public static bool operator !=(SliceIndex a, SliceIndex b) => !(a == b);

        public override bool Equals(object? obj)
        {
            if (obj is null || obj.GetType() != typeof(SliceIndex)) return false;
            var other = (SliceIndex)obj;
            return Start == other.Start && Stop == other.Stop && Step == other.Step;
        }

        public override int GetHashCode() => ToString().GetHashCode();

        public override string ToString()
        {
            if (IsIndex) return (Start ?? 0).ToString();
            if (IsNewAxis) return "np.newaxis";
            if (IsEllipsis) return "...";
            string head = Start == 0 ? "" : Start?.ToString() ?? "";
            string tail = Stop?.ToString() ?? "";
            string stride = Step == 1 ? "" : ":" + Step;
            return head + ":" + tail + stride;
        }

        /// <summary>
        /// Resolves this selection against an axis of the given size into an
        /// absolute start, step, and element count.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public SliceDef ToSliceDef(int dim)
        {
            if (IsIndex)
            {
                int index = Start ?? 0;
                if (index < 0)
                {
                    if (Math.Abs(index) > dim)
                        throw new ArgumentException($"Index {index} is out of bounds for the axis with size {dim}");
                    return new SliceDef(dim + index);
                }
                if (index > 0 && index >= dim)
                    throw new ArgumentException($"Index {index} is out of bounds for the axis with size {dim}");
                return new SliceDef(index);
            }
            if (Step == 0) return new SliceDef() { Count = 0, Start = 0, Step = 0 };
            // Long magnitude and count arithmetic: extreme steps (int.MinValue
            // saturates in from int64 bounds) must not overflow Abs or the
            // round-up division (verified against ORT 1.29: huge steps take
            // one element, or none when walking away).
            long magnitude = Math.Abs((long)Step);
            if (Step > 0)
            {
                int start = Start ?? 0;
                int stop = Stop ?? dim;
                if (start >= dim) return new SliceDef() { Count = 0, Start = 0, Step = 0 };
                if (start < 0) start = Math.Abs(start) <= dim ? dim + start : 0;
                if (stop > dim) stop = dim;
                if (stop < 0) stop = Math.Abs(stop) <= dim ? dim + stop : 0;
                if (start >= stop) return new SliceDef() { Count = 0, Start = 0, Step = 0 };
                return new SliceDef() { Start = start, Step = Step, Count = (int)(((long)Math.Abs(start - stop) + magnitude - 1) / magnitude) };
            }
            // Negative steps walk downward. A start below -dim clamps to 0
            // rather than yielding empty; kept deliberately for parity.
            int downStart = Start ?? (dim - 1);
            int downStop = Stop ?? -1;
            if (downStart < 0) downStart = Math.Abs(downStart) <= dim ? dim + downStart : 0;
            if (downStart >= dim) downStart = dim - 1;
            if (Stop < 0) downStop = Math.Abs(downStop) <= dim ? dim + downStop : -1;
            if (downStart <= downStop) return new SliceDef() { Count = 0, Start = 0, Step = 0 };
            return new SliceDef() { Start = downStart, Step = Step, Count = (int)(((long)Math.Abs(downStart - downStop) + magnitude - 1) / magnitude) };
        }

        public static SliceIndex operator ++(SliceIndex a)
        {
            if (a.Start.HasValue) a.Start++;
            if (a.Stop.HasValue) a.Stop++;
            return a;
        }

        public static SliceIndex operator --(SliceIndex a)
        {
            if (a.Start.HasValue) a.Start--;
            if (a.Stop.HasValue) a.Stop--;
            return a;
        }

        public static implicit operator SliceIndex(int index) => Index(index);
        public static implicit operator SliceIndex(string slice) => new SliceIndex(slice);
        public static implicit operator SliceIndex(Range range)
        {
            if (range.Equals(Range.All)) return Ellipsis;
            return new SliceIndex(start: FromIndex(range.Start), stop: FromIndex(range.End));
        }

        public static int? FromIndex(Index idx)
        {
            if (idx.Equals(^0)) return null;
            if (idx.IsFromEnd) return -idx.Value;
            return idx.Value;
        }

        public static SliceIndex FromObj(object index) => index switch
        {
            string s => (SliceIndex)s,
            int i => (SliceIndex)i,
            Range range => (SliceIndex)range,
            _ => throw new NotSupportedException(),
        };
    }

    public struct SliceDef
    {
        public int Start;
        public int Step;
        public int Count;

        public SliceDef(int start, int step, int count)
        {
            Start = start;
            Step = step;
            Count = count;
        }

        public SliceDef(int idx)
        {
            Start = idx;
            Step = 1;
            Count = -1;
        }

        public SliceDef(string def)
        {
            if (def == "()")
            {
                Start = 0;
                Step = 0;
                Count = 0;
                return;
            }
            int arrow = def.IndexOf(">>", StringComparison.Ordinal);
            int star = def.IndexOf('*', arrow + 2);
            if (!def.StartsWith("(", StringComparison.Ordinal) || !def.EndsWith(")", StringComparison.Ordinal) || arrow < 0 || star < 0)
                throw new FormatException("Invalid slice definition: '" + def + "'.");
            Start = int.Parse(def.Substring(1, arrow - 1));
            Step = int.Parse(def.Substring(arrow + 2, star - arrow - 2));
            Count = int.Parse(def.Substring(star + 1, def.Length - star - 2));
        }

        public bool IsIndex
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get => Count == -1;
        }

        public SliceDef Invert()
        {
            return new SliceDef() { Count = Count, Start = Start + Step * Count, Step = -Step };
        }

        public override string ToString()
        {
            if (IsIndex) return "[" + Start + "]";
            if (Count <= 0) return "()";
            return "(" + Start + ">>" + Step + "*" + Count + ")";
        }

        public SliceDef Merge(SliceDef other)
        {
            if (other.Count == 0) return new SliceDef() { Start = 0, Step = 0, Count = 0 };
            if (other.IsIndex) return new SliceDef(Start + other.Start * Step);
            return new SliceDef() { Start = Start + other.Start * Step, Step = Step * other.Step, Count = other.Count };
        }

        public static int[] InferNegativeCoordinates(int[] dimensions, int[] coords)
        {
            for (int i = 0; i < coords.Length; i++)
            {
                if (coords[i] < 0) coords[i] = dimensions[i] + coords[i];
            }
            return coords;
        }

        public static int[] ReplaySlicingOnCoords(int[] parentCoords, SliceDef[] slices)
        {
            var coords = new List<int>();
            for (int i = 0; i < parentCoords.Length; i++)
            {
                var slice = slices[i];
                var coord = parentCoords[i];
                if (slice.Count == -1) continue;
                if (slice.Count == 0) return Array.Empty<int>();
                bool ahead = slice.Step > 0 ? slice.Start > coord : slice.Start < coord;
                if (ahead) return Array.Empty<int>();
                if (coord % Math.Abs(slice.Step) != 0) return Array.Empty<int>();
                coords.Add((coord - slice.Start) / slice.Step);
            }
            return coords.ToArray();
        }
    }
}
