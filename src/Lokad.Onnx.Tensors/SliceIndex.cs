namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;


/// <summary>                                                                                                                                         <br></br>
/// NDArray can be indexed using slicing                                                                                                              <br></br>
/// A slice is constructed by start:stop:step notation                                                                                                <br></br>
///                                                                                                                                                   <br></br>
/// Examples:                                                                                                                                         <br></br>
///                                                                                                                                                   <br></br>
/// a[start:stop]  # items start through stop-1                                                                                                       <br></br>
/// a[start:]      # items start through the rest of the array                                                                                        <br></br>
/// a[:stop]       # items from the beginning through stop-1                                                                                          <br></br>
///                                                                                                                                                   <br></br>
/// The key point to remember is that the :stop value represents the first value that is not                                                          <br></br>
/// in the selected slice. So, the difference between stop and start is the number of elements                                                        <br></br>
/// selected (if step is 1, the default).                                                                                                             <br></br>
///                                                                                                                                                   <br></br>
/// There is also the step value, which can be used with any of the above:                                                                            <br></br>
/// a[:]           # a copy of the whole array                                                                                                        <br></br>
/// a[start:stop:step] # start through not past stop, by step                                                                                         <br></br>
///                                                                                                                                                   <br></br>
/// The other feature is that start or stop may be a negative number, which means it counts                                                           <br></br>
/// from the end of the array instead of the beginning. So:                                                                                           <br></br>
/// a[-1]    # last item in the array                                                                                                                 <br></br>
/// a[-2:]   # last two items in the array                                                                                                            <br></br>
/// a[:-2]   # everything except the last two items                                                                                                   <br></br>
/// Similarly, step may be a negative number:                                                                                                         <br></br>
///                                                                                                                                                   <br></br>
/// a[::- 1]    # all items in the array, reversed                                                                                                    <br></br>
/// a[1::- 1]   # the first two items, reversed                                                                                                       <br></br>
/// a[:-3:-1]  # the last two items, reversed                                                                                                         <br></br>
/// a[-3::- 1]  # everything except the last two items, reversed                                                                                      <br></br>
///                                                                                                                                                   <br></br>
/// NumSharp is kind to the programmer if there are fewer items than                                                                                  <br></br>
/// you ask for. For example, if you  ask for a[:-2] and a only contains one element, you get an                                                      <br></br>
/// empty list instead of an error.Sometimes you would prefer the error, so you have to be aware                                                      <br></br>
/// that this may happen.                                                                                                                             <br></br>
///                                                                                                                                                   <br></br>
/// Adapted from Greg Hewgill's answer on Stackoverflow: https://stackoverflow.com/questions/509211/understanding-slice-notation                      <br></br>
///                                                                                                                                                   <br></br>
/// Note: special IsIndex == true                                                                                                                     <br></br>
/// It will pick only a single value at Start in this dimension effectively reducing the Shape of the sliced matrix by 1 dimension.                   <br></br>
/// It can be used to reduce an N-dimensional array/matrix to a (N-1)-dimensional array/matrix                                                        <br></br>
///                                                                                                                                                   <br></br>
/// Example:                                                                                                                                          <br></br>
/// a=[[1, 2], [3, 4]]                                                                                                                                <br></br>
/// a[:, 1] returns the second column of that 2x2 matrix as a 1-D vector                                                                              <br></br>
/// </summary>
[DebuggerStepThrough]
public class SliceIndex
{
    /// <summary>
    /// return : for this dimension
    /// </summary>
    public static readonly SliceIndex All = new SliceIndex(null, null);

    /// <summary>
    /// return 0:0 for this dimension
    /// </summary>
    public static readonly SliceIndex None = new SliceIndex(0, 0, 1);

    /// <summary>
    /// fill up the missing dimensions with : at this point, corresponds to ... 
    /// </summary>
    public static readonly SliceIndex Ellipsis = new SliceIndex(0, 0, 1) { IsEllipsis = true };

    /// <summary>
    /// insert a new dimension at this point
    /// </summary>
    public static readonly SliceIndex NewAxis = new SliceIndex(0, 0, 1) { IsNewAxis = true };

    /// <summary>
    /// return exactly one element at this dimension and reduce the shape from n-dim to (n-1)-dim
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static SliceIndex Index(int index) => new SliceIndex(index, index + 1) { IsIndex = true };

    ///// <summary>
    ///// return multiple elements for this dimension specified by the given index array (or boolean mask array)
    ///// </summary>
    ///// <param name="index_array_or_mask"></param>
    ///// <returns></returns>
    //[MethodImpl(MethodImplOptions.AggressiveInlining)]
    //public static Slice Select(NDArray index_array_or_mask) => new Slice(null, null) { Selection=index_array_or_mask };

    public int? Start;
    public int? Stop;
    public int Step;
    public bool IsIndex;
    public bool IsEllipsis;
    public bool IsNewAxis;

    ///// <summary>
    ///// Array of integer indices to select elements by index extraction or boolean values to select by masking the elements of the given dimension.
    ///// </summary>
    //public NDArray Selection = null;

    /// <summary>
    /// Length of the slice. 
    /// <remarks>
    /// The length is not guaranteed to be known for i.e. a slice like ":". Make sure to check Start and Stop 
    /// for null before using it</remarks>
    /// </summary>
    public int? Length => Stop - Start;

    /// <summary>
    /// ndarray can be indexed using slicing
    /// slice is constructed by start:stop:step notation
    /// </summary>
    /// <param name="start">Start index of the slice, null means from the start of the array</param>
    /// <param name="stop">Stop index (first index after end of slice), null means to the end of the array</param>
    /// <param name="step">Optional step to select every n-th element, defaults to 1</param>
    public SliceIndex(int? start = null, int? stop = null, int step = 1)
    {
        Start = start;
        Stop = stop;
        Step = step;
    }

    public SliceIndex(string slice_notation)
    {
        Parse(slice_notation);
    }

    /// <summary>
    /// Parses Python array slice notation and returns an array of Slice objects
    /// </summary>
    public static SliceIndex[] ParseSlices(string multi_slice_notation)
    {
        return Regex.Split(multi_slice_notation, @",\s*").Where(s => !string.IsNullOrWhiteSpace(s)).Select(token => new SliceIndex(token)).ToArray();
    }

    /// <summary>
    /// Creates Python array slice notation out of an array of Slice objects (mainly used for tests)
    /// </summary>
    public static string FormatSlices(params SliceIndex[] slices)
    {
        return string.Join(",", slices.Select(s => s.ToString()));
    }

    private void Parse(string slice_notation)
    {
        if (string.IsNullOrEmpty(slice_notation))
            throw new ArgumentException("Slice notation expected, got empty string or null");
        var match = Regex.Match(slice_notation, @"^\s*((?'start'[+-]?\s*\d+)?\s*:\s*(?'stop'[+-]?\s*\d+)?\s*(:\s*(?'step'[+-]?\s*\d+)?)?|(?'index'[+-]?\s*\d+)|(?'ellipsis'\.\.\.)|(?'newaxis'(np\.)?newaxis))\s*$");
        if (!match.Success)
            throw new ArgumentException($"Invalid slice notation: '{slice_notation}'");
        if (match.Groups["ellipsis"].Success)
        {
            Start = 0;
            Stop = 0;
            Step = 1;
            IsEllipsis = true;
            return;
        }
        if (match.Groups["newaxis"].Success)
        {
            Start = 0;
            Stop = 0;
            Step = 1;
            IsNewAxis = true;
            return;
        }
        if (match.Groups["index"].Success)
        {
            if (!int.TryParse(Regex.Replace(match.Groups["index"].Value ?? "", @"\s+", ""), out var start))
                throw new ArgumentException($"Invalid value for index: '{match.Groups["index"].Value}'");
            Start = start;
            Stop = start + 1;
            Step = 1; // special case for dimensionality reduction by picking a single element
            IsIndex = true;
            return;
        }
        var start_string = Regex.Replace(match.Groups["start"].Value ?? "", @"\s+", ""); // removing spaces from match to be able to parse what python allows, like: "+ 1" or  "-   9";
        var stop_string = Regex.Replace(match.Groups["stop"].Value ?? "", @"\s+", "");
        var step_string = Regex.Replace(match.Groups["step"].Value ?? "", @"\s+", "");

        if (string.IsNullOrWhiteSpace(start_string))
            Start = null;
        else
        {
            if (!int.TryParse(start_string, out var start))
                throw new ArgumentException($"Invalid value for start: {start_string}");
            Start = start;
        }

        if (string.IsNullOrWhiteSpace(stop_string))
            Stop = null;
        else
        {
            if (!int.TryParse(stop_string, out var stop))
                throw new ArgumentException($"Invalid value for start: {stop_string}");
            Stop = stop;
        }

        if (string.IsNullOrWhiteSpace(step_string))
            Step = 1;
        else
        {
            if (!int.TryParse(step_string, out var step))
                throw new ArgumentException($"Invalid value for start: {step_string}");
            Step = step;
        }
    }

    #region Equality comparison

    public static bool operator ==(SliceIndex a, SliceIndex b)
    {
        if (ReferenceEquals(a, b))
            return true;

        if (a is null || b is null)
            return false;

        return a.Start == b.Start && a.Stop == b.Stop && a.Step == b.Step;
    }

    public static bool operator !=(SliceIndex a, SliceIndex b)
    {
        return !(a == b);
    }

    public override bool Equals(object obj)
    {
        if (obj == null)
            return false;

        if (obj.GetType() != typeof(SliceIndex))
            return false;

        var b = (SliceIndex)obj;
        return Start == b.Start && Stop == b.Stop && Step == b.Step;
    }

    public override int GetHashCode()
    {
        return ToString().GetHashCode();
    }

    #endregion

    public override string ToString()
    {
        if (IsIndex)
            return $"{Start ?? 0}";
        else if (IsNewAxis)
            return "np.newaxis";
        else if (IsEllipsis)
            return "...";
        var optional_step = Step == 1 ? "" : $":{Step}";
        return $"{(Start == 0 ? "" : Start.ToString())}:{(Stop == null ? "" : Stop.ToString())}{optional_step}";
    }

    // return the size of the slice, given the data dimension on this axis
    // note: this works only with sanitized shapes!
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetSize()
    {
        var astep = Math.Abs(Step);
        return (Math.Abs(Start.Value - Stop.Value) + (astep - 1)) / astep;
    }

    /// <summary>
    /// Converts the user Slice into an internal SliceDef which is easier to calculate with
    /// </summary>
    /// <param name="dim"></param>
    /// <returns></returns>
    [MethodImpl((MethodImplOptions)768)]
    public SliceDef ToSliceDef(int dim)
    {
        if (IsIndex)
        {
            var index = Start ?? 0;
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

        if (Step == 0)
            return new SliceDef() { Count = 0, Start = 0, Step = 0 };
        var astep = Math.Abs(Step);
        if (Step > 0)
        {
            var start = Start ?? 0;
            var stop = Stop ?? dim;
            if (start >= dim)
                return new SliceDef() { Count = 0, Start = 0, Step = 0 };
            if (start < 0)
                start = Math.Abs(start) <= dim ? dim + start : 0;
            if (stop > dim)
                stop = dim;
            if (stop < 0)
                stop = Math.Abs(stop) <= dim ? dim + stop : 0;
            if (start >= stop)
                return new SliceDef() { Count = 0, Start = 0, Step = 0 };
            var count = (Math.Abs(start - stop) + (astep - 1)) / astep;
            return new SliceDef() { Start = start, Step = Step, Count = count };
        }
        else
        {
            // negative step!
            var start = Start ?? dim - 1;
            var stop = Stop ?? -1;
            if (start < 0)
                start = Math.Abs(start) <= dim ? dim + start : 0;
            if (start >= dim)
                start = dim - 1;
            if (Stop < 0)
                stop = Math.Abs(stop) <= dim ? dim + stop : -1;
            if (start <= stop)
                return new SliceDef() { Count = 0, Start = 0, Step = 0 };
            var count = (Math.Abs(start - stop) + (astep - 1)) / astep;
            var retval = new SliceDef() { Start = start, Step = Step, Count = count };
            return retval;
        }
    }


    #region Operators

    public static SliceIndex operator ++(SliceIndex a)
    {
        if (a.Start.HasValue)
            a.Start++;
        if (a.Stop.HasValue)
            a.Stop++;
        return a;
    }

    public static SliceIndex operator --(SliceIndex a)
    {
        if (a.Start.HasValue)
            a.Start--;
        if (a.Stop.HasValue)
            a.Stop--;
        return a;
    }

    public static implicit operator SliceIndex(int index) => Index(index);
    public static implicit operator SliceIndex(string slice) => new SliceIndex(slice);
    public static implicit operator SliceIndex(Range range)
    {
        if (range.Equals(Range.All))
        {
            return Ellipsis;
        }
        else
        {
            return new SliceIndex(start: range.Start.Value, stop: range.End.IsFromEnd ? -range.End.Value : range.End.Value);
        }
    }
        //new SliceIndex(range.Start.Value, range.End.Value);
    //public static implicit operator Slice(NDArray selection) => Slice.Select(selection);

    #endregion

    public static SliceIndex FromObj(object index) => index switch
    {
        string s => (SliceIndex)s,
        int i => (SliceIndex)i,
        _ => throw new NotSupportedException()
    };

}

public struct SliceDef
{
    public int Start; // start index in array
    public int Step; // positive => forward from Start, 
    public int Count; // number of steps to take from Start (1 means just take Start, 0 means take nothing, -1 means this is an index)

    public SliceDef(int start, int step, int count)
    {
        (Start, Step, Count) = (start, step, count);
    }

    public SliceDef(int idx)
    {
        (Start, Step, Count) = (idx, 1, -1);
    }

    /// <summary>
    /// (Start>>Step*Count)
    /// </summary>
    /// <param name="def"></param>
    public SliceDef(string def)
    {
        if (def == "()")
        {
            (Start, Step, Count) = (0, 0, 0);
            return;
        }

        var m = Regex.Match(def, @"\((\d+)>>(-?\d+)\*(\d+)\)");
        Start = int.Parse(m.Groups[1].Value);
        Step = int.Parse(m.Groups[2].Value);
        Count = int.Parse(m.Groups[3].Value);
    }

    public bool IsIndex
    {
        [MethodImpl((MethodImplOptions)768)]
        get => Count == -1;
    }

    /// <summary>
    /// reverts the order of the slice sequence
    /// </summary>
    /// <returns></returns>
    [MethodImpl((MethodImplOptions)768)]
    public SliceDef Invert()
    {
        return new SliceDef() { Count = Count, Start = (Start + Step * Count), Step = -Step };
    }

    public override string ToString()
    {
        if (IsIndex)
            return $"[{Start}]";
        if (Count <= 0)
            return "()";
        return $"({Start}>>{Step}*{Count})";
    }

    /// <summary>
    /// Merge calculates the resulting one-time slice on the original data if it is sliced repeatedly
    /// </summary>
    [MethodImpl((MethodImplOptions)768)]
    public SliceDef Merge(SliceDef other)
    {
        if (other.Count == 0)
            return new SliceDef() { Start = 0, Step = 0, Count = 0 };
        var self = this;
        if (other.IsIndex)
            return new SliceDef(self.Start + other.Start * self.Step);
        var result = new SliceDef() { Start = self.Start + other.Start * self.Step, Step = Step * other.Step, Count = other.Count, };
        return result;
    }

    /// <summary>
    ///     Translates coordinates with negative indices, e.g:<br></br>
    ///     np.arange(9)[-1] == np.arange(9)[8]<br></br>
    ///     np.arange(9)[-2] == np.arange(9)[7]<br></br>
    /// </summary>
    /// <param name="dimensions">The dimensions these coordinates are targeting</param>
    /// <param name="coords">The coordinates.</param>
    /// <returns>Coordinates without negative indices.</returns>
    public static int[] InferNegativeCoordinates(int[] dimensions, int[] coords)
    {
        for (int i = 0; i < coords.Length; i++)
        {
            var curr = coords[i];
            if (curr < 0)
                coords[i] = dimensions[i] + curr;
        }

        return coords;
    }

    /// <summary>
    ///     Get offset index out of coordinate indices.
    ///
    ///     The offset is the absolute offset in memory for the given coordinates.
    ///     Even for shapes that were sliced and reshaped after slicing and sliced again (and so forth)
    ///     this returns the absolute memory offset.
    ///
    ///     Note: the inverse operation to this is GetCoordinatesFromAbsoluteIndex
    /// </summary>
    /// <param name="indices">The coordinates to turn into linear offset</param>
    /// <returns>The index in the memory block that refers to a specific value.</returns>
    /// <remarks>Handles sliced indices and broadcasting</remarks>
    
    public static int[] ReplaySlicingOnCoords(int[] parentCoords, SliceDef[] slices)
    {
        var coords = new List<int>();
        for (int i = 0; i < parentCoords.Length; i++)
        {
            var slice = slices[i];
            var coord = parentCoords[i];
            if (slice.Count == -1) // this is a Slice.Index so we remove this dim from coords
                continue;
            if (slice.Count == 0) // this is a Slice.None which means there is no set of coordinates that can index anything in this shape
                return new int[0];
            if (slice.Start > coord && slice.Step > 0 || slice.Start < coord && slice.Step < 0) // outside of the slice, return empty coords
                return new int[0];
            if (coord % Math.Abs(slice.Step) != 0) // coord is between the steps, so we are "outside" of this shape, return empty coords
                return new int[0];
            coords.Add((coord - slice.Start) / slice.Step);
        }

        return coords.ToArray();

    }
}



