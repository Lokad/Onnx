namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Collections.Generic;

using static Lokad.Onnx.OpResult;
using CPU = CPUExecutionProvider;

public partial struct Node
{
    public long ID;
    public string Name;
    public Dictionary<string, object>? Attributes;
    public OpType Op;
    public string[] Inputs;
    public string[] Outputs;
    public string? Domain;
    public string? OpTypeName;
    public int OpsetVersion;
    public bool IsFused;

    public string DescribeOperator() => OperatorSchemas.Describe(this);

    public static bool IsStandardDomain(string? domain) =>
        OperatorSchemas.IsStandardDomain(domain);

    public bool HasAttr<T>(string name) => Attributes is not null && Attributes.ContainsKey(name) && Attributes[name].GetType() == typeof(T);

    public T? Attr<T>(string name, T? d) => Attributes is not null && Attributes.ContainsKey(name) && Attributes[name].GetType() == typeof(T) ?
        (T)Attributes[name] : d;

    public T RequiredAttr<T>(string name) => Attributes is not null && Attributes.ContainsKey(name) && Attributes[name].GetType() == typeof(T) ?
    (T)Attributes[name] : throw new ArgumentException("The attribute " + name + " is required for node execution but was not found.");

    public object? OneOfAttr(params string[] names)
    {
        var a = Attributes;
        if (a is null) return null;
        for (int i = 0; i < names.Length; i++)
        {
            if (a.TryGetValue(names[i], out var value)) return value;
        }
        return null;
    }

    public int? Int(string name, int? d) => GetInt(name, d);

    public int? GetInt(string name, int? d)
    {
        if (Attributes is null) return d;
        if (!Attributes.TryGetValue(name, out var v) || v is null) return d;
        if (v is int i) return i;
        if (v is long l) return checked((int)l);
        throw new ArgumentException("The attribute " + name + " must be an integer.");
    }

    public float? GetFloat(string name, float? d)
    {
        if (Attributes is null) return d;
        if (!Attributes.TryGetValue(name, out var v) || v is null) return d;
        if (v is float f) return f;
        if (v is double g) return checked((float)g);
        if (v is int i) return i;
        if (v is long l) return checked((float)l);
        throw new ArgumentException("The attribute " + name + " must be a float.");
    }

    public bool? GetBool(string name, bool? d)
    {
        if (Attributes is null) return d;
        if (!Attributes.TryGetValue(name, out var v) || v is null) return d;
        if (v is bool b) return b;
        if (v is int i) return i != 0;
        if (v is long l) return l != 0L;
        throw new ArgumentException("The attribute " + name + " must be a boolean or 0/1 integer.");
    }

    public bool GetReshapeAllowZero()
    {
        if (Attributes is not null)
        {
            if (Attributes.TryGetValue("allowzero", out var v) && v is not null)
            {
                if (v is bool b) return b;
                if (v is int i) return i != 0;
                if (v is long l) return l != 0L;
                throw new ArgumentException("The attribute allowzero must be a 0/1 integer.");
            }
            if (Attributes.TryGetValue("allow_zero", out var w) && w is not null)
            {
                if (w is bool b2) return b2;
                throw new ArgumentException("The attribute allow_zero must be a boolean.");
            }
        }
        return false;
    }

    public int RequiredInt(string name) => Int(name, null) ?? throw new ArgumentException($"The Int attribute {name} is required but was not found.");

    public int[]? Ints(string name)
    {
        if (Attributes is null) return null;
        if (!Attributes.TryGetValue(name, out var v) || v is null) return null;
        if (v is int[] ia) return ia;
        if (v is long[] la)
        {
            var converted = new int[la.Length];
            for (int i = 0; i < la.Length; i++) converted[i] = checked((int)la[i]);
            return converted;
        }
        throw new ArgumentException("The attribute " + name + " must be an integer array.");
    }

    public int[] RequiredInts(string name) => Ints(name) ?? throw new ArgumentException($"The Ints attribute {name} is required but was not found.");
    /// <summary>
    /// Legacy Pad constant value: pre-11 stores a FLOAT attribute while newer
    /// graphs carry it as an input. Converts the attribute to a single-element
    /// tensor matching the data dtype so the provider sees one scalar form.
    /// </summary>
    public ITensor? PadConstantAttr(ComputationalGraph graph)
    {
        if (Attributes is null || !Attributes.TryGetValue("value", out var v) || v is null) return null;
        if (v is ITensor t) return t;
        float f;
        if (v is float ff) f = ff;
        else if (v is double dd) f = checked((float)dd);
        else if (v is int ii) f = ii;
        else if (v is long ll) f = checked((float)ll);
        else throw new ArgumentException("The attribute value must be a float or tensor.");
        TensorElementType? dtype = null;
        try
        {
            if (Inputs.Length > 0 && !string.IsNullOrEmpty(Inputs[0])) dtype = graph.GetInputTensor(Inputs[0]).ElementType;
        }
        catch { dtype = null; }
        if (dtype == TensorElementType.Double) return new DenseTensor<double>(new double[] { f }, new[] { 1 });
        if (dtype == TensorElementType.Int32) return new DenseTensor<int>(new int[] { checked((int)f) }, new[] { 1 });
        if (dtype == TensorElementType.Int64) return new DenseTensor<long>(new long[] { checked((long)f) }, new[] { 1 });
        return new DenseTensor<float>(new float[] { f }, new[] { 1 });
    }


    public ITensor? InputTensor(ComputationalGraph graph, int index) =>
        index < Inputs.Length && !string.IsNullOrEmpty(Inputs[index]) ? graph.GetInputTensor(Inputs[index]) : null;

    /// <summary>
    /// Transpose with a prepared-plan shortcut: constant-only transpositions
    /// are computed once, stored as plan-owned initializers, and reused while
    /// the source initializer is unchanged. Anything else computes normally.
    /// </summary>
    internal OpResult TransposePrepared(ComputationalGraph graph, ExecutionOptions? opt)
    {
        var input = InputTensor(graph, 0);
        if (graph.TryGetFoldedTranspose(Name, Inputs, input, out var prepared) && prepared is not null)
            return Success(OpType.Transpose, prepared);
        var r = CPU.Transpose(input, Ints("perm"), opt, graph.ActivePool);
        if (r.Status == OpStatus.Success && r.Outputs.Length == 1 && r.Outputs[0] is not null)
            graph.FoldTranspose(Name, Inputs, input, r.Outputs[0]);
        return r;
    }

    /// <summary>
    /// Version driving version-sensitive dispatch for this node: the resolved
    /// node version when set, else the graph opset for the node domain with
    /// the same empty-string fallback the importer uses, else 0 for legacy
    /// attribute-form behavior on version-unknown hand-built graphs.
    /// </summary>
    public int ResolvedOpsetVersion(ComputationalGraph graph)
    {
        if (OpsetVersion > 0) return OpsetVersion;
        if (!string.IsNullOrEmpty(Domain) && graph.Opset.TryGetValue(Domain, out var dv)) return dv;
        if (graph.Opset.TryGetValue("", out var v)) return v;
        return 0;
    }


    /// <summary>
    /// Executes the selected If branch with scoped locals against the enclosing run
    /// bindings. Captured outer values (inputs, initializers, already
    /// computed intermediates) resolve from the parent maps; branch-local
    /// initializers and intermediates shadow any outer binding only for the
    /// branch duration and are restored afterwards, so sibling branches,
    /// repeated condition flips, and nesting cannot observe each other. The If
    /// outputs are collected before restore for the dispatch loop to rebind
    /// under the node output names.
    /// </summary>
    OpResult ExecuteIf(ComputationalGraph graph, ExecutionOptions? options)
    {
        var op = OpType.If;
        var cond = InputTensor(graph, 0);
        if (cond is null) return MissingInput(op, "cond");
        if (cond.ElementType != TensorElementType.Bool || cond.Length != 1)
            return WrongInputType(op, "cond", "The If condition must be a scalar bool tensor.", cond);
        bool take = ((Tensor<bool>)cond).ToArray()[0];
        string key = take ? "then_branch" : "else_branch";
        if (Attributes is null || !Attributes.TryGetValue(key, out var bv) || bv is not ComputationalGraph branch)
            return MissingAttribute(op, key, "The If branch was not imported as an executable subgraph.");
        var saved = new Dictionary<string, (bool exists, ITensor? value)>(StringComparer.Ordinal);
        void SaveBinding(string name)
        {
            if (saved.ContainsKey(name)) return;
            if (graph.IntermediateOutputs.TryGetValue(name, out var prev)) saved[name] = (true, prev);
            else saved[name] = (false, null);
        }
        try
        {
            foreach (var kv in branch.Initializers)
            {
                SaveBinding(kv.Key);
                graph.IntermediateOutputs[kv.Key] = kv.Value;
            }
            foreach (var bn in branch.Nodes)
            {
                Profiler.StartNodeProfile(bn.ID, bn.Op, () => bn.Name);
                OpResult r;
                try
                {
                    r = bn.Execute(graph, ExecutionProvider.CPU, options);
                }
                finally
                {
                    Profiler.StopNodeProfile();
                }
                if (r.Status != OpStatus.Success)
                    return Failure(op, "Branch " + key + " node " + bn.Name + " (" + bn.Op + ") failed: " + r.Message, r.Cause);
                for (int i = 0; i < bn.Outputs.Length; i++)
                {
                    if (string.IsNullOrEmpty(bn.Outputs[i])) continue;
                    SaveBinding(bn.Outputs[i]);
                    graph.IntermediateOutputs[bn.Outputs[i]] = r.Outputs[i];
                    r.Outputs[i].Name = bn.Outputs[i];
                }
            }
            // Branch outputs map positionally onto the If outputs; the names
            // differ (the dispatch loop rebinds the returned tensors under the
            // node output names), so collect the branch-declared names.
            var outs = new ITensor[Outputs.Length];
            for (int i = 0; i < Outputs.Length; i++)
            {
                string bname = i < branch.OutputDescs.Count ? branch.OutputDescs[i].Name : "";
                if (string.IsNullOrEmpty(bname) || !graph.IntermediateOutputs.TryGetValue(bname, out var bound) || bound is null)
                    return Failure(op, "Branch " + key + " did not produce output " + Outputs[i] + ".");
                outs[i] = bound;
            }
            return Success(op, outs);
        }
        finally
        {
            foreach (var kv in saved)
            {
                if (kv.Value.exists) graph.IntermediateOutputs[kv.Key] = kv.Value.value;
                else graph.IntermediateOutputs.Remove(kv.Key);
            }
        }
    }

    public OpResult Execute(ComputationalGraph graph, ExecutionProvider provider, ExecutionOptions? options)
    {
        try
        {
            if (provider == ExecutionProvider.CPU)
            {
 
                var r = ExecuteCPU(graph, options);
                
                if (r.Status == OpStatus.Success && r.Outputs.Length != Outputs.Length)
                {
                    return Failure(Op, $"The operation returned {r.Outputs.Length} outputs but the graph node has {Outputs.Length} outputs.");
                }
                else
                {
                    for (int i = 0; i < r.Outputs.Length; i++)
                    {
                        r.Outputs[i].Name = this.Outputs[i];
                    }
                    return r;
                }
            }
            else
            {
                throw new NotSupportedException();
            }
        }
        catch (ArgumentNullException ane)
        {
            var r = !string.IsNullOrEmpty(ane.ParamName) ? MissingInput(Op, ane.ParamName) : Failure(Op, ane.Message);
            r.Cause = ane;
            return r;
        }
        catch (TensorInputShapeException tise)
        {
            var r = WrongInputShape(Op, tise.Name, tise.Shape, tise.Input);
            r.Cause = tise;
            return r;
        }
        catch (Exception e) when (!Runtime.IsFatal(e))
        {
            return Failure(Op, e.Message, e);
        }
    }

        static OpResult CastChecked(ITensor? input, int to, int? saturate, ExecutionOptions? opt)
    {
        var op = OpType.Cast;
        if (!Enum.IsDefined(typeof(TensorElementType), to)) return AttributeNotSupported(op, "to", to.ToString(), null);
        if (saturate.HasValue && saturate.Value != 1) return AttributeNotSupported(op, "saturate", saturate.Value.ToString(), "Only saturating casts are supported.");
        return CPU.Cast(input, (TensorElementType)to, opt);
    }

    /// <summary>
    /// Sub-32-bit integer arithmetic (int8/uint8/int16/uint16) joined the
    /// Add/Sub/Mul/Div kernels at opset 14 (verified against ORT 1.29:
    /// opset-13 refuses all four widths); older or version-unknown graphs
    /// keep the descriptive provider refusal.
    /// </summary>
    OpResult? Sub32Gate(ComputationalGraph graph, OpType op)
    {
        int v = ResolvedOpsetVersion(graph);
        if (v >= 14) return null;
        var t = Inputs.Length > 0 ? InputTensor(graph, 0) : null;
        if (t is not null && (t.ElementType is TensorElementType.Int8 or TensorElementType.UInt8 or TensorElementType.Int16 or TensorElementType.UInt16))
            return InputTypeNotSupported(op, "A", t, null);
        return null;
    }
    /// <summary>
    /// int32 MatMul joined at opset 9 (verified against ORT 1.29: older
    /// versions refuse int32 inputs); older or version-unknown graphs keep
    /// the descriptive provider refusal.
    /// </summary>
    /// <summary>
    /// int8/int32 Relu joined at opset 14 (verified against ORT 1.29:
    /// opset-13 refuses both); older or version-unknown graphs keep the
    /// descriptive provider refusal.
    /// </summary>
    OpResult? ReluIntGate(ComputationalGraph graph)
    {
        int v = ResolvedOpsetVersion(graph);
        if (v >= 14) return null;
        var t = Inputs.Length > 0 ? InputTensor(graph, 0) : null;
        if (t is not null && (t.ElementType is TensorElementType.Int8 or TensorElementType.Int32))
            return InputTypeNotSupported(OpType.Relu, "X", t, null);
        return null;
    }
    /// <summary>
    /// Routes Mul nodes carrying the Sigmoid+Mul fusion marker to the fused
    /// gated-multiply kernel; unmarked nodes fall through to plain Mul.
    /// </summary>
    OpResult? MulSigmoidGate(ComputationalGraph graph, ExecutionOptions? opt)
    {
        int? k = GetInt("fuse_sigmoid", null);
        if (!k.HasValue) return null;
        if (k.Value != 0 && k.Value != 1) return OpResult.AttributeNotSupported(OpType.Mul, "fuse_sigmoid", k.Value.ToString(), "Fused Sigmoid+Mul selects input 0 or 1.");
        return CPU.MulSigmoid(InputTensor(graph, 0), InputTensor(graph, 1), k.Value, opt, graph.ActivePool);
    }
    /// <summary>
    /// Routes MatMul nodes carrying the MatMul+scale fusion marker to the scaled
    /// provider; unmarked nodes fall through to plain MatMul.
    /// </summary>
    /// <summary>
    /// Routes Conv nodes carrying a fused blocked-region plan to the region
    /// executor; unmarked nodes fall through to the standard convolution.
    /// </summary>
    OpResult? RegionGate(ComputationalGraph graph, ExecutionOptions? opt)
    {
        if (Attributes is null || !Attributes.TryGetValue("fuse_region", out var r) || r is not BlockedRegionSpec spec) return null;
        return CPU.RunBlockedRegion(spec, Inputs.Length > 0 ? InputTensor(graph, 0) : null, graph, opt);
    }

    OpResult? MatMulScaleGate(ComputationalGraph graph, ExecutionOptions? opt)
    {
        float? s = GetFloat("fuse_scale", null);
        if (!s.HasValue) return null;
        return CPU.MatMulScaled(InputTensor(graph, 0), InputTensor(graph, 1), s.Value, opt, graph.ActivePool);
    }
    OpResult? MatMulIntGate(ComputationalGraph graph)
    {
        int v = ResolvedOpsetVersion(graph);
        if (v >= 9) return null;
        var t = Inputs.Length > 0 ? InputTensor(graph, 0) : null;
        if (t is not null && t.ElementType == TensorElementType.Int32)
            return InputTypeNotSupported(OpType.MatMul, "A", t, null);
        return null;
    }

    public OpResult ExecuteCPU(ComputationalGraph graph, ExecutionOptions? options)
    {
        var opt = options ?? graph.Options;
        if (!OperatorSchemas.TryResolve(this, out _, out var rejection))
        {
            return Failure(Op, rejection ?? "The operator " + DescribeOperator() + " is not supported by the backend.");
        }
        if (Inputs is not null && Op != OpType.SequenceAt && Op != OpType.Identity)
        {
            for (int i = 0; i < Inputs.Length; i++)
            {
                if (InputTensor(graph, i) is TensorSequence seq)
                {
                    return WrongInputType(Op, Inputs[i], "Input must not be a sequence.", seq);
                }
            }
        }
        return Op switch
    {
        OpType.Reshape => CPU.Reshape(InputTensor(graph, 0), InputTensor(graph, 1), GetReshapeAllowZero(), opt),

        OpType.Identity => CPU.Identity(InputTensor(graph, 0), opt),

        OpType.Add => Sub32Gate(graph, OpType.Add) ?? CPU.Add(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Sub => Sub32Gate(graph, OpType.Sub) ?? CPU.Sub(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Mul => Sub32Gate(graph, OpType.Mul) ?? MulSigmoidGate(graph, opt) ?? CPU.Mul(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Div => Sub32Gate(graph, OpType.Div) ?? CPU.Div(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Pow => CPU.Pow(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Sqrt => CPU.Sqrt(InputTensor(graph, 0), opt),

        OpType.Conv => RegionGate(graph, opt) ?? CPU.Conv(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2),
            Attr<string>("auto_pad", null), Ints("dilations"), GetInt("group", null), Ints("kernel_shape"), Ints("pads"), Ints("strides"), opt, GetInt("fuse_relu", null) == 1),

        OpType.Relu => ReluIntGate(graph) ?? CPU.Relu(InputTensor(graph, 0), opt),

        OpType.Erf => CPU.Erf(InputTensor(graph, 0), opt, graph.ActivePool),

        OpType.MaxPool => CPU.MaxPool(InputTensor(graph, 0), Attr<string>("auto_pad", null), GetInt("ceil_mode", null), Ints("dilations"), Ints("kernel_shape"), Ints("pads"), GetInt("storage_order", null), Ints("strides"), opt),

        OpType.GlobalAveragePool => CPU.GlobalAveragePool(InputTensor(graph, 0), opt),

        OpType.MatMul => MatMulScaleGate(graph, opt) ?? MatMulIntGate(graph) ?? CPU.MatMul(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Gemm => CPU.Gemm(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), GetFloat("alpha", 1f) ?? 1f, GetFloat("beta", 1f) ?? 1f, opt, GetInt("transA", 0) ?? 0, GetInt("transB", 0) ?? 0),

        OpType.Transpose => TransposePrepared(graph, opt),

        OpType.Constant => CPU.Constant(OneOfAttr("sparse_value", "value", "value_float", "value_floats", "value_int", "value_ints", "value_string", "value_strings"), opt),

        OpType.ConstantOfShape => CPU.ConstantOfShape(InputTensor(graph, 0), OneOfAttr("value") as ITensor, opt),

        OpType.Cast => CastChecked(InputTensor(graph, 0), RequiredInt("to"), GetInt("saturate", null), opt),

        OpType.Concat => CPU.Concat(graph.GetInputTensors(Inputs), RequiredInt("axis"), opt),

        OpType.Shape => CPU.Shape(InputTensor(graph, 0), Int("start", null), Int("end", null), opt),

        OpType.Gather => CPU.Gather(InputTensor(graph, 0), InputTensor(graph, 1), Int("axis", null), opt),

        OpType.Slice => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 10 => CPU.Slice(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), InputTensor(graph, 3), InputTensor(graph, 4), opt),
            _ => CPU.Slice(InputTensor(graph, 0), RequiredInts("starts").ToTensor<int>(), RequiredInts("ends").ToTensor<int>(), Ints("axes")?.ToTensor<int>(), null, opt),
        },

        OpType.Split => CPU.Split(InputTensor(graph, 0), InputTensor(graph, 1), Int("axis", null), Ints("split"), Int("num_outputs", null), opt, Outputs.Length),

        OpType.Pad => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 11 => CPU.Pad(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), Attr<string>("mode", null), null, OneOfAttr("value") as ITensor, opt),
            _ => CPU.Pad(InputTensor(graph, 0), null, null, Attr<string>("mode", null), Ints("pads"), PadConstantAttr(graph), opt),
        },

        OpType.Clip => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 11 => CPU.Clip(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), null, null, opt),
            _ => CPU.Clip(InputTensor(graph, 0), null, null, GetFloat("min", null), GetFloat("max", null), opt),
        },

        OpType.Equal => CPU.Equal(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Less => CPU.Less(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Where => CPU.Where(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), opt),

        OpType.Expand => CPU.Expand(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Resize => ResolvedOpsetVersion(graph) switch
        {
            // Resize-10 carries scales as the second input (no roi/sizes
            // slots) and samples asymmetrically (verified against ORT 1.29:
            // v10 linear/cubic match v11 asymmetric bit-identically);
            // 11+ uses roi/scales/sizes positions.
            int v when v >= 11 => CPU.Resize(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), InputTensor(graph, 3),
                Attr<string>("mode", "nearest"), Attr<string>("coordinate_transformation_mode", "half_pixel"), Attr<string>("nearest_mode", "round_prefer_floor"),
                GetFloat("cubic_coeff_a", -0.75f), GetFloat("extrapolation_value", 0f), opt,
                GetInt("antialias", null), Ints("axes"), GetInt("exclude_outside", null), Attr<string>("keep_aspect_ratio_policy", null)),
            _ => CPU.Resize(InputTensor(graph, 0), null, InputTensor(graph, 1), null,
                Attr<string>("mode", "nearest"), "asymmetric", Attr<string>("nearest_mode", "round_prefer_floor"),
                GetFloat("cubic_coeff_a", -0.75f), GetFloat("extrapolation_value", 0f), opt,
                GetInt("antialias", null), Ints("axes"), GetInt("exclude_outside", null), Attr<string>("keep_aspect_ratio_policy", null)),
        },

        OpType.Unsqueeze => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 13 => CPU.Unsqueeze(InputTensor(graph, 0), InputTensor(graph, 1), opt),
            _ => CPU.Unsqueeze(InputTensor(graph, 0), RequiredInts("axes"), opt),
        }, 

        OpType.ReduceSum => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 13 => CPU.ReduceSum(InputTensor(graph, 0), InputTensor(graph, 1), Int("keepdims", null), Int("noop_with_empty_axes", null), opt),
            _ => CPU.ReduceSum(InputTensor(graph, 0), Ints("axes")?.ToTensor<int>(), Int("keepdims", null), Int("noop_with_empty_axes", null), opt),
        },

        OpType.ReduceMean => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 18 => CPU.ReduceMean(InputTensor(graph, 0), InputTensor(graph, 1), Int("keepdims", null), Int("noop_with_empty_axes", null), opt),
            _ => CPU.ReduceMean(InputTensor(graph, 0), Ints("axes")?.ToTensor<int>(), Int("keepdims", null), Int("noop_with_empty_axes", null), opt),
        },
        
        OpType.ReduceMax => ResolvedOpsetVersion(graph) switch { int v when v >= 18 => CPU.ReduceMax(InputTensor(graph, 0), InputTensor(graph, 1), Int("keepdims", null), Int("noop_with_empty_axes", null), opt), _ => CPU.ReduceMax(InputTensor(graph, 0), Ints("axes")?.ToTensor<int>(), Int("keepdims", null), Int("noop_with_empty_axes", null), opt), },

        OpType.Softmax => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 13 => CPU.Softmax(InputTensor(graph, 0), Int("axis", null) ?? -1, opt, graph.ActivePool, v),
            int v => CPU.Softmax(InputTensor(graph, 0), Int("axis", null) ?? 1, opt, graph.ActivePool, v),
        },

        OpType.Abs => CPU.Abs(InputTensor(graph, 0), opt),

        OpType.Cos => CPU.Cos(InputTensor(graph, 0), opt),

        OpType.Sin => CPU.Sin(InputTensor(graph, 0), opt),

        OpType.Tanh => CPU.Tanh(InputTensor(graph, 0), opt),

        OpType.Sigmoid => CPU.Sigmoid(InputTensor(graph, 0), opt),

        OpType.Floor => CPU.Floor(InputTensor(graph, 0), opt),

        OpType.And => CPU.And(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Not => CPU.Not(InputTensor(graph, 0), opt),

        OpType.Neg => CPU.Neg(InputTensor(graph, 0), opt),

        OpType.LeakyRelu => CPU.LeakyRelu(InputTensor(graph, 0), GetFloat("alpha", null), opt),

        OpType.LogSoftmax => CPU.LogSoftmax(InputTensor(graph, 0), Int("axis", null), opt, graph.ActivePool, ResolvedOpsetVersion(graph)),

        OpType.Gelu => CPU.Gelu(InputTensor(graph, 0), Attr<string>("approximate", null), opt, graph.ActivePool),

        OpType.Squeeze => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 13 => CPU.Squeeze(InputTensor(graph, 0), InputTensor(graph, 1), opt),
            _ => CPU.Squeeze(InputTensor(graph, 0), Ints("axes")?.ToTensor<int>(), opt),
        },

        OpType.Range => CPU.Range(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), opt),

        OpType.Tile => CPU.Tile(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.LayerNormalization => CPU.LayerNormalization(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), Int("axis", null), GetFloat("epsilon", null), Int("stash_type", null), Outputs.Length, opt, graph.ActivePool),

        OpType.InstanceNormalization => CPU.InstanceNorm(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), GetFloat("epsilon", null), opt),

        OpType.LSTM => CPU.Lstm(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), InputTensor(graph, 3), InputTensor(graph, 4), InputTensor(graph, 5), InputTensor(graph, 6), InputTensor(graph, 7),
            Attr<string>("direction", null), Attr<string[]>("activations", null), Attr<float[]>("activation_alpha", null), Attr<float[]>("activation_beta", null),
            GetFloat("clip", null), RequiredInt("hidden_size"), (GetInt("input_forget", 0) ?? 0) == 1, GetInt("layout", 0) ?? 0, Outputs.Length, opt, graph.ActivePool),

        OpType.SplitToSequence => CPU.SplitToSequence(InputTensor(graph, 0), InputTensor(graph, 1), Int("axis", null), Int("keepdims", null), opt),

        OpType.SequenceAt => CPU.SequenceAt(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.RotaryEmbedding => CPU.RotaryEmbedding(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), RequiredInt("half"), Int("axis", null), Int("concatAxis", null), opt, graph.ActivePool),

        OpType.If => ExecuteIf(graph, opt),

        _ => NotSupported(Op)
    };
    }
}

