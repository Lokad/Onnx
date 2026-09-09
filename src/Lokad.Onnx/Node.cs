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


    public ITensor? InputTensor(ComputationalGraph graph, int index) =>
        index < Inputs.Length && !string.IsNullOrEmpty(Inputs[index]) ? graph.GetInputTensor(Inputs[index]) : null;

    /// <summary>
    /// Transpose with a prepared-plan shortcut: constant-only transpositions
    /// are computed once, stored as plan-owned initializers, and reused while
    /// the source initializer is unchanged. Anything else computes normally.
    /// </summary>
    OpResult TransposePrepared(ComputationalGraph graph, ExecutionOptions? opt)
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

        static OpResult CastChecked(ITensor? input, int to, ExecutionOptions? opt)
    {
        var op = OpType.Cast;
        if (!Enum.IsDefined(typeof(TensorElementType), to)) return AttributeNotSupported(op, "to", to.ToString(), null);
        return CPU.Cast(input, (TensorElementType)to, opt);
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

        OpType.Add => CPU.Add(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Sub => CPU.Sub(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Mul => CPU.Mul(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Div => CPU.Div(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Pow => CPU.Pow(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Sqrt => CPU.Sqrt(InputTensor(graph, 0), opt),

        OpType.Conv => CPU.Conv(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2),
            Attr<string>("auto_pad", null), Ints("dilations"), GetInt("group", null), Ints("kernel_shape"), Ints("pads"), Ints("strides"), opt),

        OpType.Relu => CPU.Relu(InputTensor(graph, 0), opt),

        OpType.Erf => CPU.Erf(InputTensor(graph, 0), opt, graph.ActivePool),

        OpType.MaxPool => CPU.MaxPool(InputTensor(graph, 0), Attr<string>("auto_pad", null), GetInt("ceil_mode", null), Ints("dilations"), Ints("kernel_shape"), Ints("pads"), GetInt("storage_order", null), Ints("strides"), opt),

        OpType.GlobalAveragePool => CPU.GlobalAveragePool(InputTensor(graph, 0), opt),

        OpType.MatMul => CPU.MatMul(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Gemm => CPU.Gemm(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), GetFloat("alpha", 1f) ?? 1f, GetFloat("beta", 1f) ?? 1f, opt, GetInt("transA", 0) ?? 0, GetInt("transB", 0) ?? 0),

        OpType.Transpose => TransposePrepared(graph, opt),

        OpType.Constant => CPU.Constant(OneOfAttr("sparse_value", "value", "value_float", "value_floats", "value_int", "value_ints", "value_string", "value_strings"), opt),

        OpType.ConstantOfShape => CPU.ConstantOfShape(InputTensor(graph, 0), OneOfAttr("value") as ITensor, opt),

        OpType.Cast => CastChecked(InputTensor(graph, 0), RequiredInt("to"), opt),

        OpType.Concat => CPU.Concat(graph.GetInputTensors(Inputs), RequiredInt("axis"), opt),

        OpType.Shape => CPU.Shape(InputTensor(graph, 0), Int("start", null), Int("end", null), opt),

        OpType.Gather => CPU.Gather(InputTensor(graph, 0), InputTensor(graph, 1), Int("axis", null), opt),

        OpType.Slice => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 10 => CPU.Slice(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), InputTensor(graph, 3), InputTensor(graph, 4), opt),
            _ => CPU.Slice(InputTensor(graph, 0), RequiredInts("starts").ToTensor<int>(), RequiredInts("ends").ToTensor<int>(), Ints("axes")?.ToTensor<int>(), null, opt),
        },

        OpType.Split => CPU.Split(InputTensor(graph, 0), InputTensor(graph, 1), Int("axis", null), Ints("split"), Int("num_outputs", null), opt, Outputs.Length),

        OpType.Equal => CPU.Equal(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Less => CPU.Less(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Where => CPU.Where(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), opt),

        OpType.Expand => CPU.Expand(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Resize => CPU.Resize(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), InputTensor(graph, 3),
            Attr<string>("mode", "nearest"), Attr<string>("coordinate_transformation_mode", "half_pixel"), Attr<string>("nearest_mode", "round_prefer_floor"),
            GetFloat("cubic_coeff_a", -0.75f), GetFloat("extrapolation_value", 0f), opt,
            GetInt("antialias", null), Ints("axes"), GetInt("exclude_outside", null), Attr<string>("keep_aspect_ratio_policy", null)),

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
        
        OpType.ReduceMax => ResolvedOpsetVersion(graph) switch { int v when v >= 18 => CPU.ReduceMax(InputTensor(graph, 0), InputTensor(graph, 1), Int("keepdims", null), null, opt), _ => CPU.ReduceMax(InputTensor(graph, 0), Ints("axes")?.ToTensor<int>(), Int("keepdims", null), null, opt), },

        OpType.Softmax => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 13 => CPU.Softmax(InputTensor(graph, 0), Int("axis", null) ?? -1, opt, graph.ActivePool, v),
            int v => CPU.Softmax(InputTensor(graph, 0), Int("axis", null) ?? 1, opt, graph.ActivePool, v),
        },

        OpType.Abs => CPU.Abs(InputTensor(graph, 0), opt),

        OpType.Cos => CPU.Cos(InputTensor(graph, 0), opt),

        OpType.Sin => CPU.Sin(InputTensor(graph, 0), opt),

        OpType.Tanh => CPU.Tanh(InputTensor(graph, 0), opt),

        OpType.Neg => CPU.Neg(InputTensor(graph, 0), opt),

        OpType.Gelu => CPU.Gelu(InputTensor(graph, 0), Attr<string>("approximate", null), opt, graph.ActivePool),

        OpType.Squeeze => ResolvedOpsetVersion(graph) switch
        {
            int v when v >= 13 => CPU.Squeeze(InputTensor(graph, 0), InputTensor(graph, 1), opt),
            _ => CPU.Squeeze(InputTensor(graph, 0), Ints("axes")?.ToTensor<int>(), opt),
        },

        OpType.Range => CPU.Range(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), opt),

        OpType.Tile => CPU.Tile(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.LayerNormalization => CPU.LayerNormalization(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), Int("axis", null), GetFloat("epsilon", null), Int("stash_type", null), Outputs.Length, opt, graph.ActivePool),

        OpType.SplitToSequence => CPU.SplitToSequence(InputTensor(graph, 0), InputTensor(graph, 1), Int("axis", null), Int("keepdims", null), opt),

        OpType.SequenceAt => CPU.SequenceAt(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.RotaryEmbedding => CPU.RotaryEmbedding(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), RequiredInt("half"), Int("axis", null), Int("concatAxis", null), opt, graph.ActivePool),

        _ => NotSupported(Op)
    };
    }
}

