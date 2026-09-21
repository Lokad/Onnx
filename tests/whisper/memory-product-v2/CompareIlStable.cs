using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;

if (args.Length != 3) throw new ArgumentException("old bin, corrected bin, new result");
var opcodes = typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
    .Where(f => f.FieldType == typeof(OpCode)).Select(f => (OpCode)(f.GetValue(null) ?? throw new InvalidDataException()))
    .ToDictionary(o => unchecked((ushort)o.Value));
static string Hash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
static string Member(MemberInfo member) => (member.DeclaringType?.ToString() ?? "") + "::" + member;

Dictionary<string, string> Inspect(Assembly assembly)
{
    var results = new Dictionary<string, string>();
    const BindingFlags all = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;
    foreach (var type in assembly.GetTypes())
    foreach (var method in type.GetMethods(all).Cast<MethodBase>().Concat(type.GetConstructors(all)))
    {
        string key = type + "::" + method.Name + "::" + method;
        var body = method.GetMethodBody();
        if (body is null) { results.Add(key, "NO-BODY"); continue; }
        var code = body.GetILAsByteArray() ?? throw new InvalidDataException("Missing IL");
        var instructions = new List<object>(); int offset = 0;
        while (offset < code.Length)
        {
            int start = offset; ushort value = code[offset++];
            if (value == 0xfe) value = (ushort)(0xfe00 | code[offset++]);
            var op = opcodes[value]; object operand;
            switch (op.OperandType)
            {
                case OperandType.InlineNone: operand = ""; break;
                case OperandType.InlineString:
                    operand = method.Module.ResolveString(BitConverter.ToInt32(code, offset)); offset += 4; break;
                case OperandType.InlineField:
                case OperandType.InlineMethod:
                case OperandType.InlineType:
                case OperandType.InlineTok:
                    int token = BitConverter.ToInt32(code, offset); offset += 4;
                    var member = method.Module.ResolveMember(token, type.GetGenericArguments(), method.IsGenericMethod ? method.GetGenericArguments() : null)
                        ?? throw new InvalidDataException("Unresolved member");
                    operand = Member(member); break;
                case OperandType.InlineSig:
                    operand = Convert.ToHexString(method.Module.ResolveSignature(BitConverter.ToInt32(code, offset))); offset += 4; break;
                case OperandType.InlineSwitch:
                    int count = BitConverter.ToInt32(code, offset); offset += 4;
                    operand = Convert.ToHexString(code.AsSpan(offset, checked(count * 4))); offset += count * 4; break;
                default:
                    int size = op.OperandType switch
                    {
                        OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar => 1,
                        OperandType.InlineVar => 2,
                        OperandType.InlineBrTarget or OperandType.InlineI or OperandType.ShortInlineR => 4,
                        OperandType.InlineI8 or OperandType.InlineR => 8,
                        _ => throw new InvalidDataException("Unsupported operand " + op.OperandType)
                    };
                    operand = Convert.ToHexString(code.AsSpan(offset, size)); offset += size; break;
            }
            instructions.Add(new { offset = start, opcode = op.Name, operand });
        }
        if (offset != code.Length) throw new InvalidDataException("IL boundary");
        var exceptions = body.ExceptionHandlingClauses.Select(c => new {
            flags = (int)c.Flags, c.TryOffset, c.TryLength, c.HandlerOffset, c.HandlerLength,
            filter = c.Flags == ExceptionHandlingClauseOptions.Filter ? c.FilterOffset : -1,
            caught = c.Flags == ExceptionHandlingClauseOptions.Clause ? c.CatchType?.ToString() : null });
        results.Add(key, JsonSerializer.Serialize(new { body.InitLocals, body.MaxStackSize,
            locals = body.LocalVariables.Select(v => new { type = v.LocalType.ToString(), v.IsPinned }), exceptions, instructions }));
    }
    return results;
}

Assembly Load(string directory, string name, string contextName)
{
    var context = new AssemblyLoadContext(contextName, isCollectible: false);
    context.Resolving += (owner, reference) => {
        string path = Path.Combine(directory, reference.Name + ".dll");
        return File.Exists(path) ? owner.LoadFromAssemblyPath(path) : null;
    };
    return context.LoadFromAssemblyPath(Path.Combine(directory, name));
}

var observations = new List<object>();
foreach (string name in new[] { "Lokad.Onnx.dll", "Lokad.Onnx.Data.dll" })
{
    string before = Path.GetFullPath(args[0]), after = Path.GetFullPath(args[1]);
    var oldAssembly = Load(before, name, "old-" + name);
    var newAssembly = Load(after, name, "new-" + name);
    var oldMethods = Inspect(oldAssembly); var newMethods = Inspect(newAssembly);
    if (oldMethods.Count == 0 || !oldMethods.Keys.Order().SequenceEqual(newMethods.Keys.Order())) throw new InvalidDataException("Method coverage differs");
    var differences = oldMethods.Keys.Where(k => oldMethods[k] != newMethods[k]).ToArray();
    if (differences.Length != 0) throw new InvalidDataException("Runtime IL differs: " + string.Join("; ", differences));
    bool? correctedAnnotation = null;
    if (name == "Lokad.Onnx.Data.dll")
    {
        var type = newAssembly.GetType("Lokad.Onnx.WhisperDecoderWeights") ?? throw new InvalidDataException("Missing helper");
        var method = type.GetMethod("Eligible", BindingFlags.Static | BindingFlags.NonPublic) ?? throw new InvalidDataException("Missing eligibility method");
        var annotation = method.GetParameters().Last().GetCustomAttributesData().Single(a => a.AttributeType.FullName == "System.Diagnostics.CodeAnalysis.NotNullWhenAttribute");
        correctedAnnotation = annotation.ConstructorArguments.Count == 1 && Equals(annotation.ConstructorArguments[0].Value, true);
        if (correctedAnnotation != true) throw new InvalidDataException("Incorrect nullable flow annotation");
    }
    observations.Add(new { assembly = name, methods = oldMethods.Count, before_sha256 = Hash(Path.Combine(before, name)),
        after_sha256 = Hash(Path.Combine(after, name)), corrected_nullable_annotation = correctedAnnotation,
        normalized_methods = oldMethods, equal = true });
}
using (var output = new FileStream(args[2], FileMode.CreateNew))
    JsonSerializer.Serialize(output, new { passed = true, scope = "All Core/Data method IL, resolved operands, locals, stack and exception clauses", observations }, new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine(JsonSerializer.Serialize(new { passed = true, assemblies = 2, result = args[2] }));
