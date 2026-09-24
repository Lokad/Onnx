using System.Reflection;
using System.Reflection.Emit;
using System.Text.Json;

static partial class Program
{
    static int Inspect(string destination)
    {
        var method = typeof(Program).GetMethod("MeasureBatch", BindingFlags.Static | BindingFlags.NonPublic)!;
        var body = method.GetMethodBody()!;
        byte[] il = body.GetILAsByteArray()!;
        var opcodes = typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
            .Where(f => f.FieldType == typeof(OpCode)).Select(f => (OpCode)f.GetValue(null)!)
            .ToDictionary(o => unchecked((ushort)o.Value));
        var rows = new List<object>();
        int position = 0;
        while (position < il.Length)
        {
            int offset = position;
            ushort key = il[position++];
            if (key == 0xfe) key = (ushort)(0xfe00 | il[position++]);
            OpCode opcode = opcodes[key];
            object? operand = null;
            switch (opcode.OperandType)
            {
                case OperandType.InlineNone: break;
                case OperandType.ShortInlineI: operand = (sbyte)il[position++]; break;
                case OperandType.ShortInlineVar: operand = il[position++]; break;
                case OperandType.InlineVar: operand = BitConverter.ToUInt16(il, position); position += 2; break;
                case OperandType.InlineI: operand = BitConverter.ToInt32(il, position); position += 4; break;
                case OperandType.InlineI8: operand = BitConverter.ToInt64(il, position); position += 8; break;
                case OperandType.ShortInlineBrTarget:
                    int shortDelta = (sbyte)il[position++]; operand = position + shortDelta; break;
                case OperandType.InlineBrTarget:
                    int delta = BitConverter.ToInt32(il, position); position += 4; operand = position + delta; break;
                case OperandType.InlineMethod:
                    var callee = method.Module.ResolveMethod(BitConverter.ToInt32(il, position))!; position += 4;
                    operand = callee.DeclaringType!.FullName + ":" + callee; break;
                case OperandType.InlineType:
                    operand = method.Module.ResolveType(BitConverter.ToInt32(il, position)).FullName; position += 4; break;
                default: throw new InvalidDataException("Unexpected timing-body operand " + opcode.OperandType);
            }
            rows.Add(new { offset, opcode = opcode.Name, operand });
        }
        File.WriteAllText(destination, JsonSerializer.Serialize(new
        {
            passed = true, pid = Environment.ProcessId, runtime = Environment.Version.ToString(),
            assembly = FileHash(Assembly.GetExecutingAssembly().Location),
            method = method.ToString(), implementation_flags = (int)method.GetMethodImplementationFlags(),
            locals = body.LocalVariables.Select(v => v.LocalType.FullName).ToArray(),
            exceptions = body.ExceptionHandlingClauses.Count, il_bytes = il.Length, instructions = rows
        }, Json));
        return 0;
    }
}
