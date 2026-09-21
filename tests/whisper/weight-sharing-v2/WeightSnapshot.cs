using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;

// Diagnostic policy only: cached transpose output names change during first use.
// Every other field, including every weight byte digest, must remain exact.
internal static class WeightSnapshot
{
    internal static void Validate(object before, object after)
    {
        var expected = JsonSerializer.SerializeToNode(before)!;
        foreach (var (graph, key) in new[] { ("first", "folded:Transpose_1010"), ("past", "folded:Transpose_801") })
        {
            const string output = "model.decoder.embed_tokens.weight_transposed";
            var initializer = expected[graph]!["initializers"]!.AsArray()
                .Single(r => r!["name"]!.GetValue<string>() == key)!;
            if (initializer["tensor_name"]!.GetValue<string>() != key)
                throw new InvalidDataException("Unexpected initial cached transpose name");
            var node = expected[graph]!["nodes"]!.AsArray()
                .Single(n => n!["Name"]!.GetValue<string>() == key["folded:".Length..])!;
            if (node["op"]!.GetValue<string>() != "Transpose" || node["Outputs"]!.AsArray().Count != 1
                || node["Outputs"]![0]!.GetValue<string>() != output)
                throw new InvalidDataException("Unexpected cached transpose binding");
            initializer["tensor_name"] = output;
        }
        if (!JsonNode.DeepEquals(expected, JsonSerializer.SerializeToNode(after)))
            throw new InvalidDataException("Decoder values, structure or storage changed beyond the two prepared output names");
    }
}
