using System;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;

if(args.Length!=4)throw new ArgumentException("Four saved diagnostic results required");
int accepted=0,rejected=0;
foreach(string path in args)
{
    var value=JsonNode.Parse(File.ReadAllText(path))!;
    var before=value["snapshots"]![0]!.DeepClone().AsObject();before.Remove("stage");
    var after=value["snapshots"]![4]!.DeepClone().AsObject();after.Remove("stage");
    WeightSnapshot.Validate(before,after);accepted++;
    Action<JsonNode>[] corruptions={
        n=>n["first"]!["initializers"]![0]!["sha256"]=new string('0',64),
        n=>n["past"]!["initializers"]![0]!["shape"]=new JsonArray(-1),
        n=>n["unique_arrays"]=n["unique_arrays"]!.GetValue<int>()+1,
        n=>n["unique_payload_bytes"]=n["unique_payload_bytes"]!.GetValue<long>()+1,
        n=>n["first"]!["packed_bytes"]=n["first"]!["packed_bytes"]!.GetValue<long>()+1,
        n=>n["past"]!["nodes"]![0]!["op"]="Invalid",
        n=>n["first"]!["initializers"]!.AsArray().Single(r=>r!["name"]!.GetValue<string>()=="folded:Transpose_1010")!["tensor_name"]="unexpected"
    };
    foreach(var corrupt in corruptions)
    {
        var bad=after.DeepClone();corrupt(bad);
        try{WeightSnapshot.Validate(before,bad);}
        catch(InvalidDataException){rejected++;continue;}
        throw new InvalidDataException("Corruption accepted");
    }
}
Console.WriteLine($"Accepted {accepted} real transitions; rejected {rejected} damaged snapshots.");
