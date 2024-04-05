namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using FastBertTokenizer;

public class Text : Runtime
{
    public static (ITensor[]?, Func<long[], string>?) BertTokenize(string text, string tokenizer)
    {
        var op = Begin("Tokenizing {len} characters using BERT tokenizer {tok}", text.Length, tokenizer);
        var tok = new BertTokenizer();
          
        tok.LoadFromHuggingFaceAsync(tokenizer).Wait();
        var (inputIds, attentionMask, tokenTypeIds) = tok.Encode(text, 512, 512);
        op.Complete();
        return (new ITensor[3] {
            DenseTensor<long>.OfValues(inputIds.ToArray()).PadLeft().WithName("input_ids"),
            DenseTensor<long>.OfValues(attentionMask.ToArray()).PadLeft().WithName("attention_mask"),
            DenseTensor<long>.OfValues(tokenTypeIds.ToArray()).PadLeft().WithName("token_type_ids"),
        }, (t) => tok.Decode(t));
    }

    public static (ITensor[]?, Func<long[], string>?) GetTextTensors(string text, string props)
    {
        var tprops = props.Split(':');
        if (tprops.Length == 0 || tprops[0] == "me5s")
        {
            return BertTokenize(text, "bert-base-uncased");
        }
        else
        {
            Error("Could not interpret text using properties {p}.", props);
            return (null, null);
        }
    }

    public static (ITensor[]?, Func<long[], string>?) GetTextTensorsFromFileArg(string name, string[] p)
    {
        if (!File.Exists(name)) 
        {
            Error("File {name} does not exist.", name); 
            return (null, null);
        }
        else
        {
            return GetTextTensors(File.ReadAllText(name), p.Length > 1 ? p[1] : "");
        }
    }

    public static string[] TextExtensions = new string[] { ".txt" };
}

