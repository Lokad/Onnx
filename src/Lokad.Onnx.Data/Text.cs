namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Lokad.Tokenizers.Tokenizer;
using FastBertTokenizer;

public class Text : Runtime
{
    public static ITensor[]? BertTokenize(string text, string tokenizer)
    {
        var op = Begin("Tokenizing {len} characters using BERT tokenizer {tok}", text.Length, tokenizer);
        var tok = new BertTokenizer();
          
        tok.LoadFromHuggingFaceAsync(tokenizer).Wait();
        var (inputIds, attentionMask, tokenTypeIds) = tok.Encode(text, 512);
        op.Complete();
        return new ITensor[3] {
            DenseTensor<long>.OfValues(inputIds.ToArray()).PadLeft().WithName("input_ids"),
            DenseTensor<long>.OfValues(attentionMask.ToArray()).PadLeft().WithName("attention_mask"),
            DenseTensor<long>.OfValues(tokenTypeIds.ToArray()).PadLeft().WithName("token_type_ids"),
        };
    }

    public static ITensor[]? RobertaTokenize(string text1, string tokenizer)
    {
        XLMRobertaTokenizer? tok = null;
        switch (tokenizer)
        {
            case "me5s":
                Info("Using multilingual-e5-small tokenizer.");
                var tokenizerPath = Path.Combine(Runtime.AssemblyLocation, "me5s-sentencepiece.bpe.model");
                if (!File.Exists(tokenizerPath))
                {
                    if (!DownloadFile(
                        "sentencepiece.bpe.model",
                        new Uri("https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/sentencepiece.bpe.model"),
                        tokenizerPath))
                    {
                        Error("Could not download model file.");
                        return null;    
                    }
                }
                tok = new XLMRobertaTokenizer(tokenizerPath, false);
                break;
            default:
                Error("Unknown Roberta tokenizer: {t}.", tokenizer);
                return null;
        }
        var t = tok!.Encode(tok, text1, null, 512, TruncationStrategy.OnlyFirst, 0);
        if (t is null) 
        {
            Error("Could not encode text.");
            return null;
        }
        else
        {
            return new ITensor[3]
            {
                 DenseTensor<long>.OfValues(t.TokenIds.ToArray()).PadLeft().WithName("input_ids"),
                 DenseTensor<long>.Ones(1, t.TokenIds.Count).WithName("attention_mask"),
                 DenseTensor<long>.Zeros(1, t.TokenIds.Count).WithName("token_type_ids"),
            };
        }
        
    }
    public static ITensor[]? GetTextTensors(string text, string props)
    {
        var tprops = props.Split(':');
        if (tprops.Length == 0 || tprops[0] == "me5s")
        {
            return RobertaTokenize(text, "me5s");
        }
        else if (tprops[0] == "bert")
        {
            return BertTokenize(text, "bert-base-uncased");
        }
        else
        {
            Error("Could not tokenize text using properties {p}.", props);
            return null;
        }
    }

    public static ITensor[]? GetTextTensorsFromFileArg(string name, string[] p)
    {
        if (!File.Exists(name)) 
        {
            Error("File {name} does not exist.", name); 
            return null;
        }
        else
        {
            return GetTextTensors(File.ReadAllText(name), p.Length > 1 ? p[1] : "");
        }
    }

    public static string[] TextExtensions = new string[] { ".txt" };
}

