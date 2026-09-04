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

    public static XLMRobertaTokenizer LoadRobertaTokenizerFromFile(string tokenizerModelPath)
    {
        if (!File.Exists(tokenizerModelPath))
        {
            throw new FileNotFoundException($"Tokenizer model file does not exist: {tokenizerModelPath}.", tokenizerModelPath);
        }
        return new XLMRobertaTokenizer(tokenizerModelPath, false);
    }

    static string NormalizeRobertaText(string text) => System.Text.RegularExpressions.Regex.Replace(text, "  +", " ");

    static ITensor[]? EncodeSingleRoberta(XLMRobertaTokenizer tok, string text, string tokDesc)
    {
        var op = Begin("Tokenizing text of length {l} chars using {tok_desc} tokenizer", text.Length, tokDesc);
        var t = tok.Encode(NormalizeRobertaText(text), null, 512, TruncationStrategy.OnlyFirst, 0);
        if (t is null)
        {
            op.Abandon();
            return null;
        }
        else
        {
            op.Complete();
            return new ITensor[3]
            {
                 DenseTensor<long>.OfValues(t.TokenIds.ToArray()).PadLeft().WithName("input_ids"),
                 DenseTensor<long>.Ones(1, t.Mask.Count).WithName("attention_mask"),
                 DenseTensor<long>.Zeros(1, t.TokenIds.Count).WithName("token_type_ids"),
            };
        }
    }

    static ITensor[] EncodeBatchRoberta(XLMRobertaTokenizer tok, string[] texts, string tokDesc)
    {
        var op = Begin("Tokenizing text array of length {l} using {tok_desc} tokenizer", texts.Length, tokDesc);
        var results = texts.Select(text1 =>
        {
            var t = tok.Encode(NormalizeRobertaText(text1), null, 512, TruncationStrategy.OnlyFirst, 0);
            if (t is null)
            {
                op.Abandon();
                throw new Exception("Error tokenizing text " + text1 + ". Stopping.");
            }
            else
            {
                return new ITensor[3]
                {
                    DenseTensor<long>.OfValues(t.TokenIds.ToArray()).WithName("input_ids"),
                    DenseTensor<long>.Ones(1, t.TokenIds.Count).WithName("attention_mask"),
                    DenseTensor<long>.Zeros(1, t.TokenIds.Count).WithName("token_type_ids"),
                };
            }
        });
        var maxl = results.Select(r => r[0].Length).Max();
        var inputids = new List<long[]>();
        var attentionMask = new List<long[]>();
        var typeids = new List<long[]>();
        foreach (var r in results)
        {
            var length = r[0].Length;
            var padl = maxl - length;
            var padding = new long[padl];
            Array.Fill(padding, 1L);
            inputids.Add(r[0].AsTensor<long>().Concat(padding).ToArray());
            attentionMask.Add(r[1].AsTensor<long>().Concat(new long[padl]).ToArray());
            typeids.Add(r[2].AsTensor<long>().Concat(new long[padl]).ToArray());
        }
        op.Complete();
        return new ITensor[] {
            inputids.ToArray().To2DArray<long>().ToTensor<long>().WithName("input_ids"),
            attentionMask.ToArray().To2DArray<long>().ToTensor<long>().WithName("attention_mask"),
            typeids.ToArray().To2DArray<long>().ToTensor<long>().WithName("token_type_ids")
        };
    }

    public static ITensor[]? RobertaTokenize(string text1, string tokenizer)
    {
        switch (tokenizer)
        {
            case "me5s":
                if (!Tokenizers.ContainsKey("me5s"))
                {
                    var tokenizerPath = Path.Combine(AssemblyLocation, "me5s-sentencepiece.bpe.model");
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
                    Tokenizers["me5s"] = new XLMRobertaTokenizer(tokenizerPath, false);
                }
                return EncodeSingleRoberta((XLMRobertaTokenizer)Tokenizers["me5s"], text1, "multilingual-e5-small");
            default:
                Error("Unknown Roberta tokenizer: {t}.", tokenizer);
                return null;
        }
    }

    public static ITensor[]? RobertaTokenize(string[] text, string tokenizer)
    {
        switch (tokenizer)
        {
            case "me5s":
                if (!Tokenizers.ContainsKey("me5s"))
                {
                    var tokenizerPath = Path.Combine(AssemblyLocation, "me5s-sentencepiece.bpe.model");
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
                    Tokenizers["me5s"] = new XLMRobertaTokenizer(tokenizerPath, false);
                }
                return EncodeBatchRoberta((XLMRobertaTokenizer)Tokenizers["me5s"], text, "multilingual-e5-small");
            default:
                Error("Unknown Roberta tokenizer: {t}.", tokenizer);
                return null;
        }
    }

    public static ITensor[]? RobertaTokenizeFromFile(string text, string tokenizerModelPath)
    {
        return EncodeSingleRoberta(LoadRobertaTokenizerFromFile(tokenizerModelPath), text, "multilingual-e5-small");
    }

    public static ITensor[]? RobertaTokenizeFromFile(IReadOnlyList<string> texts, string tokenizerModelPath)
    {
        return EncodeBatchRoberta(LoadRobertaTokenizerFromFile(tokenizerModelPath), texts.ToArray(), "multilingual-e5-small");
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

    public static ITensor[]? GetTextTensors(string[] text, string props) 
    {
        var tprops = props.Split(':');
        if (tprops.Length == 0 || tprops[0] == "me5s")
        {
            return RobertaTokenize(text, "me5s");
        }
        //else if (tprops[0] == "bert")
        //{
        //    return BertTokenize(text, "bert-base-uncased");
        //}
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
    public static Dictionary<string, object> Tokenizers = new Dictionary<string, object>();
}

