namespace Lokad.Onnx.CLI;

using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

using CommandLine;

#region Base classes
public class Options
{
    [Option("debug", Required = false, HelpText = "Enable debug mode.")]
    public bool Debug { get; set; }

    [Option("options", Required = false, HelpText = "Any additional options for the selected operation.")]
    public string AdditionalOptions { get; set; } = String.Empty;

    [Value(0, Required = true, HelpText = "The ONNX model file to open.")]
    public string File { get; set; } = String.Empty;

    public static Dictionary<string, object> Parse(string o)
    {
        Dictionary<string, object> options = new Dictionary<string, object>();
        Regex re = new Regex(@"(\w+)\=([^\,]+)", RegexOptions.Compiled);
        string[] pairs = o.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (string s in pairs)
        {
            Match m = re.Match(s);
            if (!m.Success)
            {
                options.Add("_ERROR_", s);
            }
            else if (options.ContainsKey(m.Groups[1].Value))
            {
                options[m.Groups[1].Value] = m.Groups[2].Value;
            }
            else
            {
                options.Add(m.Groups[1].Value, m.Groups[2].Value);
            }
        }
        return options;
    }
}
#endregion

[Verb("info", HelpText = "Get information on an ONNX model.")]
public class InfoOptions : Options
{
    [Option("ops", Required = false, HelpText = "Only print out a list of distinct ops present in the model.")]
    public bool Ops { get; set; }

    [Option("init", Required = false, HelpText = "Only print out a list of initializers present in the model.")]
    public bool Initializers { get; set; }

    [Option("filter-op", Required = false, HelpText = "Filter on ops with this type.")]
    public string? FilterOp { get; set; }
}