namespace Lokad.Onnx.CLI;

using System;
using System.Collections.Generic;

#region Option records
public class Options
{
    public bool Debug { get; set; }
}

public class InfoOptions : Options
{
    public string File { get; set; } = String.Empty;
    public bool Ops { get; set; }
    public bool Initializers { get; set; }
    public string? OpFilter { get; set; }
}

public class RunOptions : Options
{
    public string File { get; set; } = String.Empty;
    public List<string> Inputs { get; set; } = new List<string>();
    public bool SaveInput { get; set; }
    public bool Softmax { get; set; }
    public string Node { get; set; } = "";
    public string Text { get; set; } = "";
    public bool PrintInput { get; set; }
    public bool DisableSimd { get; set; }
    public bool EnableIntrinsics { get; set; }
    public bool EnableProfiler { get; set; }
    public bool OptimizeMemory { get; set; }
    public int Threads { get; set; } = 1;
}

#endregion

#region Bounded parser
public enum ParseOutcome
{
    Ok,
    Help,
    Version,
    Error,
}

public sealed class ParseResult
{
    public ParseOutcome Outcome;
    public string Verb = "";
    public Options? Value;
    public string Message = "";
    public ExitResult Exit = ExitResult.SUCCESS;
}

public static class ArgsParser
{
    static readonly HashSet<string> InfoFlags = new HashSet<string>(StringComparer.Ordinal)
        { "ops", "init", "op-filter", "debug" };
    static readonly HashSet<string> RunFlags = new HashSet<string>(StringComparer.Ordinal)
        { "save-input", "softmax", "node", "text", "print-input", "disable-simd",
          "enable-intrinsics", "profile", "optimize-memory", "threads", "debug" };

    static bool IsFlag(string token) => token.StartsWith("--", StringComparison.Ordinal);
    static bool IsShort(string token) => token.Length == 2 && token[0] == '-' && token[1] != '-';

    public static ParseResult Parse(string[] args)
    {
        var result = new ParseResult();
        bool debug = false;
        bool help = false;
        bool version = false;
        string verb = "";
        var rest = new List<string>();
        foreach (var token in args)
        {
            if (token == "--debug" || token == "-d") { debug = true; continue; }
            if (token == "--help") { help = true; continue; }
            if (token == "--version") { version = true; continue; }
            if (verb.Length == 0 && !IsFlag(token) && !IsShort(token)) { verb = token; continue; }
            rest.Add(token);
        }
        if (version)
        {
            result.Outcome = ParseOutcome.Version;
            return result;
        }
        if (verb.Length == 0)
        {
            if (help)
            {
                result.Outcome = ParseOutcome.Help;
                return result;
            }
            result.Outcome = ParseOutcome.Error;
            result.Message = "No command specified.";
            result.Exit = ExitResult.INVALID_OPTIONS;
            return result;
        }
        if (verb != "info" && verb != "run")
        {
            result.Outcome = ParseOutcome.Error;
            result.Message = "Unknown command: " + verb + ".";
            result.Exit = ExitResult.INVALID_OPTIONS;
            return result;
        }
        if (help)
        {
            result.Outcome = ParseOutcome.Help;
            result.Verb = verb;
            return result;
        }
        result.Verb = verb;
        var parsed = verb == "info" ? ParseInfo(rest) : ParseRun(rest);
        if (parsed.Outcome == ParseOutcome.Ok && parsed.Value is not null) parsed.Value.Debug = debug;
        return parsed;
    }

    static void SplitFlag(string token, out string name, out string? value)
    {
        int eq = token.IndexOf('=');
        if (eq < 0)
        {
            name = token.Substring(2);
            value = null;
        }
        else
        {
            name = token.Substring(2, eq - 2);
            value = token.Substring(eq + 1);
        }
    }

    static ParseResult Fail(string? message)
    {
        return new ParseResult { Outcome = ParseOutcome.Error, Message = message ?? "Invalid arguments.", Exit = ExitResult.INVALID_OPTIONS };
    }

    static bool TakeBool(string verb, string name, string? value, out bool parsed, out string? error)
    {
        if (value is null)
        {
            parsed = true;
            error = null;
            return true;
        }
        if (bool.TryParse(value, out parsed))
        {
            error = null;
            return true;
        }
        error = "Option --" + name + " for command " + verb + " must be true or false.";
        return false;
    }

    static bool TakeString(string verb, string name, List<string> rest, ref int i, string? inline, out string value, out string? error)
    {
        if (inline is not null)
        {
            value = inline;
            error = null;
            return true;
        }
        if (i + 1 < rest.Count && !IsFlag(rest[i + 1]) && !IsShort(rest[i + 1]))
        {
            value = rest[i + 1];
            i++;
            error = null;
            return true;
        }
        value = "";
        error = "Option --" + name + " for command " + verb + " requires a value.";
        return false;
    }

    static bool TakeInt(string verb, string name, List<string> rest, ref int i, string? inline, out int value, out string? error)
    {
        if (!TakeString(verb, name, rest, ref i, inline, out var text, out error))
        {
            value = 0;
            return false;
        }
        if (int.TryParse(text, out value))
        {
            error = null;
            return true;
        }
        error = "Option --" + name + " for command " + verb + " must be an integer.";
        return false;
    }

    static ParseResult ParseInfo(List<string> rest)
    {
        var options = new InfoOptions();
        var positionals = new List<string>();
        for (int i = 0; i < rest.Count; i++)
        {
            var token = rest[i];
            if (IsFlag(token))
            {
                SplitFlag(token, out var name, out var inline);
                if (!InfoFlags.Contains(name)) return Fail("Unknown option: " + token + ".");
                if (name == "ops" || name == "init")
                {
                    if (!TakeBool("info", name, inline, out var b, out var eb)) return Fail(eb);
                    if (name == "ops") options.Ops = b; else options.Initializers = b;
                }
                else if (name == "op-filter")
                {
                    if (!TakeString("info", "op-filter", rest, ref i, inline, out var v, out var e)) return Fail(e);
                    options.OpFilter = v;
                }
            }
            else if (IsShort(token))
            {
                return Fail("Unknown option: " + token + ".");
            }
            else
            {
                positionals.Add(token);
            }
        }
        if (positionals.Count < 1) return Fail("The info command requires a model file.");
        if (positionals.Count > 1) return Fail("The info command takes a single model file.");
        options.File = positionals[0];
        return new ParseResult { Outcome = ParseOutcome.Ok, Verb = "info", Value = options };
    }

    static ParseResult ParseRun(List<string> rest)
    {
        var options = new RunOptions();
        var positionals = new List<string>();
        for (int i = 0; i < rest.Count; i++)
        {
            var token = rest[i];
            if (IsFlag(token))
            {
                SplitFlag(token, out var name, out var inline);
                if (!RunFlags.Contains(name)) return Fail("Unknown option: " + token + ".");
                switch (name)
                {
                    case "save-input":
                    case "softmax":
                    case "print-input":
                    case "disable-simd":
                    case "enable-intrinsics":
                    case "profile":
                    case "optimize-memory":
                        if (!TakeBool("run", name, inline, out var b, out var eb)) return Fail(eb);
                        if (name == "save-input") options.SaveInput = b;
                        else if (name == "softmax") options.Softmax = b;
                        else if (name == "print-input") options.PrintInput = b;
                        else if (name == "disable-simd") options.DisableSimd = b;
                        else if (name == "enable-intrinsics") options.EnableIntrinsics = b;
                        else if (name == "profile") options.EnableProfiler = b;
                        else options.OptimizeMemory = b;
                        break;
                    case "node":
                        if (!TakeString("run", "node", rest, ref i, inline, out var node, out var e1)) return Fail(e1);
                        options.Node = node;
                        break;
                    case "text":
                        if (!TakeString("run", "text", rest, ref i, inline, out var text, out var e2)) return Fail(e2);
                        options.Text = text;
                        break;
                    case "threads":
                        if (!TakeInt("run", "threads", rest, ref i, inline, out var threads, out var e3)) return Fail(e3);
                        options.Threads = threads;
                        break;
                }
            }
            else if (IsShort(token))
            {
                return Fail("Unknown option: " + token + ".");
            }
            else
            {
                positionals.Add(token);
            }
        }
        if (positionals.Count < 1) return Fail("The run command requires a model file.");
        if (positionals.Count < 2) return Fail("The run command requires at least one model input.");
        options.File = positionals[0];
        options.Inputs = positionals.GetRange(1, positionals.Count - 1);
        return new ParseResult { Outcome = ParseOutcome.Ok, Verb = "run", Value = options };
    }
}

#endregion
