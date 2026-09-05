namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Linq;
using System.Text;

public enum LogLevel
{
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

/// <summary>
/// Minimal in-core logging sink. Call sites use NLog-style message templates
/// with named holes ({name}) substituted positionally, exactly as the former
/// Microsoft.Extensions.Logging provider did with ParseMessageTemplates.
/// The sink is null by default (library runs silent, as before with
/// NullLogger); hosts such as the CLI assign it to their own logger.
/// </summary>
public static class Log
{
    public static Action<LogLevel, string>? Sink;

    public static void Write(LogLevel level, string messageTemplate, params object?[] args)
    {
        var sink = Sink;
        if (sink is null) return;
        sink(level, Render(messageTemplate, args));
    }

    public static string Render(string messageTemplate, object?[] args)
    {
        if (string.IsNullOrEmpty(messageTemplate) || args is null || args.Length == 0) return messageTemplate;
        var output = new StringBuilder(messageTemplate.Length + 32);
        int used = 0;
        int i = 0;
        while (i < messageTemplate.Length)
        {
            char current = messageTemplate[i];
            if (current == '{' && i + 1 < messageTemplate.Length)
            {
                if (messageTemplate[i + 1] == '{')
                {
                    output.Append('{');
                    i += 2;
                    continue;
                }
                int end = messageTemplate.IndexOf('}', i + 1);
                if (end < 0)
                {
                    output.Append(messageTemplate, i, messageTemplate.Length - i);
                    break;
                }
                if (used < args.Length) output.Append(RenderValue(args[used++]));
                else output.Append(messageTemplate, i, end - i + 1);
                i = end + 1;
                continue;
            }
            if (current == '}' && i + 1 < messageTemplate.Length && messageTemplate[i + 1] == '}')
            {
                output.Append('}');
                i += 2;
                continue;
            }
            output.Append(current);
            i++;
        }
        return output.ToString();
    }

    static string RenderValue(object? value) => value switch
    {
        null => string.Empty,
        string text => text,
        IEnumerable items => string.Join(", ", items.Cast<object?>().Select(item => item is string text ? "\"" + text + "\"" : RenderValue(item))),
        _ => value.ToString() ?? string.Empty,
    };
}
