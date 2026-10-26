namespace Lokad.Onnx.CLI;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

using static Lokad.Onnx.Data;
using static Lokad.Onnx.Text;
using static Lokad.Onnx.Runtime;

#region Enums
public enum ExitResult
{
    SUCCESS = 0,
    UNHANDLED_EXCEPTION = 1,
    INVALID_OPTIONS = 2,
    NOT_FOUND = 4,
    INVALID_INPUT = 5,
    UNKNOWN_ERROR = 7,
    CANCELED = 130
}
#endregion

class Program
{
    #region Constructor
    static Program()
    {
        AppDomain.CurrentDomain.UnhandledException += Program_UnhandledException;
        Console.CancelKeyPress += Console_CancelKeyPress;
        Console.OutputEncoding = Encoding.UTF8;
    }
    #endregion

    #region Methods

    #region Entry point
    static void Main(string[] args)
    {
        bool debug = (args.Contains("--debug") || args.Contains("-d"));
        var parsed = ArgsParser.Parse(args);
        UseConsoleLogging(debug, parsed.Verb is "transcribe" or "diarize" or "embed");
        switch (parsed.Outcome)
        {
            case ParseOutcome.Version:
                Console.WriteLine("Lokad.Onnx v" + AssemblyVersion.ToString(3));
                Exit(ExitResult.SUCCESS);
                return;
            case ParseOutcome.Help:
                if (string.IsNullOrEmpty(parsed.Verb)) PrintGlobalHelp();
                else PrintVerbHelp(parsed.Verb);
                Exit(ExitResult.SUCCESS);
                return;
            case ParseOutcome.Error:
                Error(parsed.Message);
                if (parsed.Verb is not ("transcribe" or "diarize" or "embed")) PrintGlobalHelp();
                Exit(parsed.Exit);
                return;
            default:
                break;
        }
        switch (parsed.Verb)
        {
            case "info":
                if (parsed.Value is InfoOptions info) ShowInfo(info);
                break;
            case "run":
                if (parsed.Value is RunOptions run) Run(run);
                break;
            case "transcribe":
                if (parsed.Value is TranscribeOptions transcribe) Exit(Transcribe(transcribe));
                break;
            case "diarize":
                if (parsed.Value is DiarizeOptions diarize) Exit(Diarize(diarize));
                break;
            case "embed":
                if (parsed.Value is EmbedOptions embed) Exit(Embed(embed));
                break;
        }
    }
    #endregion

    static void PrintGlobalHelp()
    {
        Console.WriteLine("Lokad.Onnx command-line help");
        Console.WriteLine("Usage: lonnx <command> [options]");
        Console.WriteLine("Commands:");
        Console.WriteLine("  info <file> [--ops] [--init] [--op-filter <type>]   Get information on an ONNX model.");
        Console.WriteLine("  run <file> <inputs...> [options]                    Run an ONNX model or node.");
        Console.WriteLine("  transcribe <model-directory> <audio.wav> [options]   Transcribe with Whisper Large V3 Turbo or Parakeet.");
        Console.WriteLine("  diarize <audio.wav> [model options]                 Produce Community-1 speaker timelines.");
        Console.WriteLine("  embed <prepared-directory> [options]                Produce EmbeddingGemma 2 text vectors.");
        Console.WriteLine("Common options:");
        Console.WriteLine("  --debug, -d   Enable debug mode.");
        Console.WriteLine("  --help        Show this help and exit.");
        Console.WriteLine("  --version     Show version and exit.");
    }

    static void PrintVerbHelp(string verb)
    {
        if (verb == "info")
        {
            Console.WriteLine("Lokad.Onnx command-line help");
            Console.WriteLine("Usage: lonnx info <file> [options]");
            Console.WriteLine("Options:");
            Console.WriteLine("  --ops               Only print out a list of distinct ops present in the model.");
            Console.WriteLine("  --init              Only print out a list of initializers present in the model.");
            Console.WriteLine("  --op-filter <type>  Filter on ops with this type.");
        }
        else if (verb == "run")
        {
            Console.WriteLine("Lokad.Onnx command-line help");
            Console.WriteLine("Usage: lonnx run <file> <inputs...> [options]");
            Console.WriteLine("Options:");
            Console.WriteLine("  --save-input        Save any input arguments to the model as additional files.");
            Console.WriteLine("  --softmax           Apply the softmax function to output vectors.");
            Console.WriteLine("  --node <label>      Only run the model node with this label.");
            Console.WriteLine("  --text <model>      Read the user input as text using this model.");
            Console.WriteLine("  --print-input       Print the input tensors that will be fed to the model.");
            Console.WriteLine("  --disable-simd      Disable CPU SIMD features.");
            Console.WriteLine("  --enable-intrinsics Enable CPU SIMD intrinsics.");
            Console.WriteLine("  --profile           Enable the profiler which logs detailed stats about ONNX node execution times.");
            Console.WriteLine("  --optimize-memory   Allow earlier release of dead intermediate views.");
            Console.WriteLine("  --threads <n>       Worker threads for batch-parallel kernels (default 1, sequential).");
        }
        else if (verb == "embed")
        {
            Console.WriteLine("Usage: lonnx embed <prepared-directory> (--query <text> | --documents <file.jsonl|->) [options]");
            Console.WriteLine("Uses local prepared FP32 EmbeddingGemma 2 assets. Documents have id, text, and optional title.");
            Console.WriteLine("  --dimensions <n>           128, 256, 512, or 768 (default 768); truncate and renormalize.");
            Console.WriteLine("  --max-tokens <n>           Token limit including the task prefix and BOS/EOS (default 512); overflow fails.");
            Console.WriteLine("  --batch-size <n>           Maximum documents per evaluation (default 1).");
            Console.WriteLine("  --padded-token-budget <n>  Maximum rows times padded length (default max-tokens * batch-size).");
            Console.WriteLine("Writes JSONL vectors, dimensions, token counts, and space identities to stdout; diagnostics go to stderr.");
            Console.WriteLine("Limits above 512, up to 8192, require separate sequence-length and memory qualification.");
        }
        else if (verb == "diarize")
        {
            Console.WriteLine("Usage: lonnx diarize <audio.wav> --segmentation <model.onnx> --embedding <encoder.onnx> --projection <projection.onnx> --plda <plda.json> [--json]");
            Console.WriteLine("Uses local FP32 Community-1 models and prepared projection/PLDA assets.");
            Console.WriteLine("Accepts mono/stereo WAV, 8000..192000 Hz, at most ten minutes; mixes/resamples to mono 16000 Hz.");
            Console.WriteLine("Prints start/end seconds and speaker labels. --json includes ordinary and exclusive timelines and speaker centroids.");
            Console.WriteLine("Intervals are clipped to the recording; tied votes choose the earliest canonical speaker. Diagnostics go to stderr.");
        }
        else if (verb == "transcribe")
        {
            Console.WriteLine("Usage: lonnx transcribe <model-directory> <audio.wav> [options]");
            Console.WriteLine("  --model-type <id>   whisper (default) or parakeet.");
            Console.WriteLine("  --language <code>   Required for Whisper, such as en or fr; omit for Parakeet.");
            Console.WriteLine("  --max-tokens <n>    Generated token limit: Whisper 1..444; Parakeet 1..4096 (defaults to maximum).");
            Console.WriteLine("  --json              Print text, tokens and stop/confidence metadata as JSON.");
            Console.WriteLine("  --recording         Up to ten minutes. Whisper uses segment timestamps; Parakeet prefers quiet boundaries.");
            Console.WriteLine("  --max-windows <n>   Recording work limit: 1..512 (default 256). --max-tokens applies per window.");
            Console.WriteLine("Accepts mono/stereo PCM or float WAV, 8000..192000 Hz, at most 30 seconds.");
            Console.WriteLine("Audio is mixed to mono and resampled to 16000 Hz. Uses local FP32 split model assets.");
            Console.WriteLine("Text/JSON goes to stdout; diagnostics go to stderr. Parakeet detects language and reports encoder-frame token positions.");
            Console.WriteLine("Parakeet recording windows are independent, at most 30 seconds; JSON reports quiet or hard cuts and partial progress.");
        }
    }

    static ExitResult Embed(EmbedOptions options)
    {
        CancelableInference = true;
        try
        {
            if (!Directory.Exists(options.ModelDirectory)) throw new DirectoryNotFoundException("Prepared model directory not found: " + options.ModelDirectory);
            if (options.Documents is not null && options.Documents != "-" && !File.Exists(options.Documents))
                throw new FileNotFoundException("JSONL documents not found: " + options.Documents);
            Cts.Token.ThrowIfCancellationRequested();
            var model = new EmbeddingGemma2Embedder(options.ModelDirectory, new EmbeddingGemma2Options {
                MaximumTokens = options.MaximumTokens, MaximumBatchSize = options.BatchSize, MaximumPaddedTokens = options.PaddedTokenBudget });
            if (options.Query is not null)
            {
                WriteEmbedding(null, model.EmbedQuery(options.Query, options.Dimensions, Cts.Token));
                return ExitResult.SUCCESS;
            }
            string documentPath = options.Documents ?? throw new ArgumentException("Supply --query or --documents.");
            if (documentPath == "-") Console.InputEncoding = Encoding.UTF8;
            using var file = documentPath == "-" ? null : File.OpenText(documentPath);
            TextReader reader = file ?? Console.In;
            var documents = new List<EmbeddingGemma2Document>();
            var ids = new List<JsonElement>();
            int lineNumber = 0;
            void Flush()
            {
                if (documents.Count == 0) return;
                var vectors = model.EmbedDocuments(documents, options.Dimensions, Cts.Token);
                for (int i = 0; i < vectors.Length; i++) WriteEmbedding(ids[i], vectors[i]);
                documents.Clear(); ids.Clear();
            }
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                lineNumber++;
                Cts.Token.ThrowIfCancellationRequested();
                if (string.IsNullOrWhiteSpace(line)) continue;
                using var json = JsonDocument.Parse(line);
                var record = json.RootElement;
                if (record.ValueKind != JsonValueKind.Object || !record.TryGetProperty("id", out var id) ||
                    id.ValueKind is not (JsonValueKind.String or JsonValueKind.Number) ||
                    !record.TryGetProperty("text", out var text) || text.ValueKind != JsonValueKind.String)
                    throw new InvalidDataException("JSONL line " + lineNumber + " requires a string/number id and string text.");
                string? title = null;
                if (record.TryGetProperty("title", out var titleValue))
                {
                    if (titleValue.ValueKind is not (JsonValueKind.String or JsonValueKind.Null))
                        throw new InvalidDataException("JSONL line " + lineNumber + " title must be a string or null.");
                    title = titleValue.GetString();
                }
                ids.Add(id.Clone());
                documents.Add(new EmbeddingGemma2Document(text.GetString() ?? throw new InvalidDataException("Document text cannot be null."), title));
                if (documents.Count == options.BatchSize) Flush();
            }
            Flush();
            return ExitResult.SUCCESS;
        }
        catch (OperationCanceledException) { Console.Error.WriteLine("Embedding canceled."); return ExitResult.CANCELED; }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException) { Error(ex.Message); return ExitResult.NOT_FOUND; }
        catch (Exception ex) when (ex is InvalidDataException or IOException or ArgumentException or NotSupportedException or JsonException or UnauthorizedAccessException
                                    or KeyNotFoundException or InvalidOperationException)
        { Error(ex.Message); return ExitResult.INVALID_INPUT; }
        finally { CancelableInference = false; }
    }

    static void WriteEmbedding(JsonElement? id, EmbeddingGemma2Embedding result) => Console.WriteLine(JsonSerializer.Serialize(new {
        id, model = "embeddinggemma-2", precision = "fp32", dimensions = result.Values.Length,
        tokenCount = result.TokenCount, spaceId = result.SpaceId, embedding = result.Values }));

    static ExitResult Transcribe(TranscribeOptions options)
    {
        CancelableInference = true;
        try
        {
            if (options.ModelDirectory.StartsWith("http://", StringComparison.OrdinalIgnoreCase)
                || options.ModelDirectory.StartsWith("https://", StringComparison.OrdinalIgnoreCase)
                || options.AudioFile.StartsWith("http://", StringComparison.OrdinalIgnoreCase)
                || options.AudioFile.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Transcription requires a local model directory and WAV file.");
            if (!Directory.Exists(options.ModelDirectory)) throw new DirectoryNotFoundException("Model directory not found: " + options.ModelDirectory);
            using var input = File.OpenRead(options.AudioFile);
            var audio = WaveAudio.ReadMono(input, TimeSpan.FromSeconds(options.Recording ? 600 : 30));
            if (options.ModelType == "whisper")
            {
                using var generation = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(options.ModelDirectory, "generation_config.json")));
                if (!generation.RootElement.GetProperty("lang_to_id").TryGetProperty("<|" + options.Language + "|>", out _))
                {
                    Error("Use a language code from this model's generation configuration.");
                    return ExitResult.INVALID_OPTIONS;
                }
            }
            Cts.Token.ThrowIfCancellationRequested();
            var pcm = AudioResampler.Resample(audio.Samples, audio.SampleRate, WhisperAudio.SampleRate);
            Cts.Token.ThrowIfCancellationRequested();
            object result;
            string text;
            bool truncated;
            if (options.ModelType == "parakeet")
            {
                var model = new ParakeetTranscriber(options.ModelDirectory);
                var policy = ParakeetTranscriptionOptions.Default with { MaxTokens = options.MaxTokens };
                if (options.Recording)
                {
                    var transcription = model.TranscribeRecording(pcm, ParakeetTranscriber.SampleRate,
                        new ParakeetRecordingOptions(policy, options.MaxWindows), Cts.Token);
                    result = transcription; text = transcription.Text; truncated = transcription.StopReason != ParakeetRecordingStopReason.Completed;
                }
                else
                {
                    var transcription = model.Transcribe(pcm, ParakeetTranscriber.SampleRate, policy, Cts.Token);
                    result = transcription; text = transcription.Text; truncated = transcription.StopReason == ParakeetStopReason.TokenLimit;
                }
            }
            else
            {
                var model = new WhisperTranscriber(options.ModelDirectory);
                var policy = WhisperTranscriptionOptions.ForLanguage(options.Language) with { MaxNewTokens = options.MaxTokens };
                if (options.Recording)
                {
                    var transcription = model.TranscribeRecording(pcm, WhisperAudio.SampleRate,
                        new WhisperRecordingOptions(policy, options.MaxWindows), Cts.Token);
                    result = transcription; text = transcription.Text; truncated = transcription.StopReason != WhisperRecordingStopReason.Completed;
                }
                else
                {
                    var transcription = model.Transcribe(pcm, WhisperAudio.SampleRate, policy, Cts.Token);
                    result = transcription; text = transcription.Text; truncated = transcription.StopReason == WhisperStopReason.TokenLimit;
                }
            }
            if (options.Json)
            {
                var json = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
                json.Converters.Add(new JsonStringEnumConverter());
                Console.WriteLine(JsonSerializer.Serialize(result, json));
            }
            else Console.WriteLine(text);
            if (truncated)
                Console.Error.WriteLine(options.Recording
                    ? "Recording transcription stopped before completion; inspect --json for the stop reason and processed duration."
                    : "Transcription reached the token limit; the text may be incomplete.");
            return ExitResult.SUCCESS;
        }
        catch (OperationCanceledException)
        {
            Console.Error.WriteLine("Transcription canceled.");
            return ExitResult.CANCELED;
        }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException)
        {
            Error(ex.Message);
            return ExitResult.NOT_FOUND;
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException or ArgumentException or NotSupportedException or JsonException or UnauthorizedAccessException
                                    or KeyNotFoundException or InvalidOperationException)
        {
            Error(ex.Message);
            return ExitResult.INVALID_INPUT;
        }
        finally { CancelableInference = false; }
    }

    static ExitResult Diarize(DiarizeOptions options)
    {
        CancelableInference = true;
        try
        {
            foreach (string path in new[] { options.AudioFile, options.Segmentation, options.Embedding, options.Projection, options.Plda })
            {
                if (path.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || path.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
                    throw new ArgumentException("Diarization requires local model and WAV paths.");
                if (!File.Exists(path)) throw new FileNotFoundException("Diarization input or model not found: " + path, path);
            }
            using var input = File.OpenRead(options.AudioFile);
            var audio = WaveAudio.ReadMono(input, TimeSpan.FromMinutes(10));
            Cts.Token.ThrowIfCancellationRequested();
            var pcm = AudioResampler.Resample(audio.Samples, audio.SampleRate, Community1Diarizer.SampleRate);
            Cts.Token.ThrowIfCancellationRequested();
            var model = new Community1Diarizer(options.Segmentation, options.Embedding, options.Projection, options.Plda);
            var result = model.Diarize(pcm, Community1Diarizer.SampleRate, Cts.Token);
            if (options.Json)
            {
                var json = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
                json.Converters.Add(new JsonStringEnumConverter());
                Console.WriteLine(JsonSerializer.Serialize(result, json));
            }
            else foreach (var interval in result.Intervals)
                Console.WriteLine(FormattableString.Invariant($"{interval.Start:F6}\t{interval.End:F6}\tSPEAKER_{interval.Speaker:D2}"));
            return ExitResult.SUCCESS;
        }
        catch (OperationCanceledException) { Console.Error.WriteLine("Diarization canceled."); return ExitResult.CANCELED; }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException) { Error(ex.Message); return ExitResult.NOT_FOUND; }
        catch (Exception ex) when (ex is InvalidDataException or IOException or ArgumentException or NotSupportedException or JsonException or UnauthorizedAccessException
                                    or KeyNotFoundException or InvalidOperationException)
        { Error(ex.Message); return ExitResult.INVALID_INPUT; }
        finally { CancelableInference = false; }
    }

    static void ShowInfo(InfoOptions io)
    {
        ExitIfFileNotFound(io.File);
        if (io.Ops)
        {
            PrintModelOps(io.File);
        }
        else if (io.Initializers)
        {
            PrintModelInitializers(io.File);
        }
        else
        {
            PrintModelInfo(io.File, io.OpFilter);
        }
        ExitWithSuccess();
    }
    static void Run(RunOptions ro)
    {
        if (ro.File.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            ro.File.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            Error("Remote model locations are not supported; download the model first and pass a local file path.");
            Exit(ExitResult.NOT_FOUND);
            return;
        }
        ExitIfFileNotFound(ro.File);

        var graph = OnnxImport.Load(ro.File);
        if (graph is null)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        ITensor[]? ui;
        try
        {
            if (!string.IsNullOrEmpty(ro.Text))
            {
                if (ro.Inputs.Count() != 1)
                {
                    Error("The --text option requires exactly one input argument.");
                    Exit(ExitResult.INVALID_INPUT);
                    return;
                }
                var head = ro.Text.Split(':')[0];
                if ((string.IsNullOrEmpty(head) || head == "me5s") && !Lokad.Onnx.Text.EnsureMe5sTokenizer())
                {
                    Exit(ExitResult.INVALID_INPUT);
                    return;
                }
                ui = GetTextTensors(ro.Inputs.First(), ro.Text);
            }
            else
            {
                ui = GetInputTensorsFromFileArgs(ro.Inputs, ro.SaveInput);
            }
        }
        catch (Exception ex) when (!IsFatal(ex))
        {
            Error("Cannot prepare model inputs: {m}.", ex.Message);
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        if (ui is null || ui.Length == 0)
        {
            Exit(ExitResult.INVALID_INPUT);
            return;
        }
        if (ro.PrintInput)
        {
            Info("Printing {c} input tensor(s)...", ui.Length);
            foreach (var t in ui)
            {
                Info("{n}:{d}", t.TensorNameDesc(), t.PrintData(false));
            }
        }

        bool useSimd = !ro.DisableSimd;
        bool useIntrinsics = ro.DisableSimd
            ? false
            : (!System.Numerics.Vector.IsHardwareAccelerated
                ? HardwareIntrinsics.IsX86FmaSupported
                : (ro.EnableIntrinsics || HardwareIntrinsics.IsX86FmaSupported));
        if (ro.Threads < 1)
        {
            Error("Thread count must be at least 1.");
            Exit(ExitResult.INVALID_OPTIONS);
            return;
        }
        if (ro.Threads > 1)
        {
            Info("Multi-threaded kernels using up to {t} worker threads.", ro.Threads);
        }
        var execOptions = new ExecutionOptions(
            ro.OptimizeMemory ? OptimizationMode.Memory : OptimizationMode.Speed,
            new TensorExecutionOptions(useSimd, useIntrinsics, ro.Threads));
        if (!useSimd)
        {
            Info("CPU SIMD features disabled.");
        }
        else
        {
            Info("CPU SIMD acceleration: {a}.", System.Numerics.Vector.IsHardwareAccelerated);
            if (System.Numerics.Vector.IsHardwareAccelerated)
            {
                Info("CPU SIMD vector size: {v} bits.", System.Numerics.Vector<int>.Count * 4 * 8);

            }
        }

        if (useIntrinsics)
        {
            Info("CPU SIMD available intrinsics: {s}.", HardwareIntrinsics.GetFullInfo());
        }
        else
        {
            Info("Not using CPU SIMD intrinsics.");
        }


        using var profilerScope = ro.EnableProfiler ? Profiler.BeginExecution(true) : null;

        if (ro.Node == "")
        {
            if (graph.Execute(ui, true, ExecutionProvider.CPU, execOptions))
            {
                PrintOutputs(graph, ro);
                if (ro.EnableProfiler && graph.LastProfile is { } profile) PrintProfile(profile);
                Exit(ExitResult.SUCCESS);
            }
            else
            {
                Error("Inference failed: {m}.", graph.LastErrorMessage ?? "invalid inputs");
                Exit(graph.LastFailedNodeName is null ? ExitResult.INVALID_INPUT : ExitResult.UNKNOWN_ERROR);
            }
        }
        else
        {
            if (graph.ExecuteNode(ui, ro.Node, true, ExecutionProvider.CPU, execOptions))
            {
                PrintOutputs(graph, ro);
                if (ro.EnableProfiler && graph.LastProfile is { } profile) PrintProfile(profile);
                Exit(ExitResult.SUCCESS);
            }
            else if (!graph.Nodes.Any(n => n.Name == ro.Node))
            {
                Error("Inference failed: node {n} not found in graph.", ro.Node);
                Exit(ExitResult.NOT_FOUND);
            }
            else
            {
                Error("Inference failed at node {n}: {m}.", ro.Node, graph.LastErrorMessage ?? "invalid inputs");
                Exit(graph.LastFailedNodeName is null ? ExitResult.INVALID_INPUT : ExitResult.UNKNOWN_ERROR);
            }
        }
    }

    static void PrintOutputs(ComputationalGraph graph, RunOptions ro)
    {
        Info("Printing outputs...");
        foreach (var o in graph.Outputs.Values)
        {
            if (o is null) continue;
            if (ro.Softmax && o.Rank == 1)
            {
                Info("Applying softmax to {n}...", o.TensorNameDesc());
                Info("{n}:{v}", o.TensorNameDesc() + "-><softmax>", o.Softmax().PrintData(false));
            }
            else if (ro.Softmax && o is INumericTensor num && num.Rank == 2 && num.Dims[0] == 1)
            {
                Info("Converting {n} to vector and applying softmax...", o.TensorNameDesc());
                Info("{n}:{v}", o.TensorNameDesc() + "-><softmax>", num.RemoveDim(0).Softmax().PrintData(false));
            }
            else
            {
                Info("{n}:{v}", o.TensorNameDesc(), o.PrintData(false));
            }
        }
    }

    static void PrintModelInfo(string file, string? _opfilter)
    {
        ExitIfFileNotFound(file);
        string? opfilter = null;
        if (_opfilter is not null)
        {
            if (!Enum.TryParse<OpType>(_opfilter, true, out var op)
                || !string.Equals(_opfilter, Enum.GetName(op), StringComparison.OrdinalIgnoreCase))
            {
                Error("The specified operation type {op} is not valid.", _opfilter);
                Exit(ExitResult.INVALID_OPTIONS);
                return;
            }
            opfilter = op.ToString();
        }
        var m = ReadModelMetadataOrExit(file);
        if (m is null) return;
        var tensors = new Dictionary<string, string>();
        Info("Graph has input tensors: {i}", m.Inputs.Select(t => t.Describe()));
        Info("Graph has output tensors: {o}", m.Outputs.Select(t => t.Describe()));
        Info("Graph has initializer tensors: {i}", m.Initializers.Select(t => t.Describe()));
        void AddTensorRole(OnnxValueInfo tensor, string role)
        {
            var description = tensors.TryGetValue(tensor.Name, out var existing) ? existing : tensor.Describe();
            tensors[tensor.Name] = description + "<" + role + ">";
        }
        foreach (var t in m.Initializers)
        {
            AddTensorRole(t, "initializer");
        }
        foreach (var t in m.Inputs)
        {
            AddTensorRole(t, "input");
        }
        foreach (var t in m.Outputs)
        {
            AddTensorRole(t, "output");
        }
        foreach (var n in m.Nodes)
        {
            foreach (var o in n.Outputs)
            {
                if (!string.IsNullOrEmpty(o) && !tensors.ContainsKey(o)) tensors.Add(o, o + "<intermediate>");
            }
        }
        string GetTensorDesc(string name)
        {
            if (string.IsNullOrEmpty(name))
            {
                return "<empty>";
            }
            return tensors.TryGetValue(name, out var desc) ? desc : $"{name}<unknown>";
        }
        string GetAttributeValueDesc(object value) =>
            value switch
            {
                OnnxTensor tensor => "tensor " + tensor.Describe() + " (not materialized)",
                ITensor i => "tensor " + i.TensorNameDesc() + ":" + i.PrintData(true),
                int n => "int " + n.ToString(),
                int[] na => "int[] " + na.Print(),
                long l => "int64 " + l.ToString(),
                long[] la => "int64[] " + la.Print(),
                float f => "float " + f.ToString(),
                float[] fa => "float[] " + fa.Print(),
                string s => "string " + s,
                string[] sa => "string[] " + sa.Print(),
                OnnxSubgraph graph => $"graph {graph.Name} ({graph.Inputs.Count} inputs, {graph.Outputs.Count} outputs, {graph.Initializers.Count} initializers, {graph.Nodes.Count} nodes)",
                _ => throw new NotSupportedException(value.GetType().Name), 
            };
        if (opfilter is null)
        {
            Info("Printing graph nodes...");
        }
        else
        {
            Info("Printing graph nodes with op {op}...", opfilter);
        }
        foreach (var n in m.Nodes)
        {
            if (opfilter is not null && !string.Equals(n.OpType, opfilter, StringComparison.Ordinal))
            {
                continue;
            }
            Info("Node {node} has op type: {op}, inputs: {inputs}, outputs: {outputs} and "
                + ((n.Attributes is not null && n.Attributes.Count > 0) ?  "the following attributes:" : "no attributes."),
                n.Name, n.OpType,
                n.Inputs.Select(t => GetTensorDesc(t)).ToArray(),
                n.Outputs.Select(t => GetTensorDesc(t)).ToArray());

            if (n.Attributes is not null && n.Attributes.Count > 0)
            {
                foreach (var kv in n.Attributes)
                {
                    Info("  {n}: {v}", kv.Key, GetAttributeValueDesc(kv.Value));
                }
            }
        }
    }

    static OnnxModel? ReadModelMetadataOrExit(string file)
    {
        try { return OnnxImport.ParseMetadata(file); }
        catch (Exception ex) when (!IsFatal(ex))
        {
            Error(ex, "Could not parse {f} as ONNX model file.", file);
            Exit(ExitResult.INVALID_INPUT);
            return null;
        }
    }

    static void PrintModelOps(string file)
    {
        ExitIfFileNotFound(file);
        var m = ReadModelMetadataOrExit(file);
        if (m is null) return;
        Info("Graph has {count} input tensor(s): {in}", m.Inputs.Count, m.Inputs.Select(t => t.Describe()));
        Info("Graph has {count} output tensor(s): {out}", m.Outputs.Count, m.Outputs.Select(t => t.Describe()));
        Info("Graph has {count} initializer tensor(s): {out}", m.Initializers.Count, m.Initializers.Select(t => t.Describe()));
        List<(string Domain, string OpType)> ops = new List<(string Domain, string OpType)>();
        foreach(var node in m.Nodes)
        {
            var key = (node.Domain ?? "", node.OpType ?? "");
            if (!ops.Contains(key))
            {
                ops.Add(key);
            }
        }
        Info("Printing list of distinct ONNX operations in model {f}...", file);
        foreach(var (domain, opName) in ops)
        {
            var display = (string.IsNullOrEmpty(domain) ? "" : domain + ":") + opName + " ";
            int version = m.Opset.TryGetValue(domain ?? "", out var v) ? v
                : m.Opset.TryGetValue("", out var d) ? d : -1;
            bool supported = Enum.TryParse<OpType>(opName, false, out var op)
                && string.Equals(opName, Enum.GetName(op), StringComparison.Ordinal)
                && OperatorSchemas.IsSupported(op, domain, version, false);
            Console.WriteLine(display + (supported ? "[supported]" : "[unsupported]"));
        }
        Info("{d} total distinct operations in model. [supported] = supported by backend.", ops.Count);
    }

    static void PrintModelInitializers(string file)
    {
        ExitIfFileNotFound(file);
        var m = ReadModelMetadataOrExit(file);
        if (m is null) return;
        var inputs = m.Inputs.Select(i => i.Name);
        List<string> initializers = new List<string>();
        foreach (var i in m.Initializers)
        {
            if (inputs.Contains(i.Name))
            {
                initializers.Add(i.Describe() + "*");
            }
            else
            {
                initializers.Add(i.Describe());
            }
             
        }
        Info("Graph has {count} input tensors: {in}", m.Inputs.Count, m.Inputs.Select(t => t.Describe()));
        Info("Graph has {count} output tensors: {out}", m.Outputs.Count, m.Outputs.Select(t => t.Describe()));
        Info("Printing list of ONNX initializers in graph...");
        foreach (var i in initializers)
        {
            Console.WriteLine(i);
        }
        Info("{d} total initializers in model. * = initializer for graph input.", m.Initializers.Count);
    }

    static void PrintProfile(Stack<NodeProfile> profile)
    {
        var times = profile.Select(np => (np.Op, np.OpsProfile.Sum(op => op.Time.TotalMilliseconds)))
            .GroupBy(x => x.Item1)
            .Select(g => (g.Key, Convert.ToInt32(g.Sum(gx => gx.Item2)), g.Count()))
            .OrderByDescending(t => t.Item2)
            .ToArray();
        var times2 = profile.Select(np => (np.OpsProfile.Select(op => (op.Stage, op.Time)))).SelectMany(x => x)
         .GroupBy(x => x.Item1)
         .Select(g => (g.Key, Convert.ToInt32(g.Sum(gx => gx.Item2.TotalMilliseconds)), g.Count()))
         .OrderByDescending(t => t.Item2)
         .ToArray();
        Info("Graph op times (ms):");
        foreach (var t in times)
        {
            Info("  {op}({count}): {ms}ms", t.Item1, t.Item3, t.Item2);
        }
        Info("Total graph node count: " + profile.Count);
        Info("Total graph execution time: " + times.Sum(t => t.Item2) + "ms");
        Info("Execution time breakdown (ms):");
        foreach (var t in times2)
        {
            Info("  {stage}: {ms}ms", Profiler.StageDescription(t.Item1), t.Item2);
        }
        var times3 = profile.Select(np => (np.Op,
                                        np.OpsProfile.Select(op => (op.Stage, op.Time.TotalMilliseconds))
                                                        .GroupBy(s => s.Stage)
                                                        .Select(gs => (gs.Key, gs.Sum(i => i.TotalMilliseconds)))))
            .GroupBy(x => x.Item1)
            .Select(x => (x.Key, x.Select(i => i.Item2)
                                    .SelectMany(x => x)
                                    .GroupBy(x => x.Key)
                                    .Select(x => (x.Key, x.Sum(i => i.Item2)))))
            .ToArray();
        Info("Per-op stage breakdown (ms):");
        foreach (var op in times3)
        {
            Info("  {op}: {stages}", op.Item1, string.Join(", ", op.Item2.Select(s => Profiler.StageDescription(s.Item1) + "=" + s.Item2.ToString("F1") + "ms")));
        }
    }

    public static void Exit(ExitResult result)
    {
        Environment.Exit((int)result);
    }

    public static void ExitIfFileNotFound(string filePath)
    {
        if (filePath.StartsWith("http://") || filePath.StartsWith("https://")) return;
        if (!File.Exists(filePath))
        {
            Error("The file {0} does not exist.", filePath);
            Exit(ExitResult.NOT_FOUND);
        }
    }

    public static void ExitWithSuccess() => Exit(ExitResult.SUCCESS);

    static void UseConsoleLogging(bool debug, bool standardError)
    {
        Log.MinLevel = debug ? Lokad.Onnx.LogLevel.Debug : standardError ? Lokad.Onnx.LogLevel.Warn : Lokad.Onnx.LogLevel.Info;
        Log.Sink = (level, message) =>
        {
            var stamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.ffff");
            var writer = standardError ? Console.Error : Console.Out;
            writer.WriteLine(stamp + " " + level.ToString().ToUpperInvariant().PadRight(5) + " " + message);
        };
    }

    #endregion

    #region Event Handlers
    private static void Program_UnhandledException(object sender, UnhandledExceptionEventArgs e)
    {
        Error("Unhandled runtime error occurred. Lokad.Onnx CLI will now shutdown.");
        Console.Error.WriteLine((e.ExceptionObject as Exception)?.ToString() ?? "Unknown error.");
        Exit(ExitResult.UNHANDLED_EXCEPTION);
    }

    private static void Console_CancelKeyPress(object? sender, ConsoleCancelEventArgs e)
    {
        if (CancelableInference)
        {
            e.Cancel = true;
            Cts.Cancel();
            return;
        }
        Info("Ctrl-C pressed. Exiting.");
        Cts.Cancel();
        Exit(ExitResult.SUCCESS);
    }
    #endregion
    
    #region Fields
    static readonly CancellationTokenSource Cts = new CancellationTokenSource();
    static volatile bool CancelableInference;
    #endregion
}
