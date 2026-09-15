namespace Lokad.Onnx.Bench;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

// Replay-backed voice-model benchmark cases. Inputs load from NPY replay
// fixtures under models/voice-fixtures/replay; a missing replay file fails
// the case loudly instead of substituting a different input. Expected
// tensor contracts: encoder audio_signal float32 [1,128,128] with length
// int64 [1]; decoder encoder_outputs float32 [1,1024,8], targets int32 [1,5],
// target_length int32 [1], input_states float32 [2,1,640]; segmentation
// waveform float32 [1,1,160000]; embedding fbank_features float32 [1,200,80].
static class VoiceModelCases
{
    // Historical long-window gate (2e-4) kept for the published 2026-09-14
    // summaries: the 10-second segmentation window amplifies
    // cross-implementation rounding through normalization and recurrence,
    // so it validated under a wider gate. The canonical row now uses the
    // standard 1e-4 gate.
    public const double SegLongTolerance = 2e-4;

    public static string ReplayDir(string root) =>
        Path.Combine(root, "models", "voice-fixtures", "replay");

    public static Dictionary<string, ITensor> EmbeddingInputs(string root)
    {
        return LoadNamed(root, new (string, string)[]
        {
            ("fbank_features", "embedding_fbank.npy"),
        });
    }
    public static Dictionary<string, ITensor> EmbeddingInputs400(string root)
    {
        return LoadNamed(root, new (string, string)[]
        {
            ("fbank_features", "embedding_fbank400.npy"),
        });
    }
    public static Dictionary<string, ITensor> EmbeddingInputs800(string root)
    {
        return LoadNamed(root, new (string, string)[]
        {
            ("fbank_features", "embedding_fbank800.npy"),
        });
    }

    public static Dictionary<string, ITensor> DecoderStep1Inputs(string root)
    {
        return LoadNamed(root, new (string, string)[]
        {
            ("encoder_outputs", "decoder_enc.npy"),
            ("targets", "decoder_tgt.npy"),
            ("target_length", "decoder_tlen.npy"),
            ("input_states_1", "decoder_s0_1.npy"),
            ("input_states_2", "decoder_s0_2.npy"),
        });
    }

    public static Dictionary<string, ITensor> DecoderChainedInputs(string root, float[] state1, float[] state2)
    {
        var feeds = LoadNamed(root, new (string, string)[]
        {
            ("encoder_outputs", "decoder_enc.npy"),
            ("targets", "decoder_tgt.npy"),
            ("target_length", "decoder_tlen.npy"),
        });
        feeds["input_states_1"] = Named("input_states_1", new DenseTensor<float>(state1, new[] { 2, 1, 640 }));
        feeds["input_states_2"] = Named("input_states_2", new DenseTensor<float>(state2, new[] { 2, 1, 640 }));
        return feeds;
    }

    public static Dictionary<string, ITensor> EncoderInputs(string root)
    {
        return LoadNamed(root, new (string, string)[]
        {
            ("audio_signal", "encoder_mel128.npy"),
            ("length", "encoder_len128.npy"),
        });
    }

    public static Dictionary<string, ITensor> SegmentationInputs(string root)
    {
        return LoadNamed(root, new (string, string)[]
        {
            ("waveform", "seg_real_10s.npy"),
        });
    }

    // Representative rows (P7, --rows representative): same models, new
    // shapes from staged replay assets only. The decoder single-step rows
    // slice the first frame and token of the step-1 fixtures: with zero
    // states this is exactly a real first decoding step; with the recorded
    // s1 states it is a real mid-trajectory step. No fixture is generated
    // or altered; slices copy values and keep replay provenance.
    public static Dictionary<string, ITensor> EncoderInputs64(string root)
    {
        return LoadNamed(root, new (string, string)[]
        {
            ("audio_signal", "encoder_mel64.npy"),
            ("length", "encoder_len64.npy"),
        });
    }

    public static Dictionary<string, ITensor> EncoderInputs256(string root)
    {
        return LoadNamed(root, new (string, string)[]
        {
            ("audio_signal", "encoder_mel256.npy"),
            ("length", "encoder_len256.npy"),
        });
    }

    public static Dictionary<string, ITensor> SegmentationSynth1sInputs(string root)
    {
        return LoadNamed(root, new (string, string)[]
        {
            ("waveform", "seg_synth_1s.npy"),
        });
    }

    public static Dictionary<string, ITensor> DecoderStepSingleInputs(string root)
    {
        var full = DecoderStep1Inputs(root);
        var d = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        d["encoder_outputs"] = Named("encoder_outputs", PrefixFrames((Tensor<float>)full["encoder_outputs"], 1));
        d["targets"] = Named("targets", PrefixTokens((Tensor<int>)full["targets"], 1));
        d["target_length"] = Named("target_length", new DenseTensor<int>(new[] { 1 }, new[] { 1 }));
        d["input_states_1"] = Named("input_states_1", new DenseTensor<float>(new float[2 * 1 * 640], new[] { 2, 1, 640 }));
        d["input_states_2"] = Named("input_states_2", new DenseTensor<float>(new float[2 * 1 * 640], new[] { 2, 1, 640 }));
        return d;
    }

    public static Dictionary<string, ITensor> DecoderStepSingleCarriedInputs(string root)
    {
        var d = DecoderStepSingleInputs(root);
        var carried = LoadNamed(root, new (string, string)[]
        {
            ("input_states_1", "decoder_s1_1.npy"),
            ("input_states_2", "decoder_s1_2.npy"),
        });
        d["input_states_1"] = carried["input_states_1"];
        d["input_states_2"] = carried["input_states_2"];
        return d;
    }

    static DenseTensor<float> PrefixFrames(Tensor<float> source, int frames)
    {
        int[] dims = source.Dimensions.ToArray();
        if (dims.Length != 3) throw new InvalidOperationException("expected rank-3 encoder outputs.");
        var src = source.ToArray();
        var dst = new float[dims[0] * dims[1] * frames];
        Array.Copy(src, dst, dst.Length);
        return new DenseTensor<float>(dst, new[] { dims[0], dims[1], frames });
    }

    static DenseTensor<int> PrefixTokens(Tensor<int> source, int tokens)
    {
        int[] dims = source.Dimensions.ToArray();
        if (dims.Length != 2) throw new InvalidOperationException("expected rank-2 targets.");
        var src = source.ToArray();
        var dst = new int[dims[0] * tokens];
        Array.Copy(src, dst, dst.Length);
        return new DenseTensor<int>(dst, new[] { dims[0], tokens });
    }

    static Dictionary<string, ITensor> LoadNamed(string root, (string name, string file)[] entries)
    {
        var d = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        foreach (var (name, file) in entries)
        {
            string path = Path.Combine(ReplayDir(root), file);
            if (!File.Exists(path))
                throw new InvalidOperationException("voice replay asset missing: " + path
                    + " (git-ignored local asset; recorded in models/voice-fixtures/replay/replay.json).");
            VoiceProvenance.VerifyReplayFile(path);
            d[name] = Named(name, NpySupport.ReadTensor(path));
        }
        return d;
    }

    static ITensor Named(string name, ITensor tensor)
    {
        tensor.Name = name;
        return tensor;
    }
}
