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
