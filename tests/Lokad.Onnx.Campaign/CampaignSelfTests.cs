using System;
using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx;
using Lokad.Onnx.Bench;

// Tiny deterministic checks for the common runner itself; no model or CPU burn.
internal static class CampaignSelfTests
{
    internal static int Run()
    {
        int checks = 0;
        void Check(bool value, string name)
        {
            if (!value) throw new InvalidOperationException("selftest failed: " + name);
            checks++;
        }
        var inputs = new Dictionary<string, ITensor>
        {
            ["input_ids"] = new DenseTensor<long>(new long[] { 1, 2 }, new[] { 1, 2 }),
            ["x"] = new DenseTensor<float>(new[] { 1f, -0f }, new[] { 2 })
        };
        string hash = CampaignEvidence.HashInputs(inputs);
        // Independently computed with Python struct.pack little-endian fields.
        Check(hash == "27790380561d92b4fdc793dd38c43c54f3b85f3b5118341870ae04028bf9b089", "versioned input encoding");
        Check(CampaignEvidence.HashInputs(inputs.Reverse().ToDictionary(x => x.Key, x => x.Value)) == hash, "dictionary order is immaterial");
        inputs["x"] = new DenseTensor<float>(new[] { 1f, 0f }, new[] { 2 });
        Check(CampaignEvidence.HashInputs(inputs) != hash, "signed-zero bits are preserved");
        inputs["x"] = new DenseTensor<float>(new[] { 1f, -0f }, new[] { 1, 2 });
        Check(CampaignEvidence.HashInputs(inputs) != hash, "dimensions are part of identity");
        inputs["x"] = new DenseTensor<long>(new long[] { 1, 0 }, new[] { 2 });
        Check(CampaignEvidence.HashInputs(inputs) != hash, "dtype is part of identity");
        inputs["renamed"] = inputs["x"];
        inputs.Remove("x");
        Check(CampaignEvidence.HashInputs(inputs) != hash, "names are part of identity");
        Check(!CampaignEvidence.SteadyWindow(new double[] { 10, 10, 10 }), "three-call plateau is insufficient");
        Check(CampaignEvidence.SteadyWindow(Enumerable.Repeat(10.0, 9).ToArray()), "nine stable samples converge");
        Check(!CampaignEvidence.SteadyWindow(new double[] { 10, 10, 10, 10, 10, 10, 10, 10, 12 }), "recent spike prevents convergence");
        Check(CampaignEvidence.SteadyWindow(new double[] { 100, 10, 10, 10, 10, 10, 10, 10, 10, 10 }), "settled window excludes early warmup");
        Check(!CampaignEvidence.SteadyWindow(new double[] { 10, 10, 10, 10, 10, 10, 10, 10, double.NaN }), "non-finite warmup never converges");
        Check(IsolatedE5.Definition("e5-30pad128") is (128, 30, 0, 128, _), "isolated padded case keeps thirty real tokens");
        Check(IsolatedE5.Definition("e5-128tok").Take == 128, "isolated long input uses canonical truncation");
        string directory = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "onnx-campaign-selftest-" + Guid.NewGuid().ToString("N"));
        System.IO.Directory.CreateDirectory(directory);
        try
        {
            void Reject(Action action, string name)
            {
                bool refused = false;
                try { action(); }
                catch (System.IO.InvalidDataException) { refused = true; }
                catch (System.Text.Json.JsonException) { refused = true; }
                Check(refused, name);
            }
            string path = System.IO.Path.Combine(directory, "config.json");
            System.IO.File.WriteAllText(path, "{\"cpu\":1,\"cpu\":2}");
            Reject(() => IsolatedE5.ReadJson<IsolatedE5.Configuration>(path), "duplicate configuration property refused");
            System.IO.File.WriteAllText(path, "{}");
            Reject(() => IsolatedE5.ReadJson<IsolatedE5.Configuration>(path), "missing constructor properties refused");
            var config = new IsolatedE5.Configuration("root", "e5-8tok", 2, "output", "source", "core", "fixture", null, true);
            System.IO.File.Delete(path);
            IsolatedE5.WriteJson(path, config);
            Check(IsolatedE5.ReadJson<IsolatedE5.Configuration>(path) == config, "configuration round trip preserves nullable oracle digest");
            System.IO.File.WriteAllText(path, System.IO.File.ReadAllText(path).TrimEnd().TrimEnd('}') + ",\"unknown\":true}");
            Reject(() => IsolatedE5.ReadJson<IsolatedE5.Configuration>(path), "unknown configuration property refused");
            string tensorPath = System.IO.Path.Combine(directory, "output-0.f32");
            System.IO.File.WriteAllBytes(tensorPath, new byte[] { 0, 0, 128, 63 });
            var tensor = new IsolatedE5.OutputIdentity("out", new[] { 1 }, "float32", "output-0.f32", CampaignEvidence.HashFile(tensorPath));
            Check(IsolatedE5.ReadOutput(directory, tensor, 0).SequenceEqual(new[] { 1f }), "oracle output decoded as float32");
            Reject(() => IsolatedE5.ReadOutput(directory, tensor with { File = "../output-0.f32" }, 0), "oracle path traversal refused");
            Reject(() => IsolatedE5.ReadOutput(directory, tensor with { Dims = new[] { -1 } }, 0), "negative oracle shape refused");
            Reject(() => IsolatedE5.ReadOutput(directory, tensor with { Dims = new[] { 2 } }, 0), "oracle shape/byte mismatch refused");
            Reject(() => IsolatedE5.ReadOutput(directory, tensor with { Dtype = "int32" }, 0), "oracle dtype mismatch refused");
            Reject(() => IsolatedE5.ReadOutput(directory, tensor with { Sha256 = new string('0', 64) }, 0), "oracle digest mismatch refused");
            System.IO.File.WriteAllBytes(tensorPath, BitConverter.GetBytes(float.NaN));
            Reject(() => IsolatedE5.ReadOutput(directory, tensor with { Sha256 = CampaignEvidence.HashFile(tensorPath) }, 0), "oracle non-finite values refused");
        }
        finally { System.IO.Directory.Delete(directory, true); }
        Console.WriteLine("common runner selftest: " + checks + " checks passed");
        return 0;
    }
}
