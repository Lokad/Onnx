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
        Console.WriteLine("common runner selftest: " + checks + " checks passed");
        return 0;
    }
}
