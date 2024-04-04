// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace FastBertTokenizer;

// It might be better to use the snake_case naming convention here,
// but it is not available in .NET 6. Note that the tokenizer.json
// format does not use snake_case consequently though.
internal class TokenizerJson
{
    public  string Version { get; init; }

    [JsonPropertyName("added_tokens")]
    public  AddedToken[] AddedTokens { get; init; }

    public  NormalizerSection Normalizer { get; init; }

    [JsonPropertyName("pre_tokenizer")]
    public  PreTokenizerSection PreTokenizer { get; init; }

    [JsonPropertyName("post_processor")]
    public  PostProcessorSection PostProcessor { get; init; }

    public  ModelSection Model { get; init; }

    internal record AddedToken
    {
        public  int Id { get; init; }

        public  string Content { get; init; }
    }

    internal record NormalizerSection
    {
        public  string Type { get; init; }

        [JsonPropertyName("clean_text")]
        public bool CleanText { get; init; } = true;

        [JsonPropertyName("handle_chinese_chars")]
        public bool HandleChineseChars { get; init; } = true;

        [JsonPropertyName("strip_accents")]
        public bool? StripAccents { get; init; }

        public bool Lowercase { get; init; } = true;
    }

    internal record PreTokenizerSection
    {
        public  string Type { get; init; }
    }

    internal record PostProcessorSection
    {
        public  string Type { get; init; }

        [JsonPropertyName("special_tokens")]
        public  Dictionary<string, SpecialTokenDetails> SpecialTokens { get; init; }

        internal record SpecialTokenDetails
        {
            /// <summary>
            /// E.g. [CLS] or [SEP].
            /// </summary>
            public  string Id { get; init; }
        }
    }

    internal record ModelSection
    {
        public string? Type { get; init; }

        [JsonPropertyName("unk_token")]
        public  string UnkToken { get; init; }

        [JsonPropertyName("continuing_subword_prefix")]
        public  string ContinuingSubwordPrefix { get; init; }

        public  Dictionary<string, int> Vocab { get; set; }
    }
}
