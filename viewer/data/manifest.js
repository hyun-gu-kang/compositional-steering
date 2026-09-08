const MANIFEST = {
  "models": [
    "llama_8b",
    "llama_70b",
    "qwen_14b",
    "qwen_32b"
  ],
  "langPairs": [
    "ar-en",
    "de-en",
    "es-en",
    "fr-en",
    "ja-en",
    "ko-en",
    "pt-en",
    "ru-en",
    "zh-en"
  ],
  "single": {
    "language": {
      "llama_8b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "llama_70b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "qwen_14b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "qwen_32b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ]
    },
    "concise": [
      "llama_8b",
      "llama_70b",
      "qwen_14b",
      "qwen_32b"
    ],
    "jailbreak": [
      "llama_8b",
      "llama_70b",
      "qwen_14b",
      "qwen_32b"
    ]
  },
  "double": {
    "lang:jb": {
      "llama_8b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "llama_70b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "qwen_14b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "qwen_32b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ]
    },
    "lang:len": {
      "llama_8b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "llama_70b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "qwen_14b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "qwen_32b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ]
    }
  },
  "triple": {
    "lang:jb:len": {
      "llama_8b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "llama_70b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "qwen_14b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ],
      "qwen_32b": [
        "ar-en",
        "de-en",
        "es-en",
        "fr-en",
        "ja-en",
        "ko-en",
        "pt-en",
        "ru-en",
        "zh-en"
      ]
    }
  },
  "_layersAlphas": {
    "llama_8b": {
      "layers": [
        "layer_6",
        "layer_14",
        "layer_22",
        "layer_30"
      ],
      "alphas": [
        "1.0",
        "2.0",
        "4.0",
        "6.0",
        "8.0"
      ]
    },
    "llama_70b": {
      "layers": [
        "layer_16",
        "layer_36",
        "layer_56",
        "layer_76"
      ],
      "alphas": [
        "1.0",
        "2.0",
        "4.0",
        "6.0",
        "8.0"
      ]
    },
    "qwen_14b": {
      "layers": [
        "layer_10",
        "layer_22",
        "layer_34",
        "layer_46"
      ],
      "alphas": [
        "10.0",
        "20.0",
        "40.0",
        "60.0",
        "80.0"
      ]
    },
    "qwen_32b": {
      "layers": [
        "layer_12",
        "layer_28",
        "layer_44",
        "layer_60"
      ],
      "alphas": [
        "10.0",
        "20.0",
        "40.0",
        "60.0",
        "80.0"
      ]
    }
  }
};
