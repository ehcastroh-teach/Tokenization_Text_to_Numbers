# Tokenization: Text to Numbers

A language model never reads letters or words - it reads a sequence of integers, and the tokenizer is the table that turns your text into them. This repo teaches that table from both ends: you build a byte-pair-encoding tokenizer by hand on a tiny corpus, then compare three production tokenizers on the same text to see why one sentence costs a different number of tokens in every model. It is written for self-learners and practitioners who pay for model calls and want to understand why prompts are billed in tokens.

## Learning Objectives

- Explain why a model reads token ids rather than characters or words, and what that costs in context, price and latency
- Build a byte-pair-encoding vocabulary by counting adjacent pairs and merging the most frequent
- Encode and decode new text with learned merges, and see how an unseen word splits into known pieces
- Measure how vocabulary size trades off against sequence length
- Compare tiktoken, Hugging Face `tokenizers` and SentencePiece, and explain why their token counts differ on code, numbers, names and non-English text
- Decide what to measure before trusting a per-token price

## Notebooks

Work through them in this order.

| Order | Notebook | What it covers |
|---|---|---|
| 0 | `orientation.ipynb` | What the repo teaches, an environment check, and your first tokenizer call (about 15 min) |
| 1 | `01_bpe_by_hand.ipynb` | Build a BPE vocabulary by hand, then encode, decode and measure vocabulary size against sequence length (about 45 min) |
| 2 | `02_three_real_tokenizers.ipynb` | Run the same text through three production tokenizers and learn what to measure before trusting a price (about 45 min) |

## How to Run

You need either Nix or `uv`. No API keys are needed anywhere in this repo, so there is nothing to put in a `.env` file.

With Nix:

```bash
nix develop
uv run jupyter lab
```

With `uv` alone:

```bash
uv sync
uv run jupyter lab
```

Open `orientation.ipynb` first. Its second Part checks that your environment is complete. The first call to tiktoken downloads one vocabulary file (`cl100k_base`), so the first run needs internet access; the file is cached afterwards.

## Key Concepts Glossary

- **Token** - the unit a model reads; a piece of text mapped to one integer id
- **Tokenizer** - the table and rules that convert text to token ids and back
- **Vocabulary** - the full set of pieces a tokenizer knows
- **Byte-pair encoding (BPE)** - a method that builds a vocabulary by repeatedly merging the most frequent adjacent pair
- **Merge** - one learned rule saying two adjacent pieces fuse into one
- **Byte-level** - a base alphabet of the 256 byte values, so no input is ever unknown
- **Context window** - the maximum number of tokens a model can take in one call

## Further Reading

- Neural Machine Translation of Rare Words with Subword Units
- A New Algorithm for Data Compression
- SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing
- Language Models are Unsupervised Multitask Learners

## Credits and Acknowledgements

Authors: Sennrich, Haddow and Birch for the subword-units paper that brought BPE to language models; Gage for the original compression algorithm; Kudo and Richardson for SentencePiece; Radford et al. for byte-level BPE in GPT-2. Libraries: tiktoken, Hugging Face `tokenizers`, SentencePiece and matplotlib.
