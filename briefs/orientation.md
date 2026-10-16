# Brief: orientation

**Filename**: `orientation.ipynb`
**Estimated Completion**: 15 minutes
**Content shape**: concept

**Target User**: anyone opening the repo for the first time.

### Part 1: What this repo teaches
Prose: one paragraph; two notebooks, from a hand-built byte-pair encoder to three production tokenizers. Learning outcomes list. Coverage map as a two-row table (notebook, what you can do after it).

### Part 2: Verifying your environment
Code: import `tiktoken`, `tokenizers`, `sentencepiece`, `matplotlib`; print versions; on failure print the exact `uv sync` instruction. No API keys are needed anywhere in this repo, and the cell says so.

### Part 3: Your first call
Code: encode the sentence "Tokenization isn't magic." with `tiktoken.get_encoding("cl100k_base")`, print the pieces and the ids, decode them back. One sentence on what to notice: the space rides with the word after it.

## Sources
- tiktoken README. https://github.com/openai/tiktoken
