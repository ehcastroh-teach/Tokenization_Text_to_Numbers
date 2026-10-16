# Brief: 02_three_real_tokenizers

**Filename**: `02_three_real_tokenizers.ipynb`
**Estimated Completion**: 45 minutes
**Content shape**: decision

**One-sentence insight**: the same sentence costs a different number of tokens in every model, and the differences are largest exactly where it matters to a practitioner: code, numbers, names and non-English text.

**Learning Goals**:
1. Run the same text through three production tokenizers (tiktoken cl100k_base, a Hugging Face `tokenizers` BPE such as GPT-2, and a SentencePiece model) and read the pieces and ids each one produces.
2. Explain the three design choices that make them differ: byte-level versus character-level base alphabet, how whitespace is attached, and vocabulary size.
3. Measure token counts for English prose, Python code, a table of numbers and a Spanish paragraph, and read the differences as cost and context.
4. Decide, for a given workload, what to measure before trusting a per-token price.

**Prerequisite Knowledge**: (1) The hand-built BPE from `01_bpe_by_hand.ipynb`. (2) Python basics.

**Target User**: a practitioner who pays for model calls and wants to know why bills and context limits behave the way they do.

### Part 1: Loading three tokenizers
Prose: what each library is, what the vocabulary file contains, and that none of them need a model download beyond the vocabulary. Code: load `tiktoken.get_encoding("cl100k_base")`, `tokenizers.Tokenizer.from_pretrained("gpt2")`, and a SentencePiece model trained in the notebook on a small corpus (so there is no network dependency for the third). Print vocabulary sizes. Concept Check on which tokenizer would never produce an unknown-token id and why.

### Part 2: One sentence, three cuts
Prose: how to read a token listing; the leading-space convention (a space travels with the word after it); the byte-level alphabet. Code: `show(text)` that prints each tokenizer's pieces and ids on aligned rows for the sentence "Tokenization isn't magic." and for a sentence with a rare name. Concept Check: why "isn't" splits at the apostrophe in one tokenizer and not another.

### Part 3: What gets expensive
Prose: the four workloads where counts diverge, and why (digits split one or three at a time, indentation in code, accents and non-Latin scripts). Code: count tokens per tokenizer for four fixed samples; a grouped bar chart (matplotlib, saved to `images/`). Concept Check: estimate the cost ratio of a Spanish prompt to its English translation from the chart.

### Part 4: Measuring before trusting a price
Prose: a per-token price is meaningless without the tokenizer; a short checklist for a workload: sample real inputs, count with the target model's tokenizer, check the tails. Code: a `count_budget(samples, tokenizer)` that returns mean, p95 and max tokens, run on the four samples. Concept Check: which statistic matters for a context-limit decision, mean or max, and why.

## Sources
- tiktoken README and the cl100k_base encoding. https://github.com/openai/tiktoken
- Hugging Face `tokenizers` documentation, "Quicktour" and the byte-level BPE model. https://huggingface.co/docs/tokenizers/quicktour
- Kudo and Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", 2018. https://aclanthology.org/D18-2012/
- Radford et al., "Language Models are Unsupervised Multitask Learners", 2019, section 2.2 on byte-level BPE. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
