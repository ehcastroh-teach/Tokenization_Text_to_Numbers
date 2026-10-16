# Brief: 01_bpe_by_hand

**Filename**: `01_bpe_by_hand.ipynb`
**Estimated Completion**: 45 minutes
**Content shape**: concept

**One-sentence insight**: a tokenizer is a learned compression table, and byte-pair encoding builds it by repeatedly merging the most frequent adjacent pair, so what counts as "one token" is a fact about the training text, not about language.

**Learning Goals**:
1. Explain why a language model reads token ids, not characters or words, and what that costs (context, price, latency are all counted in tokens).
2. Build a byte-pair-encoding vocabulary by hand from a tiny corpus: count adjacent pairs, merge the most frequent, repeat for N merges.
3. Encode a new sentence with the learned merges and decode the ids back, and see where an unseen word gets split into pieces.
4. Measure how vocabulary size trades off against sequence length on the same text.

**Prerequisite Knowledge**: (1) Python basics, lists, dictionaries and loops. (2) What a language model is at the level of "it predicts the next piece of text".

**Target User**: a practitioner or decision maker who has used an LLM and wants to understand why prompts are billed in tokens and why some words are "expensive".

### Part 1: Why models read numbers
Prose: the model's input is a sequence of integers; the embedding table is indexed by them. Characters would give long sequences; whole words give a vocabulary that never ends (every typo, every name). Tokens are the compromise. Code: count characters, words and (later) tokens for the same paragraph; show the three lengths side by side. Concept Check on which of the three grows without bound.

### Part 2: Counting pairs and merging
Prose: BPE in four steps, starting from characters (plus an end-of-word marker). Code: a 30-word toy corpus; `pair_counts(corpus)`, `merge(corpus, pair)`, a loop that records each merge with its count. Print the merge table after 10, 20, 40 merges. Concept Check: given a tiny corpus, which pair merges first and why.

### Part 3: Encoding and decoding
Prose: applying the merges in learned order to new text; why an unseen word becomes known pieces; why decoding is a lookup and always round-trips. Code: `encode(text, merges)`, `decode(ids, vocab)`; run on a sentence that contains one unseen word and one common one. Show the pieces, then the ids. Concept Check: what happens to a word that shares no pieces with the corpus.

### Part 4: Vocabulary size against sequence length
Prose: more merges mean a bigger table and shorter sequences; the curve flattens. Code: sweep merges from 0 to 200, record vocabulary size and encoded length of a held-out paragraph, plot one line (matplotlib, saved to `images/`). Concept Check: why production vocabularies land in the tens of thousands rather than millions.

## Sources
- Sennrich, Haddow and Birch, "Neural Machine Translation of Rare Words with Subword Units", 2016. https://aclanthology.org/P16-1162/
- Gage, "A New Algorithm for Data Compression", 1994. https://www.derczynski.com/papers/archive/BPE_Gage.pdf
- Hugging Face LLM course, chapter on byte-pair encoding tokenization. https://huggingface.co/learn/llm-course/chapter6/5
