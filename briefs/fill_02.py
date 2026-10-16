#!/usr/bin/env python3
"""Fill the scaffolded 02_three_real_tokenizers.ipynb with prose and code.

Run from the repo root with the system python: python3 -I briefs/fill_02.py
Numbers quoted in prose are filled in by briefs/patch_02_numbers.py after the
notebook has been executed, so nothing here claims a measured value.
"""
import json
from pathlib import Path

NB = Path("02_three_real_tokenizers.ipynb")
nb = json.loads(NB.read_text())
cells = nb["cells"]


def md(i, text):
    cells[i]["cell_type"] = "markdown"
    cells[i]["source"] = text
    cells[i].pop("outputs", None)
    cells[i].pop("execution_count", None)


def code(i, text):
    cells[i]["cell_type"] = "code"
    cells[i]["source"] = text
    cells[i]["outputs"] = []
    cells[i]["execution_count"] = None
    cells[i].setdefault("metadata", {})


def insert_code_after(i, text):
    """Insert a code cell after index i and return its index."""
    new = {"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None}
    cells.insert(i + 1, new)
    return i + 1


# ---- cover -----------------------------------------------------------------
md(0, """<hr style="border: 6px solid#003262;" />

<div align="center">
    <img src="images/thumbnails/02_hero.svg" align="center" width="80%">
</div>

<br>

# THREE REAL TOKENIZERS: ONE SENTENCE AND THREE BILLS

<br>

**About:** Run the same text through three production tokenizers and read why the token counts differ, where the differences get expensive, and what to measure before trusting a per-token price.

**Learning Goals:**
- Load three production tokenizers (tiktoken `cl100k_base`, the GPT-2 byte-level BPE from the `tokenizers` library, and a SentencePiece model trained in this notebook) and read the pieces and ids each one produces.
- Explain the three design choices that make them differ: the base alphabet, how whitespace is attached, and the vocabulary size.
- Measure token counts for English prose, Python code, a table of numbers and a Spanish paragraph, and read the differences as cost and context.
- Decide, for a given workload, what to measure before trusting a per-token price.

**Keywords:** tokenization, byte pair encoding, tiktoken, sentencepiece, token cost, context window

**Prerequisite Knowledge:** (1) The hand-built byte-pair encoder from `01_bpe_by_hand.ipynb`, (2) Python basics

**Target User:** A practitioner who pays for model calls, or sets a budget for someone who does, and wants to know why bills and context limits behave the way they do.
""")

# ---- setup -----------------------------------------------------------------
code(3, """# Setup - imports and environment check
# Dependencies verified against pyproject.toml and uv.lock, 2026-10-16.
# Re-check at https://github.com/ehcastroh-teach/Tokenization_Text_to_Numbers/blob/main/pyproject.toml
import os
import sys
from pathlib import Path

import tiktoken                      # OpenAI's tokenizer library; downloads one vocabulary file on first use
from tokenizers import Tokenizer     # Hugging Face tokenizers; the GPT-2 vocabulary comes from the hub
import sentencepiece as spm          # Google's SentencePiece; we train a small model locally
import matplotlib
matplotlib.use("Agg")                # render to files, not a window, so the notebook runs anywhere
import matplotlib.pyplot as plt

Path("images").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
print(f"Python: {sys.version.split()[0]}")
print(f"tiktoken {tiktoken.__version__}, sentencepiece {spm.__version__}")
""")

# ---- Part 1: loading three tokenizers (decision shape) ---------------------
md(4, cells[4]["source"].replace("{KEY WORD}", "LOADING").replace("{NN}", "02") if isinstance(cells[4]["source"], str) else "".join(cells[4]["source"]).replace("{KEY WORD}", "LOADING").replace("{NN}", "02"))
md(5, """PART 1 - Scenario framing

A team is about to pick a model for a document-heavy product. Three vendors quote a price per million tokens, and the three prices look close. Nobody on the team has checked what a "token" is for each vendor. The team must decide: can the three quotes be compared as they stand, or does each one need its own tokenizer run on real documents first?

The constraint that decides: the tokenizer is part of the model, not a detail of the API. Each vendor's vocabulary was learned from different text, so the same document is a different number of tokens at each one.
""")
md(6, """PART 1 - Candidate options

**Option A: Compare the quoted prices directly.**
Treat a token as a unit and rank the vendors by price per million. Advantages: no work, and it is what the pricing pages invite. Trade-off: the unit is not the same across vendors, so the ranking can be wrong by a factor that depends on your text.

**Option B: Load each vendor's tokenizer and count real inputs.**
Every production tokenizer ships as a vocabulary file plus a small library: tiktoken for OpenAI models, the `tokenizers` library for most open models, SentencePiece for the models that use it. Advantages: the count is exact, and it takes seconds. Trade-off: you need a sample of real inputs, and the vocabulary file for a closed model may not be published.

**Option C: Estimate with a rule of thumb (about four characters per token in English).**
Advantages: no download, good enough for a first budget. Trade-off: the rule is for English prose and breaks on code, numbers and other languages, which is exactly where the bills diverge.
""")
md(7, """PART 1 - Tradeoff analysis

| Axis | A: compare quotes | B: count with each tokenizer | C: characters divided by four |
|---|---|---|---|
| Effort | none | minutes | none |
| Exact for your inputs | no | yes | no |
| Works for code, numbers, other languages | no | yes | poorly |
| Needs the vendor's vocabulary | no | yes | no |
| Catches the expensive tail | no | yes | no |
""")
md(8, """PART 1 - Decision + Rationale

**Option B wins in the stated scenario.**
The constraint that decides: the quotes are in different units, and only the tokenizer converts your documents into each vendor's unit.
Why the others fall short: Option A compares numbers that are not comparable, and Option C is calibrated on English prose while the product is document-heavy and will hit tables and code.
What would flip this: if the inputs were short English chat messages and the budget was a rough early estimate, Option C would be good enough to start, and Option B could wait until the first bill.

The code below loads the three tokenizers this notebook uses. Two come from vocabulary files their libraries fetch once and cache; the third is trained here on a small corpus so that nothing in Parts 2 to 4 depends on a network connection beyond the first run.
""")
i = insert_code_after(8, """# Load three production-style tokenizers.
# tiktoken: the encoding behind GPT-4-era OpenAI models. One ~1.7 MB vocabulary file, cached after the first call.
enc_tik = tiktoken.get_encoding("cl100k_base")

# Hugging Face tokenizers: GPT-2's byte-level BPE, the ancestor of most open-model tokenizers. Fetched from the hub once.
tok_gpt2 = Tokenizer.from_pretrained("gpt2")

# SentencePiece: trained right here on a small corpus, so it is tiny and needs no download.
# Real SentencePiece models (Llama, T5, Gemma) are trained the same way on far more text.
corpus = '''The quick brown fox jumps over the lazy dog. A tokenizer is a learned table.
Common words stay whole, rarer ones get split into pieces the model has seen before.
Models read numbers, not letters. You pay, wait and run out of context in tokens, not words.
def count_tokens(text): return len(encode(text))
for i in range(10): print(i, i * i, i ** 3)
El modelo lee numeros, no letras. Pagas por token, no por palabra.
La tokenizacion no es magia. Es una tabla aprendida de un corpus.
2024-01-15, 3.14159, 1,000,000, 42, 0.001, 99.5 percent, 7 days, 1 to 3 days.
'''
Path("data/corpus.txt").write_text(corpus)
spm.SentencePieceTrainer.train(
    input="data/corpus.txt", model_prefix="data/spm_small", vocab_size=300,  # 300 is close to the most this corpus supports
    model_type="bpe", character_coverage=1.0, minloglevel=2,
)
sp = spm.SentencePieceProcessor(model_file="data/spm_small.model")

print(f"cl100k_base vocabulary: {enc_tik.n_vocab:,} tokens")
print(f"GPT-2 vocabulary:       {tok_gpt2.get_vocab_size():,} tokens")
print(f"SentencePiece (ours):   {sp.get_piece_size():,} tokens")
""")
# concept check for part 1 (index shifted by 1)
md(i + 1, "".join(cells[i + 1]["source"]).split("> **Scenario extension")[0] + """> **Scenario extension (Part 1):** The team now also wants to price a model whose vendor publishes no vocabulary file at all, only a counting endpoint in its API that returns the token count for a request. The endpoint is rate limited and costs a fraction of a cent per call.

> **Question:** What is the best way to get a reliable count for a budget without blowing the rate limit?

> (a) Fall back to characters divided by four for that vendor only
> (b) Sample a few hundred representative inputs, count them through the endpoint once, and keep the counts next to the other vendors' local counts
> (c) Use one of the local tokenizers as a stand-in, since BPE vocabularies are similar enough
> (d) Skip that vendor; a model without a published tokenizer cannot be budgeted

<!-- ANSWER (for reviewer): (b). A sample through the real endpoint is exact for that vendor and cheap at a few hundred calls. (a) reintroduces the English-prose rule on a document-heavy workload, which is the error the Part is about. (c) is wrong because vocabularies differ most on exactly the inputs that matter (code, numbers, other languages), so a stand-in hides the tail. (d) is a non-answer: the count is available, it just has to be sampled. -->
""")

# ---- Part 2: one sentence, three cuts --------------------------------------
p2 = i + 3  # header of Part 2
md(p2, "".join(cells[p2]["source"]).replace("{KEY WORD}", "READING").replace("{NN}", "02"))
md(p2 + 1, """PART 2 - Scenario framing

A reviewer is reading a token listing for the first time and sees three different cuts of the same sentence, with odd characters in one of them (a `Ġ` in front of some pieces) and a `▁` in another. The team must decide: are these three tokenizers doing different things, or the same thing written three ways?

The constraint that decides: where each tokenizer puts the space. All three attach the space to the word after it; they only differ in how they write that down.
""")
md(p2 + 2, """PART 2 - Candidate options

**Option A: They are different algorithms and the listings cannot be compared.**
Advantages: cautious. Trade-off: wrong, and it stops the reviewer from reading the one thing the listings share.

**Option B: They are the same idea (merge frequent pairs, attach the space to the following word) with three notations.**
tiktoken prints the piece with a leading space (`' isn'`). GPT-2's byte-level BPE writes the space as `Ġ` because every byte, including the space, has a printable stand-in. SentencePiece writes it as `▁` because it treats the input as a raw stream and marks word starts explicitly. Advantages: once you know the notation, every listing reads the same way. Trade-off: the vocabularies still differ, so the pieces differ even when the notation is understood.

**Option C: Normalize the three listings to plain words before comparing.**
Advantages: readable. Trade-off: it throws away the split points, which is the information a token listing exists to show.
""")
md(p2 + 3, """PART 2 - Tradeoff analysis

| Axis | cl100k_base (tiktoken) | GPT-2 (tokenizers) | SentencePiece (ours) |
|---|---|---|---|
| Base alphabet | 256 bytes | 256 bytes (printable stand-ins, `Ġ` for space) | Unicode characters of the training text |
| Space handling | attached to the following word, shown as a leading space | attached to the following word, shown as `Ġ` | attached to the following word, shown as `▁` |
| Vocabulary size | about 100k | about 50k | 300 here; tens of thousands in real models |
| Can produce an unknown token | never (bytes cover everything) | never | yes, for a character outside the training text (unless byte fallback is on) |
| Digits | up to 3 at a time | one at a time | whatever the corpus taught |
""")
md(p2 + 4, """PART 2 - Decision + Rationale

**Option B wins in the stated scenario.**
The constraint that decides: the notation differs, the rule does not. In all three the space rides with the word after it, which is why `' isn'` or `Ġisn` is one piece and `isn` alone is a different one.
Why the others fall short: Option A denies a real shared mechanism and leaves the reviewer unable to read any listing; Option C erases the split points.
What would flip this: if a tokenizer attached the space to the word before it (a few older ones did), the listings would genuinely read differently, and the reviewer would need that rule stated first.

`show()` below prints the three cuts on aligned rows so the notation differences sit side by side.
""")
j = insert_code_after(p2 + 4, """def pieces_tik(text):
    return [enc_tik.decode([t]) for t in enc_tik.encode(text)]

def pieces_gpt2(text):
    return tok_gpt2.encode(text).tokens          # 'Ġ' marks a space in GPT-2's byte-level alphabet

def pieces_spm(text):
    return sp.encode(text, out_type=str)         # '▁' marks a word start in SentencePiece

def show(text):
    rows = [("cl100k_base", pieces_tik(text), enc_tik.encode(text)),
            ("GPT-2", pieces_gpt2(text), tok_gpt2.encode(text).ids),
            ("SentencePiece", pieces_spm(text), sp.encode(text))]
    print(repr(text))
    for name, pcs, ids in rows:
        print(f"  {name:<14}{len(ids):>3} tokens  pieces: {pcs}")
        print(f"  {'':<14}{'':>3}         ids:    {ids}")

show("Tokenization isn't magic.")
print()
show("Ask for Dr. Okonkwo-Baptiste in Reykjavik.")
""")
md(j + 1, "".join(cells[j + 1]["source"]).split("> **Scenario extension")[0] + """> **Scenario extension (Part 2):** The reviewer encodes the single word `isn't` on its own, with no leading space, in cl100k_base, and gets a different first piece from the one in the sentence above.

> **Question:** Why does the same word get a different first piece when it starts the text?

> (a) The tokenizer is non-deterministic and the ids vary between calls
> (b) The space before `isn` in the sentence is part of the piece `' isn'`, so the bare word has to use a different piece that has no leading space
> (c) Apostrophes are handled by a separate rule that only fires mid-sentence
> (d) Words at the start of a text are always encoded character by character

<!-- ANSWER (for reviewer): (b). The leading space is literally inside the vocabulary entry, so ' isn' and 'isn' are two different tokens. (a) is false: encoding is a deterministic lookup. (c) invents a rule; the apostrophe split happens in both cases. (d) is false: a common word at the start of a text is still one or two pieces, it just uses the no-space variants. -->
""")

# ---- Part 3: what gets expensive ------------------------------------------
p3 = j + 3
md(p3, "".join(cells[p3]["source"]).replace("{KEY WORD}", "PRICING").replace("{NN}", "02"))
md(p3 + 1, """PART 3 - Scenario framing

The product has four kinds of input: English prose, Python code, tables of numbers, and Spanish text from a second market. The English estimate came in on budget; the first real bill did not. The team must decide: which of the four inputs is driving the overrun, and is it the same input at every vendor?

The constraint that decides: a vocabulary is a record of what was common in its training text. Anything that was rare there (indentation, long digit strings, accented words) is cut into more pieces, so it costs more, and each vocabulary was rare in different places.
""")
md(p3 + 2, """PART 3 - Candidate options

**Option A: Assume prose dominates and re-estimate with a larger safety margin.**
Advantages: one number, easy to defend. Trade-off: the margin is a guess, and it will be wrong in different directions for the four inputs.

**Option B: Count each input type with each tokenizer and read the ratios.**
Advantages: shows exactly which input is expensive and whether the cost is a property of the input or of one vendor. Trade-off: needs representative samples of each input type.

**Option C: Reformat the inputs to be cheaper (strip indentation, round numbers, translate Spanish to English before sending).**
Advantages: can genuinely cut the bill. Trade-off: changes the input the model sees, which can change the answer; translation costs tokens of its own.
""")
md(p3 + 3, """PART 3 - Tradeoff analysis

| Input | Why it can be expensive | Which tokenizers it hits hardest |
|---|---|---|
| English prose | rarely; common words are single tokens | none |
| Python code | indentation is runs of spaces, and each run is its own token unless the vocabulary learned it; symbols rarely merge | older vocabularies trained mostly on prose |
| Tables of numbers | digits are cut one or three at a time, never as the number you see | one-digit tokenizers (GPT-2 style) |
| Spanish | accented letters and word endings that were rare in an English-heavy corpus split into pieces | any vocabulary trained mostly on English |
""")
md(p3 + 4, """PART 3 - Decision + Rationale

**Option B wins in the stated scenario.**
The constraint that decides: the overrun is a property of the input-vocabulary pair, so only a count per pair can locate it.
Why the others fall short: Option A hides the cause behind a margin; Option C may be the right fix later, but it cannot be chosen until Option B says which input to fix.
What would flip this: if the product took one kind of input only, a single measured ratio would do, and the chart below would be one bar.

The chart below counts the four samples with the three tokenizers. The samples are short and fixed so the notebook runs in a second; a real budget uses hundreds of real inputs (Part 4).
""")
k = insert_code_after(p3 + 4, """samples = {
    "English prose": "The report is the product. People judge an assessment by the page they get back, not by the model behind it.",
    "Python code":   "def score(rows):\\n    total = 0\\n    for r in rows:\\n        if r.get('valid'):\\n            total += r['points'] * 1.5\\n    return total\\n",
    "Numbers":       "2024-01-15, 1,000,000, 3.14159, 0.001, 42, 99.5, 7, 1,010, 333, 0.891, 2021, 46, 43",
    "Spanish":       "El informe es el producto. La gente juzga una evaluacion por la pagina que recibe, no por el modelo que hay detras.",
}
counters = {
    "cl100k_base": lambda t: len(enc_tik.encode(t)),
    "GPT-2":       lambda t: len(tok_gpt2.encode(t).ids),
    "SentencePiece (ours)": lambda t: len(sp.encode(t)),
}
counts = {name: {tk: f(text) for tk, f in counters.items()} for name, text in samples.items()}
chars = {name: len(text) for name, text in samples.items()}

print(f"{'input':<16}{'chars':>6}" + "".join(f"{tk:>22}" for tk in counters))
for name in samples:
    print(f"{name:<16}{chars[name]:>6}" + "".join(f"{counts[name][tk]:>22}" for tk in counters))
print()
print("characters per token (higher is cheaper):")
for name in samples:
    print(f"  {name:<16}" + "".join(f"{chars[name] / counts[name][tk]:>22.1f}" for tk in counters))

# Grouped bars: one group per input, one bar per tokenizer.
fig, ax = plt.subplots(figsize=(8, 4))
names = list(samples); tks = list(counters); w = 0.26
for b, tk in enumerate(tks):
    xs = [x + (b - 1) * w for x in range(len(names))]
    ax.bar(xs, [counts[n][tk] for n in names], width=w, label=tk, color=["#1C1A17", "#8F8B7F", "#A8561A"][b])
ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
ax.set_ylabel("tokens for the sample"); ax.set_title("Same four samples, three tokenizers")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("images/02_tokens_per_workload.png", dpi=150)
plt.close(fig)
print("saved images/02_tokens_per_workload.png")
""")
md(k + 1, "".join(cells[k + 1]["source"]).split("> **Scenario extension")[0] + """> **Scenario extension (Part 3):** The team adds a fifth sample, a JSON payload with long hexadecimal ids, and finds it costs more tokens per character than any of the four above, at every tokenizer.

> **Question:** What is the most likely reason the hex ids are expensive everywhere, and what is the cheapest honest fix?

> (a) Hex strings are random to every vocabulary, so they split into short pieces at all three; replace the ids with short integers before sending and map them back afterwards
> (b) JSON braces are expensive; switch the payload to YAML
> (c) The tokenizers are mis-counting; use characters divided by four for this input
> (d) Hex is cheaper at a tokenizer with a bigger vocabulary; switch vendor

<!-- ANSWER (for reviewer): (a). A 32-character hex id has no frequent substrings in any natural text corpus, so every vocabulary cuts it into many short pieces; swapping ids for short surrogates removes the cost without changing the meaning the model needs. (b) misattributes: braces are a handful of tokens, the ids are dozens each. (c) ignores an exact measurement in favour of a heuristic. (d) is wrong because vocabulary size barely helps for strings that were never frequent in training, which the chart's Numbers column already hints at. -->
""")

# ---- Part 4: measuring before trusting a price ----------------------------
p4 = k + 3
md(p4, "".join(cells[p4]["source"]).replace("{KEY WORD}", "MEASURING").replace("{NN}", "02"))
md(p4 + 1, """PART 4 - Scenario framing

The budget owner wants one number per vendor to put in a spreadsheet. The engineer has a folder of real inputs. The team must decide: which number represents the workload, the average tokens per input or the largest, and does one number even do the job?

The constraint that decides: price scales with the sum, so the mean is the number for cost; a context window is a ceiling, so the maximum (or a high percentile) is the number for whether requests will fail.
""")
md(p4 + 2, """PART 4 - Candidate options

**Option A: Report the mean tokens per input.**
Advantages: multiplies straight into a monthly cost. Trade-off: says nothing about the tail, and one input over the context limit is a failed request, not a slightly more expensive one.

**Option B: Report the maximum.**
Advantages: tells you whether anything will be cut off. Trade-off: overstates the bill by the ratio of max to mean, which can be large.

**Option C: Report mean, 95th percentile and max, per vendor, from a sample of real inputs.**
Advantages: the mean prices the bill, the p95 sizes a safe context budget, the max flags the inputs that need special handling. Trade-off: three numbers instead of one, and the sample has to be representative.
""")
md(p4 + 3, """PART 4 - Tradeoff analysis

| Decision | Statistic that answers it | Why the others mislead |
|---|---|---|
| Monthly cost | mean (times volume) | max overstates, p95 overstates less but still |
| Will requests fail on the context limit | max, then p95 | the mean can be far under the limit while the tail is over it |
| Which inputs need special handling | the inputs above p95 | a single summary number hides them |
""")
md(p4 + 4, """PART 4 - Decision + Rationale

**Option C wins in the stated scenario.**
The constraint that decides: cost and context are two different questions, and no single statistic answers both.
Why the others fall short: Option A can pass a budget review and then fail in production on the long tail; Option B protects against failures and overstates the spend.
What would flip this: if every input were the same length (a fixed form), mean, p95 and max would coincide and any one of them would do.

`count_budget()` below returns the three numbers for a list of inputs and one tokenizer. It is run here on the four samples from Part 3, which is far too few for a real budget; the point is the shape of the output.
""")
m = insert_code_after(p4 + 4, """def count_budget(texts, count):
    \"\"\"Return mean, 95th percentile and max token counts for a list of inputs.

    `count` is a function from text to token count, so the same budget can be run per vendor.
    \"\"\"
    ns = sorted(count(t) for t in texts)
    p95 = ns[min(len(ns) - 1, int(round(0.95 * (len(ns) - 1))))]   # nearest-rank percentile; fine for a budget
    return {"n": len(ns), "mean": sum(ns) / len(ns), "p95": p95, "max": ns[-1]}

texts = list(samples.values())
for tk, f in counters.items():
    b = count_budget(texts, f)
    print(f"{tk:<22} n={b['n']}  mean={b['mean']:.1f}  p95={b['p95']}  max={b['max']}")
""")
md(m + 1, "".join(cells[m + 1]["source"]).split("> **Scenario extension")[0] + """> **Scenario extension (Part 4):** A real sample of 2,000 support tickets gives mean 310, p95 1,900 and max 41,000 tokens at the chosen vendor, whose context limit is 32,000 tokens.

> **Question:** What does this sample say about the budget and the design?

> (a) The budget is fine; 310 is far under the limit, and the max is an outlier that can be ignored
> (b) Price the bill on the mean, size the normal path for about 2,000 tokens, and route the few inputs above the limit through a separate chunking step, because at least one real ticket cannot be sent whole
> (c) Set the context budget to 41,000 and switch to a vendor with a larger window
> (d) Price the bill on the max to be safe

<!-- ANSWER (for reviewer): (b). The mean prices cost, p95 sizes the normal path, and the max proves the long tail exists and needs a design, not a margin. (a) ignores a real failing input. (c) changes vendor for one ticket in two thousand and resizes everything for the tail. (d) overstates the monthly bill by more than a hundred times. -->
""")

# ---- Sources Consulted -----------------------------------------------------
for idx, c in enumerate(cells):
    if c["cell_type"] == "markdown" and "Sources Consulted" in "".join(c["source"]):
        md(idx, """## Sources Consulted

- [tiktoken README and the `cl100k_base` encoding](https://github.com/openai/tiktoken). Retrieved 2026-10-16.
- [Hugging Face `tokenizers` documentation, Quicktour and the byte-level BPE model](https://huggingface.co/docs/tokenizers/quicktour). Retrieved 2026-10-16.
- [Kudo and Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", EMNLP 2018](https://aclanthology.org/D18-2012/). Retrieved 2026-10-16.
- [Radford et al., "Language Models are Unsupervised Multitask Learners", 2019, section 2.2 on byte-level BPE](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Retrieved 2026-10-16.

Claim strength: the mechanics (byte-level alphabet, space attachment, vocabulary sizes) are documented behaviour of the named libraries and were verified by running them in this notebook. The pricing and context guidance is practitioner practice, not a published result.
""")

# normalise sources to lists of lines for nbformat
for c in cells:
    if isinstance(c["source"], str):
        c["source"] = c["source"].splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {NB} with {len(cells)} cells")
