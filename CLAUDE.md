# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NLP HW2: implement a Byte Pair Encoding (BPE) tokenizer from scratch, train three
variants, and have an existing NER model judge them. Graded on tokenizer F1 on a
downstream binary NER task **plus** tokenizer efficiency (tokens/char) and encoding
speed (a weighted competition with trade-offs between the three). The full spec is
in `HW2 26.pdf` (Hebrew); `README.md` is the English working guide.

## The one file you write

**`code/bpe_tokenizer.py`** — the `BPETokenizer(BaseTokenizer)` class. You implement
`train`, `encode`, `decode`, and set `self.space_token` (currently `None`, which
fails the submission check). You may add helper modules **inside `code/`**, plus your
own scripts (e.g. edit `train_tokenizer.py` / `test_tokenizer.py`).

**Do not edit or submit** these — the grader runs the *original* versions and your
pickles are loaded against them:
`code/base_tokenizer.py`, `code/train_ner_model.py`, `generate_tokenizers.py`,
`check_submission.py`. Treat them as read-only contracts.

## Hard requirements the submission checker enforces

`check_submission.py` rejects the submission unless:
- **`space_token` is not `None`** — pick a marker (e.g. `▁` or `_`).
- **At least one bigram token exists**: a non-special vocab token whose surface, after
  replacing `space_token` with a real space, contains an internal space between two
  words (`' ' in surface.strip()`). A tokenizer with no bigram is disqualified. This
  is the assignment's twist on standard BPE — your merges must be able to cross the
  word boundary (space marker) at least once.
- `encode`/`decode` run on `"Hello world!"` without raising.

## Two non-obvious contracts (read before implementing)

**1. Pickle reproducibility.** `BaseTokenizer.save` does `pickle.dump(self)`; loaders
(`generate_tokenizers.py`, `check_submission.py`, `train_ner_model.py`) only do
`BaseTokenizer.load` / `pickle.load` after putting `code/` on `sys.path`. So every
class/attribute referenced by a pickled tokenizer must be importable by **bare module
name** from `code/` — use flat imports like `from base_tokenizer import BaseTokenizer`,
never `from code.base_tokenizer import ...`, and don't stash lambdas/closures on the
instance (unpicklable). The loader knows only `BaseTokenizer`, not your subclass name,
beyond what's importable from `code/`.

**2. F1 depends on `decode` faithfulness, not just vocab.** `train_ner_model.py`'s
`NERDataset` aligns tokens to words by **decoding growing prefixes** (`decode(ids[:i+1])`)
and diffing string lengths to get per-token char spans, then assigning each word's
label to its first overlapping token. If `decode` doesn't reconstruct text at
consistent char offsets, alignment drifts and F1 drops — independent of how good the
vocabulary is. `test_tokenizer.py`'s reconstruction check compares ignoring spaces, so
it won't catch offset drift; the NER F1 will.

Reserved IDs are fixed by `BaseTokenizer.__init__`: `[PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3`
(NER pads `input_ids` with `0`). Keep these; build your learned vocab above them.

## The three tokenizers

`generate_tokenizers.py` trains all three reproducibly (this is what the grader runs):
- **tokenizer_1** → `domain_1_train.txt`, evaluated on NER domain 1.
- **tokenizer_2** → `domain_2_train.txt`, evaluated on NER domain 2.
- **tokenizer_3** → both domains combined (naive baseline), evaluated on a **hidden,
  different domain** — the open problem is making it robust (data mix / strategy via
  `--train_files_3`). Domain 1 is noisy tweets; domain 2 differs.

`vocab_size` defaults: `5000` in `generate_tokenizers.py` and `train_tokenizer.py`,
but `10000` in the `BPETokenizer.__init__` signature. The grader uses
`generate_tokenizers.py` defaults unless your `train_commands.txt` overrides them.

## Commands

Run everything from the repo root. `bash init.sh` once to set up the `uv` env
(installs torch from the CUDA 12.6 index for the course's Tesla M60 GPUs).

```bash
# Reproducibly train all three tokenizers -> trained_tokenizers/tokenizer_{1,2,3}.pkl
uv run python generate_tokenizers.py
uv run python generate_tokenizers.py --vocab_size 8000 --train_files_3 domain_1_train.txt domain_2_train.txt

# Train one tokenizer (debugging)
uv run python code/train_tokenizer.py --domain_file data/domain_1_train.txt --output_dir trained_tokenizers --vocab_size 5000

# Tokenizer-only metrics: efficiency (tokens/char), encoding speed, reconstruction
uv run python code/test_tokenizer.py --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/domain_1_train.txt --test_file data/domain_1_dev.txt

# Train + eval the NER model -> prints dev F1 (the score that's graded). GPU recommended.
uv run python code/train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged

# Validate the final zip end-to-end (structure + smoke-train NER on domains 1 & 2)
uv run python check_submission.py HW2_123456789.zip
```

`train_ner_model.py` hyperparameters (seed 42, 20 epochs, lr 0.01, batch 32) are
**fixed** — only the `--*_file` paths may change, since every student's tokenizer is
judged against an identical NER model. There is no test suite; "tests" = running
`test_tokenizer.py` (efficiency/speed) and `train_ner_model.py` (F1) per domain.

## Data formats

- `data/domain_{1,2}_{train,dev}.txt`: one raw sentence per line (tokenizer training).
- `data/ner_data/*.tagged`: `token\t tag`, blank line separates sentences. NER label
  is binarized in `read_ner_data` as `1 if tag != '0' else 0`.

## Submission (format mismatch = grade 0)

Zip named `HW2_<id>.zip` / `HW2_<id1>_<id2>.zip` containing: `code/` (your
`bpe_tokenizer.py` + helpers + your scripts, **not** the read-only provided files),
`trained_tokenizers/tokenizer_{1,2,3}.pkl`, `train_commands.txt` (exact training
commands at the zip root, for reproduction), and `report_<id>.pdf` (max 1 A4 page,
Arial 10, 2.54cm margins). The VM has a 10-hour total runtime budget (`time_left`).
