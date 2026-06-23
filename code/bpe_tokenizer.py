"""
Byte-level Byte-Pair-Encoding (BPE) tokenizer for HW2.

Design decisions (and *why* they matter for this assignment):

1. BYTE-LEVEL ALPHABET.
   The base vocabulary is the 256 possible bytes, mapped to printable Unicode
   characters using the standard GPT-2 `bytes_to_unicode` trick. Consequence:
   *no input character is ever out-of-vocabulary* -- emojis, curly quotes,
   foreign scripts in the hidden domain all decompose to bytes we already know.
   This is the robustness the three-dataset (one hidden) setup demands.

2. OFFSET-FAITHFUL `decode`.
   `train_ner_model.py` recovers each token's character span from
   `len(decode(ids[:i+1])) - len(decode(ids[:i]))`, then aligns those spans to
   word spans taken from the *raw* text. So `encode` must be a faithful, in-order
   segmentation of the text: concatenating token surfaces (with the space marker
   turned back into a real space) reproduces the input exactly. We therefore do
   NOT lowercase, NOT add a leading space marker, and NOT inject [BOS]/[EOS] into
   the stream -- any of those would shift every downstream offset and silently
   wreck F1 (the reconstruction test in test_tokenizer.py strips spaces and would
   not catch it; the NER F1 would).

3. CONTROLLED CROSS-WORD BIGRAMS.
   The spec requires >= 1 token spanning two words, or the submission is
   disqualified. But a cross-word token occupies ONE sequence position, so the
   LSTM emits ONE label for the two words it covers -- if they differ in
   entity-hood, F1 suffers. We therefore learn word-internal merges first
   (good subwords, F1-friendly), then add a *small, tunable* number of cross-word
   merges (`num_bigram_merges`) chosen as the most frequent adjacent word pairs
   -- which are overwhelmingly function-word pairs ("of the", "in the") that are
   almost never entities, so they are the safest possible bigrams. Raising this
   knob trades a little F1 for better efficiency in the competition.

4. DETERMINISM. Merge selection breaks ties by the lexicographically smallest
   pair, so training is reproducible regardless of dict/set iteration order
   (no hash-seed dependence).

Picklability: the whole object is pickled by BaseTokenizer.save. We keep only
plain dicts/lists/tuples on the instance and drop the (rebuildable) encode cache
in __getstate__, so the grader can load us knowing only BaseTokenizer.
"""

import heapq
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from base_tokenizer import BaseTokenizer


# Default number of cross-word merges (bigrams). 1 is the safe minimum that
# satisfies the spec with minimal F1 risk; tune upward on dev for efficiency.
DEFAULT_NUM_BIGRAM_MERGES = 1

# Stop word-internal merging once the best pair occurs fewer than this many
# times -- merging hapax pairs just memorises noise and hurts generalisation.
MIN_PAIR_FREQ = 2


def bytes_to_unicode() -> Dict[int, str]:
    """Reversible map from each byte (0-255) to a printable Unicode char.

    Standard GPT-2 mapping. Bytes that are already printable map to themselves;
    the rest (controls, space, etc.) map to code points starting at 256. The
    space byte (0x20) deterministically becomes 'Ġ' (U+0120), which we use as the
    space marker.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 10000,
                 num_bigram_merges: int = DEFAULT_NUM_BIGRAM_MERGES):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_bigram_merges = num_bigram_merges

        # Byte <-> unicode-surface maps (plain dicts -> picklable).
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {ch: b for b, ch in self.byte_encoder.items()}

        # The space marker the NER pipeline and submission checker read.
        self.space_token = self.byte_encoder[ord(" ")]  # 'Ġ'

        # Learned merges, in order. word_* apply within a word; cross_* apply
        # across word boundaries (these create the bigram tokens).
        self.word_merges: List[Tuple[str, str]] = []
        self.cross_merges: List[Tuple[str, str]] = []
        self.word_ranks: Dict[Tuple[str, str], int] = {}
        self.cross_ranks: Dict[Tuple[str, str], int] = {}

        # Seed the vocabulary with all 256 byte symbols (guarantees zero OOV).
        for b in range(256):
            self._add_token(self.byte_encoder[b])

        self._cache: Dict[tuple, List[str]] = {}

    # ------------------------------------------------------------------ vocab
    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    # -------------------------------------------------------------- pre-token
    def _text_to_units(self, text: str) -> List[List[str]]:
        """Map text -> byte symbols, split into word-units at each space marker.

        A new unit begins at every space marker, so the marker stays attached to
        the FOLLOWING word (GPT-2 convention) and the first word carries no
        leading marker. This keeps surfaces a faithful partition of the text.
        """
        symbols = [self.byte_encoder[b] for b in text.encode("utf-8")]
        units: List[List[str]] = []
        cur: List[str] = []
        for s in symbols:
            if s == self.space_token:
                if cur:
                    units.append(cur)
                cur = [s]
            else:
                cur.append(s)
        if cur:
            units.append(cur)
        return units

    # ------------------------------------------------------------ merge logic
    @staticmethod
    def _apply_merges(tokens: List[str],
                      ranks: Dict[Tuple[str, str], int]) -> List[str]:
        """Greedily merge the lowest-rank adjacent pair until none remain."""
        if len(tokens) < 2:
            return tokens
        tokens = list(tokens)
        while True:
            best_rank = None
            best_pair = None
            for i in range(len(tokens) - 1):
                r = ranks.get((tokens[i], tokens[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = (tokens[i], tokens[i + 1])
            if best_pair is None:
                break
            a, b = best_pair
            merged = a + b
            out, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    def _encode_unit(self, unit: Tuple[str, ...]) -> List[str]:
        cached = self._cache.get(unit)
        if cached is not None:
            return cached
        out = self._apply_merges(list(unit), self.word_ranks)
        self._cache[unit] = out
        return out

    # ----------------------------------------------------------------- train
    def train(self, texts: List[str]) -> None:
        # 1) Collect unique word-units and their frequencies (the classic BPE
        #    speed-up: train on unique words weighted by count, not the full
        #    corpus). Strip only the trailing newline; keep everything else.
        unit_freqs: Counter = Counter()
        for line in texts:
            line = line.rstrip("\n").rstrip("\r")
            for unit in self._text_to_units(line):
                unit_freqs[tuple(unit)] += 1
        if not unit_freqs:
            return

        splits = {u: list(u) for u in unit_freqs}

        # Budget: total vocab minus specials, the 256 base bytes, and the
        # reserved cross-word merge slots.
        base = len(self.token_to_id)  # specials + 256 bytes
        budget_word = self.vocab_size - base - self.num_bigram_merges
        budget_word = max(budget_word, 0)

        # 2) Word-internal BPE with INCREMENTAL pair counts.
        #    Naive BPE rescans the whole corpus every merge -> O(merges*corpus).
        #    Instead we keep a running count of every adjacent pair and an index
        #    of which units contain each pair, then on each merge touch ONLY the
        #    units that actually held the merged pair, updating counts by delta.
        #    The chosen merges (and thus the output) are identical to the naive
        #    version; only the cost changes.
        def _adj(sym: List[str]) -> Counter:
            c: Counter = Counter()
            for x, y in zip(sym, sym[1:]):
                c[(x, y)] += 1
            return c

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_units: Dict[Tuple[str, str], set] = defaultdict(set)
        for u, freq in unit_freqs.items():
            local = _adj(splits[u])
            for p, n in local.items():
                pair_counts[p] += n * freq
                pair_units[p].add(u)

        # Pick the best pair with a LAZY MAX-HEAP instead of rescanning every
        # pair each merge. An entry (-count, pair) orders by highest count, then
        # lexicographically smallest pair -- exactly the old deterministic
        # tie-break. We never delete from the heap; a popped entry is "stale" if
        # its stored count no longer matches pair_counts, and is simply skipped.
        # Whenever a count changes we push the new value. Selection drops from
        # O(#pairs) to O(log #pairs) amortised; chosen merges are unchanged.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        for _ in range(budget_word):
            best = None
            while heap:
                neg, p = heapq.heappop(heap)
                if pair_counts.get(p, 0) == -neg:  # still current -> the true best
                    best, max_count = p, -neg
                    break
            if best is None or max_count < MIN_PAIR_FREQ:
                break
            a, b = best
            merged = a + b
            self.word_merges.append(best)
            self._add_token(merged)

            # Only the units containing `best` can change.
            for u in list(pair_units.get(best, ())):
                freq = unit_freqs[u]
                old = splits[u]
                before = _adj(old)
                new, i = [], 0
                while i < len(old):
                    if i < len(old) - 1 and old[i] == a and old[i + 1] == b:
                        new.append(merged)
                        i += 2
                    else:
                        new.append(old[i])
                        i += 1
                after = _adj(new)
                splits[u] = new

                for p in set(before) | set(after):
                    delta = (after[p] - before[p]) * freq
                    if delta:
                        pair_counts[p] += delta
                        if pair_counts[p] <= 0:
                            pair_counts.pop(p, None)
                        elif p != best:  # publish the new count for selection
                            heapq.heappush(heap, (-pair_counts[p], p))
                    had, has = before[p] > 0, after[p] > 0
                    if has and not had:
                        pair_units[p].add(u)
                    elif had and not has:
                        s = pair_units.get(p)
                        if s is not None:
                            s.discard(u)
                            if not s:
                                pair_units.pop(p, None)

            # `best` is fully consumed; drop any residual bookkeeping.
            pair_counts.pop(best, None)
            pair_units.pop(best, None)

        self.word_ranks = {p: i for i, p in enumerate(self.word_merges)}
        self._cache.clear()

        # 3) Cross-word merges -> the required bigram tokens. Re-encode the
        #    corpus with the learned word merges and count adjacent token pairs
        #    that form a GENUINE two-word bigram: both tokens must be whole words
        #    -- i.e. carry a leading space marker AND have content beyond it.
        #    Requiring this (rather than just "b starts with the marker") matters:
        #    the tweet domains are full of double spaces, which produce lone
        #    space-marker tokens ('Ġ'). Merging `(word, 'Ġ')` yields a token whose
        #    surface strips to a single word with no internal space, so it would
        #    FAIL the submission checker's bigram test and disqualify the
        #    tokenizer. Both-words-only guarantees the merged token contains an
        #    internal space (after marker->space) and is a real word-word bigram.
        sp = self.space_token
        cross_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for line in texts:
            line = line.rstrip("\n").rstrip("\r")
            units = self._text_to_units(line)
            flat: List[str] = []
            for u in units:
                flat.extend(self._encode_unit(tuple(u)))
            for a, b in zip(flat, flat[1:]):
                if (a.startswith(sp) and len(a) > len(sp)
                        and b.startswith(sp) and len(b) > len(sp)):
                    cross_counts[(a, b)] += 1

        if cross_counts:
            ordered = sorted(cross_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            for (a, b), _ in ordered[: self.num_bigram_merges]:
                self.cross_merges.append((a, b))
                self._add_token(a + b)
            self.cross_ranks = {p: i for i, p in enumerate(self.cross_merges)}

    # ---------------------------------------------------------------- encode
    def encode(self, text: str) -> List[int]:
        flat: List[str] = []
        for u in self._text_to_units(text):
            flat.extend(self._encode_unit(tuple(u)))
        if self.cross_ranks:
            flat = self._apply_merges(flat, self.cross_ranks)
        unk = self.special_tokens["[UNK]"]
        return [self.token_to_id.get(t, unk) for t in flat]

    # ---------------------------------------------------------------- decode
    def decode(self, token_ids: List[int]) -> str:
        special_ids = set(self.special_tokens.values())
        pieces = []
        for i in token_ids:
            if i in special_ids:
                continue  # specials render as empty -> zero-width, offset-safe
            tok = self.id_to_token.get(i)
            if tok is not None:
                pieces.append(tok)
        surface = "".join(pieces)
        try:
            data = bytes(self.byte_decoder[c] for c in surface)
        except KeyError:
            # Any stray char with no byte mapping: drop it rather than crash.
            data = bytes(self.byte_decoder[c] for c in surface if c in self.byte_decoder)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------- pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}  # don't persist the (rebuildable) cache
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "_cache") or self._cache is None:
            self._cache = {}