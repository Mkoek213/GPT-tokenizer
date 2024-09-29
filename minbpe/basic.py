from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=True):
        num_merges = vocab_size - 256
        ids = text.encode("utf-8", errors="replace")
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key = stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")


    def encode(self, text: str):
        tokens = text.encode("utf-8", errors="replace")
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            priorities = {pair: self.merges.get(pair, float("inf")) for pair in stats}
            pair = min(priorities, key=priorities.get)
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        text_b = b""
        for idx in ids:
            text_b += self.vocab[idx]
        return text_b.decode("utf-8", errors="replace")

