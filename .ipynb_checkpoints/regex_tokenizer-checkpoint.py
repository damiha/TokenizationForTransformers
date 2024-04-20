import regex as re
from tqdm import tqdm

class RegexTokenizer:

    def __init__(self, regex_pattern=None, special_tokens=None):

        if special_tokens is None:
            special_tokens = [
                "<|endoftext|>"
            ]

        if regex_pattern is None:
            # we use the splitting pattern from GPT4
            regex_pattern = \
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

        self.regex_pattern = re.compile(regex_pattern)
        self.special_tokens = special_tokens

        self.special_pattern = re.compile("(" + " | ".join([re.escape(special) for special in self.special_tokens]) + ")")

        self.vocab = None
        self.merges = None
        self.special_to_token_ids = None
        self.token_ids_to_special = None

    # update stats in place
    def get_stats(self, token_ids, stats):
    
        for pair in zip(token_ids[:-1], token_ids[1:]):
    
            stats[pair] = stats.get(pair, 0) + 1
    
        return stats

    def get_most_frequent_pair(self, stats):
        # first [0] extracts most frequent item, second [0] only selects pair (not count)
        return sorted(stats.items(), key=lambda t: t[1], reverse=True)[0][0]

    def replace(self, token_ids, pair, pair_token_id):

        new_token_ids = []
    
        i = 0
        while i < len(token_ids):
    
            if (i + 1) < len(token_ids) and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
    
                new_token_ids.append(pair_token_id)
                i += 2
    
            else:
                new_token_ids.append(token_ids[i])
                i += 1
    
        return new_token_ids

    def print_merges(self):
        for k, v in self.merges.items():
            print(f"{k} -> {v}")

    def merges_to_vocab(self):

        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # iterate in the order of insertion (super important, implicit in .items())
        for token_pair, new_token_id in self.merges.items():
            self.vocab[new_token_id] = self.vocab[token_pair[0]] + self.vocab[token_pair[1]]

    def train(self, text, n_merges):
    
        # 1. split at special characters (texts can be the special characters themselves)
        texts = re.split(self.special_pattern, text)
        # dont train on the special characters
        ordinary_texts = list(filter(lambda t: t not in self.special_tokens, texts))

        chunks = []

        for ordinary_text in ordinary_texts:
            chunks.extend(re.findall(self.regex_pattern, ordinary_text))

        chunk_token_ids = [list(chunk.encode("utf-8")) for chunk in chunks]
    
        highest_token_id = 255
    
        self.merges = dict()
    
        for _ in tqdm(range(n_merges)):

            stats = dict()
            for token_ids in chunk_token_ids:
                # in place update
                stats = self.get_stats(token_ids, stats)
        
            most_frequent_pair = self.get_most_frequent_pair(stats)
    
            highest_token_id += 1

            for i, token_ids in enumerate(chunk_token_ids):
                chunk_token_ids[i] = self.replace(token_ids, most_frequent_pair, highest_token_id)
    
            self.merges[most_frequent_pair] = highest_token_id

        self.merges_to_vocab()

        # special tokens get their token ids
        self.special_to_token_ids = dict()
        
        for i, s in enumerate(self.special_tokens):
            self.special_to_token_ids[s] = highest_token_id + (i + 1)

        self.token_ids_to_special = {v:k for (k, v) in self.special_to_token_ids.items()}

    # by default, text must not have special characters
    def encode(self, text, allow_special="none_raise"):

        texts = re.split(self.special_pattern, text)

        if allow_special == "all":
            special = self.special_tokens
        elif allow_special == "none_raise":
            special = {}
            assert(all(s not in text for s in self.special_tokens))

        ids = []
        
        for t in tqdm(texts):
            if t in self.special_tokens:
                ids.append(self.special_to_token_ids[t])

            else:
                # this is inefficient
                token_ids = list(t.encode("utf-8"))
                for pair, new_token_id in self.merges.items():
                    token_ids = self.replace(token_ids, pair, new_token_id)

                ids.extend(token_ids)

        return ids

    def decode(self, token_ids):

        byte_list = []
    
        for t in token_ids:

            if t in self.vocab:
                byte_list.append(self.vocab[t])
            elif t in self.token_ids_to_special:
                byte_list.append(self.token_ids_to_special[t].encode("utf-8"))
            else:
                raise ValueError("token id unknown")
    
        return b"".join(byte_list).decode("utf-8", errors="replace")
