class BasicTokenizer:

    def __init__(self):

        self.vocab = None
        self.merges = None

    def get_stats(self, token_ids):

        stats = dict()
    
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
    
        token_ids = list(text.encode("utf-8"))
    
        highest_token_id = 255
    
        self.merges = dict()
    
        for _ in range(n_merges):
    
            stats = self.get_stats(token_ids)
        
            most_frequent_pair = self.get_most_frequent_pair(stats)
    
            highest_token_id += 1
            token_ids = self.replace(token_ids, most_frequent_pair, highest_token_id)
    
            self.merges[most_frequent_pair] = highest_token_id

        self.merges_to_vocab()

    def encode(self, text):

        token_ids = list(text.encode("utf-8"))

        # replace earliest first
        for token_pair, new_token_id in self.merges.items():
            token_ids = self.replace(token_ids, token_pair, new_token_id)

        return token_ids

    def decode(self, token_ids):

        byte_list = []
    
        for t in token_ids:
            byte_list.append(self.vocab[t])
    
        return b"".join(byte_list).decode("utf-8", errors="replace")
