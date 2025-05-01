import random
import bisect
import joblib
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


dt_freq_length = {
    1: 512,
    5: 512,
    15: 512,
    60: 512,
    1440: 90
}

dt_sample_prob = {
    'CN-F': {
        1: 0.1 * 0.8,
        5: 0.1 * 0.16,
        15: 0.1 * 0.037,
        1440: 0.1 * 0.003
    },
    'CN-ETF': {
        1: 0.1 * 0.8,
        5: 0.1 * 0.195,
        1440: 0.1 * 0.005
    },
    'Crypto-P': {
        1: 0.20 * 0.65,
        5: 0.20 * 0.20,
        15: 0.20 * 0.10,
        60: 0.20 * 0.045,
        1440: 0.20 * 0.005
    },
    'Crypto-S': {
        1: 0.05 * 0.65,
        5: 0.05 * 0.20,
        15: 0.05 * 0.10,
        60: 0.05 * 0.045,
        1440: 0.05 * 0.005
    },
    'CN-A': {
        1: 0.25 * 0.80,
        5: 0.25 * 0.18,
        1440: 0.25 * 0.02,
    },
    'US-Eq': {
        1: 0.25 * 0.80,
        5: 0.25 * 0.18,
        1440: 0.25 * 0.02
    }
}


class TokenSeqDataset(Dataset):
    def __init__(self, root_data_path, dt_freq_length, dt_sample_prob, tokenizer, batch_size=64, market_list=None):
        self.root_data_path = root_data_path
        self.dt_freq_length = dt_freq_length
        self.market_list = market_list
        self.dt_sample_prob = dt_sample_prob
        self.dt_market_prob = {market: sum(dt_sample_prob[market].values()) for market in dt_sample_prob.keys()}
        self.batch_size = batch_size
        self.tokenizer = tokenizer.to("cuda:0")
        self._cache = {}
        self.dt_data, self.dt_sample_span, self.dt_sample_count = self.load_data()

    def load_data(self):
        print(f"Loading data from {self.root_data_path}...")
        dt_data = {}
        dt_sample_span = {}
        dt_sample_count = {}
        if self.market_list is None:
            self.market_list = list(set([file.split('_')[0] for file in os.listdir(self.root_data_path)]))
        for market in list(self.dt_market_prob.keys()):
            if market not in self.market_list:
                del self.dt_market_prob[market], self.dt_sample_prob[market]
        print(f"Market list: {self.market_list}")
        for market in self.market_list:
            print(f"Loading {market} data...")
            dt_market_data = joblib.load(f"{self.root_data_path}/{market}_data.joblib")
            dt_market_sample_span = joblib.load(f"{self.root_data_path}/{market}_sample_span.joblib")
            dt_market_sample_count = joblib.load(f"{self.root_data_path}/{market}_sample_count.joblib")

            dt_data[market] = dt_market_data
            dt_sample_span[market] = dt_market_sample_span
            dt_sample_count[market] = dt_market_sample_count
        return dt_data, dt_sample_span, dt_sample_count

    def __len__(self):
        total_count = 0
        for market in self.dt_sample_count.keys():
            for freq in self.dt_sample_count[market].keys():
                total_count += self.dt_sample_count[market][freq]
        return total_count

    def sample_market(self):
        market = random.choices(list(self.dt_market_prob.keys()), weights=list(self.dt_market_prob.values()), k=1)[0]
        return market

    def sample_freq(self, market):
        freq = random.choices(list(self.dt_sample_prob[market].keys()), weights=list(self.dt_sample_prob[market].values()), k=1)[0]
        return freq

    def get_symbol(self, idx, sample_span):
        span_id = id(sample_span)

        if span_id not in self._cache:
            intervals = []
            for name, (start, end) in sample_span.items():
                intervals.append((start, end, name))
            intervals.sort()

            starts = [x[0] for x in intervals]
            ends = [x[1] for x in intervals]
            names = [x[2] for x in intervals]

            self._cache[span_id] = (starts, ends, names)
        else:
            starts, ends, names = self._cache[span_id]

        pos = bisect.bisect_right(starts, idx) - 1

        if pos >= 0 and idx < ends[pos]:
            return names[pos]

        raise ValueError("Index out of range")

    def __getitem__(self, idx):
        market = self.sample_market()
        freq = self.sample_freq(market)

        sample_data = self.dt_data[market][freq]
        sample_span = self.dt_sample_span[market][freq]
        sample_count = self.dt_sample_count[market][freq]
        lookback = self.dt_freq_length[freq]

        while sample_count == 0:
            market = self.sample_market()
            freq = self.sample_freq(market)
            sample_data = self.dt_data[market][freq]
            sample_span = self.dt_sample_span[market][freq]
            sample_count = self.dt_sample_count[market][freq]
            lookback = self.dt_freq_length[freq]

        ls_batch_x = []
        ls_batch_y = []
        ls_batch_x_stamp = []

        for _ in range(self.batch_size):
            idx = random.randint(0, sample_count - 1)
            symbol = self.get_symbol(idx, sample_span)
            symbol_data = sample_data[symbol]

            idx_in_symbol = idx - sample_span[symbol][0]
            idx_in_series = idx_in_symbol + lookback

            x = symbol_data[idx_in_series - lookback: idx_in_series, :6]
            x_stamp = symbol_data[idx_in_series - lookback: idx_in_series, 6:]

            y = symbol_data[idx_in_series - lookback + 1: idx_in_series + 1, :6]

            x_mean = np.mean(x, axis=0)
            x_std = np.std(x, axis=0)
            x = (x - x_mean) / (x_std + 1e-6)
            y = (y - x_mean) / (x_std + 1e-6)

            x = np.clip(x, -30, 30)

            x = x.astype(np.float32)
            x_stamp = x_stamp.astype(np.float32)
            y = y.astype(np.float32)

            ls_batch_x.append(x)
            ls_batch_y.append(y)
            ls_batch_x_stamp.append(x_stamp)

        batch_x = torch.tensor(np.stack(ls_batch_x, axis=0), device="cpu")
        batch_y = torch.tensor(np.stack(ls_batch_y, axis=0), device="cpu")
        batch_x_stamp = torch.tensor(np.stack(ls_batch_x_stamp, axis=0), device="cpu")

        with torch.no_grad():
            token_in = self.tokenizer.encode(batch_x, half=True)
            token_out = self.tokenizer.encode(batch_y, half=True)

        return token_in, token_out, batch_x_stamp
        # return batch_x, batch_y, batch_x_stamp


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from Model import KronosTokenizer

    # root_model_path = '/data/shiyu/Kronos/result/ours'
    root_model_path = '/ssdshare/share/cs/Kronos/result/ours'
    tokenizer_tag = 'S1_9_S2_9_NH128_NL3_NRH256_TAGhf_tot_c'
    tokenizer = KronosTokenizer.from_pretrained(f"{root_model_path}/{tokenizer_tag}/checkpoints/best_model")

    # root_data_path = "/data/shiyu/data/processed_data/train"
    root_data_path = "/ssdshare/share/cs/data/processed_data/train"

    market_list = ['CN-F']
    dataset = TokenSeqDataset(root_data_path, dt_freq_length, dt_sample_prob, tokenizer, batch_size=64, market_list=market_list)
    print(len(dataset))

    x, y, x_stamp = dataset[1000]
    print(x)
    print(y)
    print(x_stamp)
    print(x[0].shape, y[0].shape, x_stamp.shape)












