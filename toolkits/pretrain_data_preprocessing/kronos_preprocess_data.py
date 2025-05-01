# Copyright (c) 2023 Alibaba PAI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Processing financial time series data for Kronos pretraining using Megatron-Core (Multi-GPU Version)."""

import argparse
import multiprocessing
import os
import sys
import time
from typing import Tuple
import joblib
import numpy as np
import torch
import tqdm  # For progress bars
import math
import traceback
import gc

# Add Megatron-Patch and Kronos paths to sys.path
# This assumes the script is run from toolkits/pretrain_data_preprocessing
_TOOLKITS_DIR = os.path.dirname(os.path.abspath(__file__))
_MEGATRON_PATCH_DIR = os.path.abspath(os.path.join(_TOOLKITS_DIR, os.pardir, os.pardir))
_KRONOS_DIR = os.path.join(_MEGATRON_PATCH_DIR, "kronos")
sys.path.insert(0, _MEGATRON_PATCH_DIR)
sys.path.insert(1, _KRONOS_DIR)

# Megatron and Kronos imports after path setup
from megatron.core.datasets import indexed_dataset
from megatron_patch.tokenizer import build_tokenizer
# Assuming KronosTokenizer is importable after path setup
# from kronos.Model.kronos import KronosTokenizer # Not strictly needed here, build_tokenizer handles it

# Constants for timestamp packing
TS_MINUTE_BITS = 6
TS_HOUR_BITS = 5
TS_WDAY_BITS = 3
TS_DAY_BITS = 5
TS_MONTH_BITS = 4
TOKEN_BITS = 18

TS_MONTH_SHIFT = 0
TS_DAY_SHIFT = TS_MONTH_SHIFT + TS_MONTH_BITS
TS_WDAY_SHIFT = TS_DAY_SHIFT + TS_DAY_BITS
TS_HOUR_SHIFT = TS_WDAY_SHIFT + TS_WDAY_BITS
TS_MINUTE_SHIFT = TS_HOUR_SHIFT + TS_HOUR_BITS
TS_TOTAL_BITS = TS_MINUTE_SHIFT + TS_MINUTE_BITS  # Should be 23

TOKEN_SHIFT = 0
TS_PACKED_SHIFT = TOKEN_SHIFT + TOKEN_BITS  # Timestamp section starts after token bits

# Global variable to hold the tokenizer for each worker process
# This avoids passing it explicitly but relies on initializer pattern
worker_tokenizer = None
worker_device = None


def worker_initializer(args, counter, lock):
    """Initializer for worker processes: sets device and loads tokenizer."""
    with lock:
        device_id = counter.value
        print(f"Device ID: {device_id}")
        counter.value += 1
    global worker_tokenizer, worker_device
    print(f"Initializing worker {os.getpid()} for device {device_id}...")
    # Assign a default device ID if not provided correctly, though it should be.
    assigned_device_id = device_id if isinstance(device_id, int) else 0
    worker_device = f"cuda:{assigned_device_id}"
    try:
        torch.cuda.set_device(worker_device)
    except Exception as e:
        print(
            f"Error setting device {worker_device} for worker {os.getpid()}: {e}. Defaulting to cuda:0"
        )
        worker_device = (
            "cuda:0"  # Fallback, might cause issues if multiple workers default
        )
        try:
            torch.cuda.set_device(worker_device)
        except Exception as e2:
            print(f"FATAL: Could not set even cuda:0 for worker {os.getpid()}: {e2}")
            return  # Cannot proceed without a device

    # Set up dummy args required by build_tokenizer
    args.rank = 0  # Build_tokenizer might check rank for logging
    args.make_vocab_size_divisible_by = 128  # Or appropriate value
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0  # Check if KronosTokenizer needs this
    args.patch_tokenizer_type = "KronosTokenizer"  # Ensure correct type
    args.load = args.tokenizer_path  # Simplify arg passing

    try:
        local_tokenizer = build_tokenizer(args)
        # Ensure the tokenizer model is on the assigned GPU
        if hasattr(local_tokenizer, "tokenizer") and hasattr(
            local_tokenizer.tokenizer, "to"
        ):
            local_tokenizer.tokenizer.to(worker_device)
            local_tokenizer.tokenizer.eval()  # Set to eval mode
        else:
            print(
                f"Warning: Could not move tokenizer model to {worker_device} for worker {os.getpid()}"
            )

        # Check vocab size
        print(
            f"Worker {os.getpid()} tokenizer initialized on {worker_device}. Vocab size: {local_tokenizer.vocab_size}"
        )
        if local_tokenizer.vocab_size != (2**TOKEN_BITS):
            print(
                f"Warning (Worker {os.getpid()}): Tokenizer vocab size {local_tokenizer.vocab_size} does not match expected {2**TOKEN_BITS}"
            )

        worker_tokenizer = local_tokenizer  # Store in global for this process
    except Exception as e:
        print(
            f"Error initializing tokenizer for worker {os.getpid()} on device {worker_device}: {e}"
        )
        traceback.print_exc()
        worker_tokenizer = None  # Ensure it's None if init failed

min_minute = float('inf')
max_minute = float('-inf')
min_hour = float('inf')
max_hour = float('-inf')
min_weekday = float('inf')
max_weekday = float('-inf')
min_day = float('inf')
max_day = float('-inf')
min_month = float('inf')
max_month = float('-inf')

def encode_batch(feature_batch_list, timestamp_batch_list):
    """
    Tokenizes a batch of feature chunks on GPU and packs results with timestamps on CPU.

    Args:
        feature_batch_list (list): List of np.ndarray[np.float32] feature chunks.
        timestamp_batch_list (list): List of np.ndarray[np.int64] timestamp chunks.

    Returns:
        list: List of np.ndarray[np.int64] combined token+timestamp chunks.
              Returns empty list on error.
    """
    global worker_tokenizer, worker_device
    if not feature_batch_list or worker_tokenizer is None or worker_device is None:
        return []

    batch_size = len(feature_batch_list)
    seq_length = feature_batch_list[0].shape[0]  # Assume all chunks have same length

    try:
        # Stack numpy arrays into a single tensor
        # Ensure consistent sequence length (should be guaranteed by chunking)
        feature_batch_np = np.stack(feature_batch_list, axis=0)
        feature_batch_tensor = torch.from_numpy(feature_batch_np).to(worker_device)

        # Tokenize on GPU
        with torch.no_grad():
            # Pass the tensor directly, the wrapper handles device if needed
            token_ids_gpu = worker_tokenizer(
                feature_batch_tensor
            )  # Expect shape [batch_size, seq_length]

        # Handle potential list output (shouldn't happen with half=False)
        if isinstance(token_ids_gpu, list):
            print(
                f"Warning (Worker {os.getpid()}): Tokenizer returned a list. Expected tensor. Using first element."
            )
            token_ids_gpu = token_ids_gpu[0]

        # Move results back to CPU for packing
        token_ids_batch_cpu = token_ids_gpu.cpu().numpy().astype(np.int64)

        # Check token range for the entire batch
        if (
            token_ids_batch_cpu.max() >= (2**TOKEN_BITS)
            or token_ids_batch_cpu.min() < 0
        ):
            print(
                f"Error (Worker {os.getpid()}): Batch Token ID out of range [0, {2**TOKEN_BITS - 1}]. Max: {token_ids_batch_cpu.max()}, Min: {token_ids_batch_cpu.min()}. Skipping batch."
            )
            return []

        # Pack timestamps and tokens on CPU
        processed_batch = []
        for i in range(batch_size):
            token_ids = token_ids_batch_cpu[i]  # Shape [seq_length]
            timestamps = timestamp_batch_list[i]  # Shape [seq_length, 5]
            # print(f"range of minute: {timestamps[:, 0].min()} - {timestamps[:, 0].max()}")
            # print(f"range of hour: {timestamps[:, 1].min()} - {timestamps[:, 1].max()}")
            # print(f"range of weekday: {timestamps[:, 2].min()} - {timestamps[:, 2].max()}")
            # print(f"range of day: {timestamps[:, 3].min()} - {timestamps[:, 3].max()}")
            # print(f"range of month: {timestamps[:, 4].min()} - {timestamps[:, 4].max()}")
            global min_minute, max_minute, min_hour, max_hour, min_weekday, max_weekday, min_day, max_day, min_month, max_month
            min_minute = min(min_minute, timestamps[:, 0].min())
            max_minute = max(max_minute, timestamps[:, 0].max())
            min_hour = min(min_hour, timestamps[:, 1].min())
            max_hour = max(max_hour, timestamps[:, 1].max())
            min_weekday = min(min_weekday, timestamps[:, 2].min())
            max_weekday = max(max_weekday, timestamps[:, 2].max())
            min_day = min(min_day, timestamps[:, 3].min())
            max_day = max(max_day, timestamps[:, 3].max())
            min_month = min(min_month, timestamps[:, 4].min())
            max_month = max(max_month, timestamps[:, 4].max())
            combined_chunk = np.zeros(seq_length, dtype=np.int64)

            for t in range(seq_length):
                token_id = token_ids[t]
                ts = timestamps[t]  # minute, hour, weekday, day, month

                packed_ts = (
                    (ts[0] << TS_MINUTE_SHIFT)
                    | (ts[1] << TS_HOUR_SHIFT)
                    | (ts[2] << TS_WDAY_SHIFT)
                    | (ts[3] << TS_DAY_SHIFT)
                    | (ts[4] << TS_MONTH_SHIFT)
                )
                combined_value = (packed_ts << TS_PACKED_SHIFT) | (
                    token_id << TOKEN_SHIFT
                )
                combined_chunk[t] = combined_value
            processed_batch.append(combined_chunk)

        return processed_batch

    except Exception as e:
        print(
            f"Error encoding batch in worker {os.getpid()} on device {worker_device}: {e}"
        )
        traceback.print_exc()
        return []


def process_tasks(task: Tuple[str, str], args: argparse.Namespace) -> int:
    global worker_tokenizer, worker_device

    market, freq = task

    seq_length = args.sequence_length
    gpu_batch_size = args.gpu_batch_size

    data_path = os.path.join(args.input, f"{market}_data.joblib")
    if not os.path.exists(data_path):
        print(f"Warning (Worker {os.getpid()}): Data file not found for market {market}. Skipping task.")
        raise FileNotFoundError(f"Data file not found for market {market} at {data_path}")

    builder_key = task

    feature_batch_list = []
    timestamp_batch_list = []
    processed_symbols_count = 0
    total_items_written_worker = 0
    builder = indexed_dataset.IndexedDatasetBuilder(
        os.path.join(args.output_prefix, f"{builder_key[0]}_{builder_key[1]}.bin"), dtype=np.int64
    )

    try:
        market_data = joblib.load(data_path)
        freq_data = market_data.get(freq, {})

        symbol_keys = list(freq_data.keys())  # Get keys to iterate over

        for symbol in symbol_keys:  # Iterate over keys, access data via dict
            symbol_data = freq_data.get(symbol)  # Use .get for safety

            if (
                symbol_data is None
                or symbol_data.shape[0] < seq_length
                or symbol_data.shape[1] < 11
            ):
                continue

            try:
                features = symbol_data[:, :6].astype(np.float32)
                timestamps = symbol_data[:, 6:].astype(np.int64)

                features_mean = np.mean(features, axis=0)
                features_std = np.std(features, axis=0)
                normalized_features = (features - features_mean) / (
                    features_std + 1e-6
                )
                normalized_features = np.clip(normalized_features, -30, 30)

                num_steps = normalized_features.shape[0]
                symbol_chunks_count = 0

                # Chunk data (CPU)
                for i in range(
                    0, num_steps - seq_length + 1, seq_length
                ):  # Ensure full chunks only
                    feature_chunk = normalized_features[i : i + seq_length]
                    timestamp_chunk = timestamps[i : i + seq_length]

                    # Accumulate batch (CPU)
                    feature_batch_list.append(feature_chunk)
                    timestamp_batch_list.append(timestamp_chunk)
                    symbol_chunks_count += 1

                    # Process batch when full
                    if len(feature_batch_list) == gpu_batch_size:
                        processed_batch = encode_batch(
                            feature_batch_list, timestamp_batch_list
                        )
                        # Write results (CPU)
                        if processed_batch:
                            for packed_chunk in processed_batch:
                                builder.add_item(
                                    torch.from_numpy(packed_chunk)
                                )
                                total_items_written_worker += 1
                        # Clear batch lists
                        feature_batch_list.clear()
                        timestamp_batch_list.clear()

            except Exception as symbol_err:
                print(
                    f"Error processing symbol {market}-{freq}-{symbol} in worker {os.getpid()}: {symbol_err}"
                )
                traceback.print_exc()
                # Clear potentially corrupted batch state if error occurred mid-symbol
                feature_batch_list.clear()
                timestamp_batch_list.clear()
            finally:
                processed_symbols_count += 1

        # Clean up market data after processing all its symbols
        del market_data, freq_data
        gc.collect()
    except Exception as e:
        print(f"Error processing task {market}-{freq}: {e}")
        traceback.print_exc()
    
    if feature_batch_list:
        processed_batch = encode_batch(feature_batch_list, timestamp_batch_list)
        if processed_batch:
            for packed_chunk in processed_batch:
                builder.add_item(
                    torch.from_numpy(packed_chunk)
                )
                total_items_written_worker += 1
    builder.end_document()
    builder.finalize(os.path.join(args.output_prefix, f"{builder_key[0]}_{builder_key[1]}.idx"))

    print(f"range of minute: {min_minute} - {max_minute}")
    print(f"range of hour: {min_hour} - {max_hour}")
    print(f"range of weekday: {min_weekday} - {max_weekday}")
    print(f"range of day: {min_day} - {max_day}")
    print(f"range of month: {min_month} - {max_month}")
    print()

    return processed_symbols_count, total_items_written_worker


def yield_market_freq_tasks(args):
    """Generates a list of all (market, freq) tuples to be processed."""
    tasks = []
    input_dir = args.input
    markets_to_process = args.markets
    print(f"Generating tasks for markets: {markets_to_process} in {input_dir}")
    for market in markets_to_process:
        data_path = os.path.join(input_dir, f"{market}_data.joblib")
        if not os.path.exists(data_path):
            print(
                f"Warning: Data file not found for market {market} at {data_path}. Skipping market."
            )
            continue
        try:
            # Peek into the joblib file to get frequencies without loading all data
            # This is a bit of a hack, might be slow for huge index files
            with open(data_path, "rb") as f:
                # Try to load just the keys if possible (depends on joblib internals/version)
                # This might still load significant metadata
                market_data_keys = joblib.load(f)  # Load the top-level dict
            if isinstance(market_data_keys, dict):
                for freq in market_data_keys.keys():
                    tasks.append((market, freq))
            else:
                print(
                    f"Warning: Could not read frequencies for market {market}. File format might be unexpected."
                )
            del market_data_keys  # Free memory
            gc.collect()
        except Exception as e:
            print(f"Error reading frequencies for market {market}: {e}")
    print(f"Generated {len(tasks)} market-frequency tasks.")
    return tasks


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert financial time series data to Megatron IndexedDataset format (Multi-GPU)."
    )

    group = parser.add_argument_group(title="Input Data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the directory containing joblib data files",
    )
    group.add_argument(
        "--markets",
        type=str,
        nargs="+",
        required=True,
        help="List of markets to process",
    )

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to the KronosTokenizer model directory",
    )
    group.add_argument(
        "--sequence-length", type=int, default=512, help="Sequence length for chunking"
    )

    group = parser.add_argument_group(title="Output Data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path prefix for output bin/idx files",
    )
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["mmap"],
        help="Dataset implementation",
    )

    group = parser.add_argument_group(title="Runtime")
    group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (ideally equals the number of GPUs)",
    )
    group.add_argument(
        "--gpu-batch-size",
        type=int,
        default=64,
        help="Number of sequences to batch together for GPU tokenization",
    )
    # group.add_argument('--log-interval', type=int, default=100, help='Log interval (handled internally by tqdm now)')

    args = parser.parse_args()
    args.patch_tokenizer_type = "KronosTokenizer"
    args.load = args.tokenizer_path
    return args


def print_devices(tasks, worker_args, device_id):
    global worker_device, worker_tokenizer
    print(f"Worker: {os.getpid()}")
    print(f"Worker device: {worker_device}")
    print(f"Tokenizer device: {next(worker_tokenizer.tokenizer.parameters()).device}")
    print(f"Target device: {device_id}")
    print()
    time.sleep(1)


def main():
    args = get_args()
    start_time = time.time()

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if args.workers > num_gpus:
        print(
            f"Warning: Requested {args.workers} workers, but only {num_gpus} GPUs are available. Using {num_gpus} workers."
        )
        args.workers = num_gpus
    if args.workers <= 0:
        print(f"Error: Number of workers must be positive. Got {args.workers}")
        sys.exit(1)

    os.makedirs(args.output_prefix, exist_ok=True)
    print(f"Starting data processing with {args.workers} GPU workers...")

    # 1. Generate all tasks
    all_tasks = yield_market_freq_tasks(args)
    if not all_tasks:
        print("No tasks generated. Exiting.")
        sys.exit(0)

    # 2. Distribute tasks among workers
    tasks_per_worker = [[] for _ in range(args.workers)]
    for i, task in enumerate(all_tasks):
        worker_index = i % args.workers
        tasks_per_worker[worker_index].append(task)

    # 4. Create and run the pool
    # Use 'spawn' context for CUDA safety in multiprocessing
    ctx = multiprocessing.get_context("spawn")

    counter = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()
    with ctx.Pool(
        processes=args.workers,
        initializer=worker_initializer,
        initargs=(args, counter, lock),
    ) as pool:
        results = pool.starmap(process_tasks, [(task, args) for task in all_tasks])

    # 5. Aggregate results (optional)
    total_symbols = sum(r[0] for r in results if r)
    total_items = sum(r[1] for r in results if r)

    total_time = time.time() - start_time
    print("-" * 60)
    print("Data processing finished.")
    print(f"Total market-frequency tasks: {len(all_tasks)}")
    print(f"Total symbols processed across all workers: {total_symbols}")
    print(f"Total items (sequences) written across all workers: {total_items}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Output files generated in: {args.output_prefix}")
    print("-" * 60)


if __name__ == "__main__":
    # Set start method for CUDA safety
    multiprocessing.set_start_method("spawn", force=True)
    main()
