#!/usr/bin/env python3
"""Convert float32 fbin dataset to fp16 (stored as float32 fbin with fp16 precision)."""
import struct
import numpy as np
import sys
import os

def convert_fbin_to_fp16(input_path, output_path):
    with open(input_path, 'rb') as f:
        nrows, dim = struct.unpack('ii', f.read(8))
        print(f"Dataset: {nrows} rows x {dim} dims")
        print(f"Reading float32 data...")
        data = np.fromfile(f, dtype=np.float32, count=nrows * dim).reshape(nrows, dim)

    print(f"Converting to fp16 and back to float32...")
    data_fp16 = data.astype(np.float16).astype(np.float32)

    # Show precision loss stats
    diff = np.abs(data - data_fp16)
    print(f"  Max absolute error: {diff.max():.6e}")
    print(f"  Mean absolute error: {diff.mean():.6e}")
    print(f"  Relative error (mean): {(diff / (np.abs(data) + 1e-10)).mean():.6e}")

    print(f"Writing fp16-precision data to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(struct.pack('ii', nrows, dim))
        data_fp16.tofile(f)

    orig_size = os.path.getsize(input_path)
    new_size = os.path.getsize(output_path)
    print(f"Done. File size: {new_size} bytes (same as original: {orig_size})")

if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/data/cohere/cohere.fbin'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '/home/ubuntu/data/cohere/cohere_fp16.fbin'
    convert_fbin_to_fp16(input_path, output_path)
