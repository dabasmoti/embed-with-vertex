import os
from pathlib import Path
import torch
import polars as pl
from sentence_transformers import SentenceTransformer
import numpy as np
from datasets import Dataset
from functools import partial

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    
print(f'Using device: {device}')


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--output_file_name', type=str, default='embeddings.parquet')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--text_col', type=str, default='text')
    parser.add_argument('--label_col', type=str, default='vec')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()
    if args.data_path.startswith("gs://") and not args.local:
        args.data_path = Path("/gcs/" + args.data_path[5:])
        
    if not os.getenv('AIP_MODEL_DIR') and not args.output_path:
        raise ValueError('Either AIP_MODEL_DIR or output_path must be set')
    else:
        args.output_path = args.output_path or os.getenv('AIP_MODEL_DIR')
    
    return args


def predict(model, batch):
    embeddings = model.encode(batch[TEXT_COL], convert_to_numpy=True)
    return {LABEL_COL: embeddings}

    
if __name__ == '__main__':
    args = get_args()
    
    TEXT_COL = args.text_col
    LABEL_COL = args.label_col
    model = SentenceTransformer(args.model_path, device=device)

    df = pl.scan_parquet(args.data_path)
    df = df.filter(pl.col(TEXT_COL).is_not_null()).collect()
    ds = Dataset.from_pandas(df.to_pandas(use_pyarrow_extension_array=True))
    predict_func = partial(predict, model)
    ds = ds.map(predict_func, batched=True, batch_size=args.batch_size)
    file_name = os.path.join(args.output_path, args.output_file_name)
    print(f'Writing to {file_name}')
    ds.to_pandas().to_parquet(file_name, compression='gzip')