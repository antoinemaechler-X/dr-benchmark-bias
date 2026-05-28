"""Load item embeddings produced by embed_items.ipynb."""

import json
import numpy as np


def load_item_embeddings(embeddings_path='item_embeddings.npy',
                          meta_path='item_embedding_meta.json'):
    """Load item embeddings and return (embeddings_array, item_id_to_index_dict, benchmark_ids).

    Parameters
    ----------
    embeddings_path : str
        Path to the .npy file with shape (N, dim).
    meta_path : str
        Path to the JSON metadata file.

    Returns
    -------
    embeddings : np.ndarray of shape (N, dim)
    id_to_idx : dict mapping item_id (str) → row index (int)
    benchmark_ids : list of str, benchmark_id per item (same order as rows)
    """
    embeddings = np.load(embeddings_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    id_to_idx = {item_id: i for i, item_id in enumerate(meta['item_ids'])}
    benchmark_ids = meta['benchmark_ids']

    return embeddings, id_to_idx, benchmark_ids


if __name__ == '__main__':
    emb, id_to_idx, bench = load_item_embeddings()
    print(f"Loaded {emb.shape[0]} embeddings of dim {emb.shape[1]}")
    print(f"Unique benchmarks: {len(set(bench))}")
    print(f"Sample item_ids: {list(id_to_idx.keys())[:5]}")
