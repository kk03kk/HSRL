"""Offline TF-IDF/LSA lexical rule features; no claim of pretrained language semantics."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
import taverngym as tg


def build_assets(path, dimensions=128):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    catalog = {c['id']: c for c in json.loads(tg.catalog_json())}
    documents = []
    vocab = []
    for i in range(1, tg.N_CARDS):
        card = catalog[tg.card_info(i)[0]]
        words = re.findall(r'[a-z]+|\d+', card['text'].lower())
        documents.append(words + [a+'_'+b for a, b in zip(words, words[1:])])
        vocab.append(card['id'])
    from collections import Counter
    df = Counter(token for words in documents for token in set(words))
    terms = sorted(df, key=lambda x: (-df[x], x))[:4096]
    index = {t: i for i, t in enumerate(terms)}
    matrix = np.zeros((len(documents), len(terms)), dtype=np.float64)
    for i, words in enumerate(documents):
        for word, count in Counter(words).items():
            if word in index:
                matrix[i, index[word]] = (1+np.log(count))*np.log(1+len(documents)/(1+df[word]))
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    # Deterministic randomized SVD, generated once and fingerprinted in the release.
    rank = min(dimensions+16, *matrix.shape)
    omega = np.random.default_rng(2701).standard_normal((matrix.shape[1], rank))
    q, _ = np.linalg.qr(matrix @ omega)
    for _ in range(2):
        q, _ = np.linalg.qr(matrix @ (matrix.T @ q))
    u, singular, _ = np.linalg.svd(q.T @ matrix, full_matrices=False)
    values = (q @ u[:, :dimensions]) * singular[:dimensions]
    for j in range(values.shape[1]):
        if values[np.abs(values[:, j]).argmax(), j] < 0:
            values[:, j] *= -1
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    padded = np.zeros((tg.N_CARDS, dimensions), dtype='<f4')
    padded[1:, :values.shape[1]] = values.astype('<f4')
    meta = {'format': 'taverngym.rules-lsa.v1', 'dimensions': dimensions,
            'vocabulary': vocab, 'terms': terms,
            'catalog_sha256': hashlib.sha256(tg.catalog_json().encode()).hexdigest(),
            'notice': 'Lexical TF-IDF latent features; no pretrained encoder, no human trajectories.'}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        np.savez_compressed(f, vectors=padded, metadata=json.dumps(meta))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **{k: v for k, v in meta.items() if k not in ('vocabulary', 'terms')}}


@lru_cache(maxsize=4)
def load_assets(path, expected_sha):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('rule asset checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        values = data['vectors'].copy()
    vocab = [tg.card_info(i)[0] for i in range(1, tg.N_CARDS)]
    if meta.get('catalog_sha256') != hashlib.sha256(tg.catalog_json().encode()).hexdigest():
        raise ValueError('rule asset catalog mismatch')
    if meta['vocabulary'] != vocab or values.shape != (tg.N_CARDS, 128) or not np.isfinite(values).all():
        raise ValueError('rule asset vocabulary, dimensions or values mismatch')
    return torch.from_numpy(values)
