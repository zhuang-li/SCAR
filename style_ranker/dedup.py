import numpy as np
from typing import List, Tuple, Dict, Any, Set
from datasketch import sha1_hash32
from nltk import word_tokenize
import re

from tqdm import tqdm

MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
NON_ALPHA = re.compile(r'\W+', re.UNICODE)

def generate_permutations(num_perm: int) -> np.ndarray:
    return np.array([
        (np.random.randint(1, MERSENNE_PRIME, dtype=np.uint64),
         np.random.randint(0, MERSENNE_PRIME, dtype=np.uint64))
        for _ in range(num_perm)
    ]).T

def tokenize(content: str, ngram_size: int) -> Set[str]:
    words = word_tokenize(content.lower())
    return set(" ".join(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1))

def embed_func(
        content: str,
        idx: int,
        *,
        num_perm: int,
        ngram_size: int,
        hashranges: List[Tuple[int, int]],
        permutations: np.ndarray,
) -> Dict[str, Any]:
    a, b = permutations
    masks: np.ndarray = np.full(shape=num_perm, dtype=np.uint64, fill_value=MAX_HASH)
    tokens: Set[str] = tokenize(content, ngram_size)
    #print(f"Document {idx} tokens: {tokens}")  # Debugging print
    hashvalues: np.ndarray = np.array([sha1_hash32(token.encode("utf-8")) for token in tokens], dtype=np.uint64)
    permuted_hashvalues = np.bitwise_and(
        ((hashvalues * np.tile(a, (len(hashvalues), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH
    )
    hashvalues = np.vstack([permuted_hashvalues, masks]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx, "__tokens__": tokens}

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def lsh_near_deduplication(documents: List[str], num_perm: int, num_bands: int, ngram_size: int, similarity_threshold: float):
    permutations = generate_permutations(num_perm)
    band_size = num_perm // num_bands
    hashranges = [(i * band_size, (i + 1) * band_size) for i in range(num_bands)]

    signatures = [
        embed_func(doc, idx, num_perm=num_perm, ngram_size=ngram_size,
                   hashranges=hashranges, permutations=permutations)
        for idx, doc in enumerate(documents)
    ]

    hash_tables = [{} for _ in range(num_bands)]
    for sig in signatures:
        for i, band in enumerate(sig['__signatures__']):
            band_hash = hash(band)
            if band_hash not in hash_tables[i]:
                hash_tables[i][band_hash] = []
            hash_tables[i][band_hash].append(sig['__id__'])

    # for i, table in enumerate(hash_tables):
    #     print(f"Hash table {i}: {table}")

    candidate_pairs = set()
    for table in hash_tables:
        for bucket in table.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i+1, len(bucket)):
                        candidate_pairs.add((min(bucket[i], bucket[j]), max(bucket[i], bucket[j])))

    #print(f"Candidate pairs: {candidate_pairs}")

    near_duplicate_pairs = []
    print(f"Comparing {len(candidate_pairs)} candidate pairs...")
    for i, j in tqdm(candidate_pairs):
        sim = jaccard_similarity(signatures[i]['__tokens__'], signatures[j]['__tokens__'])
        #print(f"Similarity between doc {i} and doc {j}: {sim:.2f}")
        if sim >= similarity_threshold:
            near_duplicate_pairs.append((i, j))

    # Add this after the LSH process
    print(f"Comparing remaining pairs...")
    for i in tqdm(range(len(documents))):
        for j in tqdm(range(i+1, len(documents))):
            if (i, j) not in candidate_pairs:
                sim = jaccard_similarity(signatures[i]['__tokens__'], signatures[j]['__tokens__'])
                if sim >= similarity_threshold:
                    near_duplicate_pairs.append((i, j))
                    #print(f"Additional pair found: Similarity between doc {i} and doc {j}: {sim:.2f}")

    #print(f"Near-duplicate pairs: {near_duplicate_pairs}")

    # Simple clustering
    clusters = []
    for i, j in near_duplicate_pairs:
        found = False
        for cluster in clusters:
            if i in cluster or j in cluster:
                cluster.update({i, j})
                found = True
                break
        if not found:
            clusters.append({i, j})

    return [list(cluster) for cluster in clusters]


def select_distinct_indexes(duplicate_clusters: List[List[int]], total_documents: int) -> List[int]:
    # Create a set of all indexes that are in some cluster
    clustered_indexes = set(idx for cluster in duplicate_clusters for idx in cluster)

    # Start with all indexes that are not in any cluster
    distinct_indexes = set(range(total_documents)) - clustered_indexes

    # Add one representative index from each cluster
    for cluster in duplicate_clusters:
        # Here we're selecting the smallest index in each cluster as the representative
        # You could use a different selection criteria if needed
        distinct_indexes.add(min(cluster))

    # Convert to a sorted list for consistent output
    return sorted(distinct_indexes)

def main():
    documents = [
        "Deduplication is so much fun!",
        "Deduplication is so much fun and easy!",
        "I wish spider dog is a thing.",
        "Completely different document.",
        "This is a longer document about deduplication. It talks about how deduplication can be used in various contexts.",
        "Another document discussing deduplication techniques and their applications in data processing.",
        "Deduplication is fun and easy!",
        "I wish spider dog was a real thing.",
    ]

    num_perm = 8
    num_bands = 4
    ngram_size = 2
    similarity_threshold = 0.5

    duplicate_clusters = lsh_near_deduplication(documents, num_perm, num_bands, ngram_size, similarity_threshold)
    print("Near-duplicate clusters:", duplicate_clusters)

    print("\nPairwise similarities:")
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            sim = jaccard_similarity(set(tokenize(documents[i], ngram_size)), set(tokenize(documents[j], ngram_size)))
            print(f"Similarity between doc {i} and doc {j}: {sim:.2f}")
            print(f"Doc {i}: {documents[i]}")
            print(f"Doc {j}: {documents[j]}")
            print()

    distinct_indexes = select_distinct_indexes(duplicate_clusters, len(documents))
    print("\nDistinct document indexes:", distinct_indexes)

    print("\nDistinct documents:")
    for idx in distinct_indexes:
        print(f"{idx}: {documents[idx]}")

if __name__ == "__main__":
    main()