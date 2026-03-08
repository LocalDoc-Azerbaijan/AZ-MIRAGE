"""
AZ-MIRAGE Retriever Benchmark

Evaluate embedding models on the AZ-MIRAGE retrieval benchmark.
Models are loaded from a text file (one HuggingFace model ID per line).

Features:
  - Auto-detects CUDA, falls back to CPU
  - Dynamic max_length from model config
  - Batch size configurable via CLI
  - Auto OOM recovery (halves batch size)
  - Metrics: F1, NDCG, Precision, Recall @ 1, 3, 5, 10
  - Final comparison table across all models
  - Results saved to JSON for further analysis

Usage:
  python evaluate_retriever.py --models models.txt --batch_size 64
  python evaluate_retriever.py --models models.txt --batch_size 32 --top_k 5
  python evaluate_retriever.py --models models.txt --data_dir dataset

Models file format (models.txt):
  BAAI/bge-m3
  intfloat/multilingual-e5-small
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Requires: pip install sentence-transformers torch numpy tqdm
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


# ============================================================================
# Device Detection
# ============================================================================
def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Device: CUDA ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU (CUDA not available)")
    return device


# ============================================================================
# Model Loader with Dynamic Config
# ============================================================================
def load_model(model_name: str, device: torch.device) -> Tuple[Any, int]:
    """
    Load a SentenceTransformer model and extract max_length from config.
    Returns (model, max_seq_length).
    """
    from sentence_transformers import SentenceTransformer

    print(f"\n  Loading model: {model_name}")
    start = time.time()

    model = SentenceTransformer(model_name, trust_remote_code=True)
    model = model.to(device)

    # Extract max_length dynamically from model config
    max_len = getattr(model, "max_seq_length", None)

    if max_len is None or max_len > 8192:
        # Fallback: try to get from tokenizer
        try:
            tokenizer = model.tokenizer
            max_len = getattr(tokenizer, "model_max_length", 512)
            if max_len > 8192:
                max_len = 512  # safety cap
        except:
            max_len = 512

    elapsed = time.time() - start
    print(f"  Loaded in {elapsed:.1f}s | max_seq_length={max_len}")

    return model, max_len


def unload_model(model: Any = None) -> None:
    """Release model from memory."""
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Data Loader
# ============================================================================
def load_benchmark_data(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load AZ-MIRAGE benchmark files (dataset + doc_pool)."""
    dataset_path = os.path.join(data_dir, "dataset.json")
    pool_path = os.path.join(data_dir, "doc_pool.json")

    for path in [dataset_path, pool_path]:
        if not os.path.exists(path):
            print(f"ERROR: Missing file: {path}")
            sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(pool_path, "r", encoding="utf-8") as f:
        doc_pool = json.load(f)

    return dataset, doc_pool


# ============================================================================
# Encoding with OOM Recovery
# ============================================================================
def encode_texts(model: Any, texts: List[str], batch_size: int,
                 device: torch.device, desc: str = "Encoding") -> np.ndarray:
    """
    Encode texts with automatic OOM recovery.
    Halves batch size on CUDA OOM, retries until batch_size=1.
    """
    current_batch = batch_size

    while current_batch >= 1:
        try:
            embeddings = model.encode(
                texts,
                batch_size=current_batch,
                device=device,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_batch > 1:
                print(f"\n  OOM with batch_size={current_batch}, reducing to {current_batch // 2}")
                current_batch = current_batch // 2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e

    raise RuntimeError("Failed to encode even with batch_size=1")


# ============================================================================
# Retrieval
# ============================================================================
def retrieve(query_embeddings: np.ndarray, chunk_embeddings: np.ndarray,
             dataset: List[Dict], doc_pool: List[Dict],
             top_k: int = 5) -> List[Dict]:
    """
    For each query, find top_k most similar chunks from the FULL pool.
    Returns list of results with top chunks and scores.
    """
    print(f"\n  Computing similarity ({len(dataset)} queries x {len(doc_pool)} chunks)...")

    # Convert to torch for fast matmul
    q_tensor = torch.from_numpy(query_embeddings).float()
    c_tensor = torch.from_numpy(chunk_embeddings).float()

    # Compute similarity in batches to avoid OOM
    results = []
    batch_size = 256

    for start in tqdm(range(0, len(dataset), batch_size), desc="  Retrieval"):
        end = min(start + batch_size, len(dataset))
        q_batch = q_tensor[start:end]

        # Cosine similarity (embeddings are already normalized)
        sim = torch.matmul(q_batch, c_tensor.T)

        for i in range(sim.size(0)):
            global_idx = start + i
            topk_scores, topk_indices = torch.topk(sim[i], min(top_k, sim.size(1)))

            top_chunks = []
            for idx in topk_indices.tolist():
                chunk = doc_pool[idx].copy()
                top_chunks.append(chunk)

            results.append({
                "query_id": dataset[global_idx]["query_id"],
                "query": dataset[global_idx]["query"],
                "top_chunks": top_chunks,
                "scores": topk_scores.tolist(),
            })

    return results


# ============================================================================
# Metrics Calculation
# ============================================================================
def calculate_dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    relevances = np.asarray(relevances, dtype=np.float64)[:k]
    if relevances.size:
        return float(np.sum(relevances / np.log2(np.arange(2, relevances.size + 2))))
    return 0.0


def calculate_ndcg_binary(relevances: List[float], total_relevant: int, k: int) -> float:
    """
    Normalized DCG with correct IDCG for binary relevance.
    IDCG is computed from the true number of relevant documents,
    not from the retrieved set.
    """
    dcg = calculate_dcg(relevances[:k], k)
    ideal_relevances = [1.0] * min(total_relevant, k)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def calculate_mrr(relevances: List[float]) -> float:
    """Reciprocal Rank for a single query (1/rank of first relevant result)."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def safe_mean(values: List[float]) -> float:
    """Safe mean that returns 0.0 for empty lists."""
    return float(np.mean(values)) if values else 0.0


def evaluate_retrieval(results: List[Dict], doc_pool: List[Dict],
                       top_k_values: List[int] = [1, 3, 5, 10],
                       max_k: int = 10) -> Dict:
    """
    Evaluate retrieval results using Precision, Recall, F1, NDCG @ k, and MRR.
    Ground truth: chunks with support=1 and matching mapped_id in doc_pool.
    """
    # Build ground truth: count of relevant chunks per query
    gt_count_by_qid = defaultdict(int)
    for p in doc_pool:
        if p["support"] == 1:
            gt_count_by_qid[p["mapped_id"]] += 1

    all_metrics = {k: [] for k in top_k_values}
    all_mrr = []

    skipped = 0
    for result in results:
        qid = result["query_id"]
        total_relevant = gt_count_by_qid.get(qid, 0)

        if total_relevant == 0:
            skipped += 1
            continue

        # Relevance list: 1 if retrieved chunk matches query and is support=1
        relevances = []
        for chunk in result["top_chunks"]:
            is_relevant = (
                chunk.get("mapped_id") == qid and
                chunk.get("support") == 1
            )
            relevances.append(1.0 if is_relevant else 0.0)

        # MRR over retrieved list (effectively MRR@top_k)
        all_mrr.append(calculate_mrr(relevances))

        for k in top_k_values:
            top_rel = relevances[:k]
            num_relevant = sum(top_rel)

            precision = num_relevant / k if k > 0 else 0
            recall = min(num_relevant / total_relevant, 1.0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            ndcg = calculate_ndcg_binary(top_rel, total_relevant, k)

            all_metrics[k].append({
                "F1": f1,
                "NDCG": ndcg,
                "Precision": precision,
                "Recall": recall,
            })

    if skipped > 0:
        print(f"  Warning: {skipped} queries skipped (no ground truth in doc_pool)")

    # Average across all queries
    avg_metrics = {}
    for k in top_k_values:
        avg_metrics[k] = {
            "F1": safe_mean([m["F1"] for m in all_metrics[k]]),
            "NDCG": safe_mean([m["NDCG"] for m in all_metrics[k]]),
            "Precision": safe_mean([m["Precision"] for m in all_metrics[k]]),
            "Recall": safe_mean([m["Recall"] for m in all_metrics[k]]),
        }

    # MRR@top_k (named with actual k for clarity)
    avg_metrics[f"MRR@{max_k}"] = safe_mean(all_mrr)

    return avg_metrics


# ============================================================================
# Results Display
# ============================================================================
def print_model_results(model_name: str, metrics: Dict) -> None:
    """Print results for a single model."""
    print(f"\n  {'='*65}")
    print(f"  Results: {model_name}")
    print(f"  {'='*65}")
    print(f"  {'@k':<6} {'F1':>10} {'NDCG':>10} {'Precision':>10} {'Recall':>10}")
    print(f"  {'-'*46}")
    for k in sorted(k for k in metrics.keys() if isinstance(k, int)):
        m = metrics[k]
        print(f"  @{k:<5} {m['F1']:>10.4f} {m['NDCG']:>10.4f} {m['Precision']:>10.4f} {m['Recall']:>10.4f}")
    # Print MRR (key is "MRR@10" or similar)
    for k, v in metrics.items():
        if isinstance(k, str) and k.startswith("MRR"):
            print(f"  {k:<6} {v:>10.4f}")


def print_comparison_table(all_results: List[Dict]) -> None:
    """Print final comparison table across all models."""
    if not all_results:
        return

    print(f"\n{'='*95}")
    print(f"FINAL COMPARISON TABLE")
    print(f"{'='*95}")

    # Determine which k values are available across all results
    available_ks = set()
    for r in all_results:
        for k in r["metrics"]:
            if isinstance(k, int):
                available_ks.add(k)
    available_ks = sorted(available_ks)

    # Find MRR key (could be "MRR@10", "MRR@5", etc.)
    mrr_key = None
    for r in all_results:
        for k in r["metrics"]:
            if isinstance(k, str) and k.startswith("MRR"):
                mrr_key = k
                break
        if mrr_key:
            break
    if mrr_key is None:
        mrr_key = "MRR@10"  # fallback label

    # Build header dynamically
    max_k = max(available_ks) if available_ks else 10
    header = f"{'Model':<45} {mrr_key:>8} {'P@1':>7}"
    for k in available_ks:
        if k > 1:
            header += f" {'R@'+str(k):>7}"
    for k in available_ks:
        if k > 1:
            header += f" {'NDCG@'+str(k):>8}"
    print(header)
    print("-" * len(header))

    # Sort by MRR descending
    def get_mrr(r):
        for k, v in r["metrics"].items():
            if isinstance(k, str) and k.startswith("MRR"):
                return v
        return 0

    sorted_results = sorted(all_results, key=get_mrr, reverse=True)

    for i, result in enumerate(sorted_results):
        name = result["model_name"]
        if len(name) > 44:
            name = "..." + name[-41:]

        mrr = get_mrr(result)
        m1 = result["metrics"].get(1, {})

        row = f"#{i+1:<3}{name:<41} {mrr:>8.4f} {m1.get('Precision',0):>7.4f}"
        for k in available_ks:
            if k > 1:
                mk = result["metrics"].get(k, {})
                row += f" {mk.get('Recall',0):>7.4f}"
        for k in available_ks:
            if k > 1:
                mk = result["metrics"].get(k, {})
                row += f" {mk.get('NDCG',0):>8.4f}"
        print(row)

    print(f"{'='*95}")
    print(f"Ranked by {mrr_key} (descending)")


# ============================================================================
# Main Pipeline
# ============================================================================
def run_single_model(model_name: str, dataset: List[Dict], doc_pool: List[Dict],
                     batch_size: int, device: torch.device, top_k: int,
                     results_dir: str) -> Optional[Dict]:
    """Run full evaluation pipeline for a single model."""
    model_id = model_name.replace("/", "_")
    result_path = os.path.join(results_dir, f"{model_id}_results.json")

    # Check if already evaluated
    if os.path.exists(result_path):
        print(f"\n  [{model_name}] Already evaluated, loading cached results...")
        with open(result_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        # Convert string keys back to int for metrics
        if "metrics" in cached:
            cached["metrics"] = {
                (int(k) if k.isdigit() else k): v
                for k, v in cached["metrics"].items()
            }
        return cached

    print(f"\n{'='*70}")
    print(f"  EVALUATING: {model_name}")
    print(f"{'='*70}")

    start_total = time.time()

    try:
        # Load model
        model, max_len = load_model(model_name, device)
        print(f"  max_seq_length: {max_len}")

        # Prepare texts
        queries = [d["query"] for d in dataset]
        chunks = [p["doc_chunk"] for p in doc_pool]
        print(f"  Queries: {len(queries):,} | Chunks: {len(chunks):,}")

        # Encode queries
        print(f"\n  Encoding queries...")
        t0 = time.time()
        query_embeddings = encode_texts(model, queries, batch_size, device, "Queries")
        t_queries = time.time() - t0
        print(f"  Query encoding: {t_queries:.1f}s ({len(queries)/t_queries:.0f} q/s)")

        # Encode chunks
        print(f"\n  Encoding chunks...")
        t0 = time.time()
        chunk_embeddings = encode_texts(model, chunks, batch_size, device, "Chunks")
        t_chunks = time.time() - t0
        print(f"  Chunk encoding: {t_chunks:.1f}s ({len(chunks)/t_chunks:.0f} c/s)")

        # Retrieve
        retrieval_results = retrieve(query_embeddings, chunk_embeddings, dataset, doc_pool, top_k)

        # Evaluate — only compute metrics for k <= top_k
        eval_ks = [k for k in [1, 3, 5, 10] if k <= top_k]
        metrics = evaluate_retrieval(retrieval_results, doc_pool, eval_ks, max_k=top_k)
        print_model_results(model_name, metrics)

        # Unload model
        unload_model(model)

        elapsed = time.time() - start_total
        print(f"\n  Total time for {model_name}: {elapsed:.1f}s")

        # Save results
        result = {
            "model_name": model_name,
            "max_seq_length": max_len,
            "batch_size": batch_size,
            "device": str(device),
            "num_queries": len(queries),
            "num_chunks": len(chunks),
            "top_k": top_k,
            "metrics": {str(k): v for k, v in metrics.items()},
            "timing": {
                "query_encoding_sec": round(t_queries, 1),
                "chunk_encoding_sec": round(t_chunks, 1),
                "total_sec": round(elapsed, 1),
            },
        }

        os.makedirs(results_dir, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {result_path}")

        # Return with int keys for comparison
        result["metrics"] = metrics
        return result

    except Exception as e:
        print(f"\n  ERROR evaluating {model_name}: {e}")
        unload_model(None) if torch.cuda.is_available() else None
        return None


def main():
    parser = argparse.ArgumentParser(
        description="AZ-MIRAGE Retriever Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_retriever.py --models models.txt --batch_size 64
  python evaluate_retriever.py --models models.txt --batch_size 32 --top_k 5
  python evaluate_retriever.py --models models.txt --data_dir dataset
        """
    )
    parser.add_argument("--models", type=str, required=True,
                        help="Path to text file with model names (one per line)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for encoding (default: 64)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-K for retrieval (default: 10)")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory with benchmark data (default: dataset)")
    parser.add_argument("--results_dir", type=str, default="evaluation_results",
                        help="Directory to save results (default: evaluation_results)")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load model list
    # -------------------------------------------------------------------------
    if not os.path.exists(args.models):
        print(f"ERROR: Models file not found: {args.models}")
        sys.exit(1)

    with open(args.models, "r", encoding="utf-8") as f:
        model_names = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not model_names:
        print("ERROR: No models found in file")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  AZ-MIRAGE Retriever Benchmark")
    print(f"{'='*70}")
    print(f"  Models to evaluate: {len(model_names)}")
    for i, name in enumerate(model_names, 1):
        print(f"    {i}. {name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Data dir: {args.data_dir}")

    # -------------------------------------------------------------------------
    # Detect device
    # -------------------------------------------------------------------------
    device = get_device()

    # -------------------------------------------------------------------------
    # Load benchmark data
    # -------------------------------------------------------------------------
    print(f"\n  Loading benchmark data from {args.data_dir}...")
    dataset, doc_pool = load_benchmark_data(args.data_dir)
    print(f"  Dataset: {len(dataset):,} queries")
    print(f"  Doc pool: {len(doc_pool):,} chunks")

    # -------------------------------------------------------------------------
    # Evaluate each model
    # -------------------------------------------------------------------------
    all_results = []

    for i, model_name in enumerate(model_names, 1):
        print(f"\n{'#'*70}")
        print(f"  MODEL {i}/{len(model_names)}: {model_name}")
        print(f"{'#'*70}")

        result = run_single_model(
            model_name, dataset, doc_pool,
            args.batch_size, device, args.top_k,
            args.results_dir
        )

        if result:
            all_results.append(result)

    # -------------------------------------------------------------------------
    # Final comparison
    # -------------------------------------------------------------------------
    print_comparison_table(all_results)

    # Save combined results
    combined_path = os.path.join(args.results_dir, "comparison.json")
    os.makedirs(args.results_dir, exist_ok=True)

    # Convert metrics keys to strings for JSON
    save_results = []
    for r in all_results:
        r_copy = r.copy()
        r_copy["metrics"] = {str(k): v for k, v in r["metrics"].items()}
        save_results.append(r_copy)

    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Combined results saved: {combined_path}")
    print(f"\nDone! Evaluated {len(all_results)}/{len(model_names)} models.")


if __name__ == "__main__":
    main()