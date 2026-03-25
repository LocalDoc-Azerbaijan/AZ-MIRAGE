# AZ-MIRAGE: Azerbaijani Retrieval-Augmented Generation Evaluation Benchmark

AZ-MIRAGE is a retrieval benchmark for evaluating embedding models on Azerbaijani text. It is built from native Azerbaijani data (LDQuAd_v2 dataset, sourced from Azerbaijani Wikipedia) and follows the methodology of the [MIRAGE benchmark](https://github.com/nlpai-lab/MIRAGE) (Park et al., NAACL 2025 Findings), with several improvements to noise quality and benchmark difficulty.


## Benchmark Statistics

| Property | Value |
|---|---|
| Language | Azerbaijani |
| QA pairs | 7,373 |
| Document pool | 40,448 chunks |
| Oracle chunks (support=1) | 7,373 |
| Noise chunks (support=0) | 33,075 |
| LLM-generated decoys | 7,349 |
| Avg chunks per query | 5.5 |
| Unique articles | 2,340 |
| Source | LDQuAd_v2 (Azerbaijani Wikipedia) |


## Retrieval Results

All models were evaluated on a single GPU with batch size 128 and top_k=10. Metrics: MRR@10, Precision@1, Recall@k, NDCG@k.

### Ranking by MRR@10

| Rank | Model | P@1 | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR@10 |
|------|-------|-----|-----|------|--------|---------|--------|
| 1 | LocalDoc/LocRet-small | 0.3132 | 0.8267 | 0.8948 | 0.5938 | 0.6162 | 0.5250 |
| 2 | BAAI/bge-m3 | 0.2310 | 0.6905 | 0.7787 | 0.4791 | 0.5079 | 0.4204 |
| 3 | perplexity-ai/pplx-embed-v1-0.6b | 0.2276 | 0.6715 | 0.7605 | 0.4677 | 0.4968 | 0.4117 |
| 4 | intfloat/multilingual-e5-large | 0.2264 | 0.6571 | 0.7454 | 0.4584 | 0.4875 | 0.4043 |
| 5 | intfloat/multilingual-e5-base | 0.2116 | 0.6353 | 0.7216 | 0.4390 | 0.4672 | 0.3852 |
| 6 | Snowflake/snowflake-arctic-embed-l-v2.0 | 0.2135 | 0.6006 | 0.6916 | 0.4218 | 0.4516 | 0.3746 |
| 7 | Qwen/Qwen3-Embedding-4B | 0.1869 | 0.6067 | 0.7036 | 0.4119 | 0.4437 | 0.3602 |
| 8 | intfloat/multilingual-e5-small | 0.1958 | 0.5927 | 0.6834 | 0.4079 | 0.4375 | 0.3586 |
| 9 | Qwen/Qwen3-Embedding-0.6B | 0.1516 | 0.4926 | 0.5956 | 0.3339 | 0.3676 | 0.2951 |
| 10 | LocalDoc/az-en-MiniLM-L6-v2 | 0.1324 | 0.4445 | 0.5427 | 0.2972 | 0.3293 | 0.2617 |
| 11 | LocalDoc/TEmA-small | 0.1031 | 0.3889 | 0.4804 | 0.2534 | 0.2832 | 0.2208 |
| 12 | sentence-transformers/LaBSE | 0.0943 | 0.3331 | 0.4145 | 0.2208 | 0.2472 | 0.1944 |
| 13 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 0.0650 | 0.2312 | 0.2927 | 0.1530 | 0.1730 | 0.1353 |
| 14 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 0.0366 | 0.1381 | 0.1786 | 0.0902 | 0.1032 | 0.0796 |
| 15 | sentence-transformers/all-MiniLM-L6-v2 | 0.0148 | 0.0556 | 0.0735 | 0.0355 | 0.0413 | 0.0313 |



## How AZ-MIRAGE Was Built

### Data Source

AZ-MIRAGE is built from [LDQuAd_v2](https://huggingface.co/datasets/LocalDoc/LDQuAd_v2), a question-answering dataset containing 350,991 QA pairs across 4,702 Azerbaijani Wikipedia articles. Unlike MIRAGE which draws from five English QA datasets (PopQA, NaturalQA, TriviaQA, DROP, IfQA), AZ-MIRAGE uses a single comprehensive Azerbaijani source, ensuring linguistic consistency.

### Step 1: Data Preparation

LDQuAd_v2 contains pre-chunked articles: the same title with different content fields represents different chunks from the same article, while the same title and content with different questions represents multiple QA pairs for one chunk.

The preparation step:
- Extracted 4,702 unique articles containing 42,220 unique chunks
- Identified 350,976 unique QA pairs (after deduplication)
- Filtered to 2,348 articles with 5 or more chunks (minimum needed for 1 oracle + 4 noise)
- Result: 36,468 chunks and 300,220 QA pairs

### Step 2: Retrieval Pool Construction

For each QA pair, the oracle chunk (the chunk containing the answer) was verified through substring matching -- confirming the answer text literally appears in the linked chunk. 73,302 QA pairs (24.4%) passed verification. The remaining 75.6% were excluded because LDQuAd_v2 answers are often paraphrased rather than extracted verbatim.

Four noise chunks were selected from the same article as the oracle chunk. Since all five chunks share the same topic, entity names, and vocabulary, distinguishing the oracle from noise requires understanding the specific information requested -- not just topical similarity.

The dataset was sampled to 7,560 QA pairs balanced across articles.

### Step 3: LLM Validation and Hardening

This is where AZ-MIRAGE goes beyond the original MIRAGE methodology. Three LLM-based quality checks were applied:

**Noise validation.** Each of the 30,240 noise chunks was checked by an LLM: "Can this chunk answer the question?" 3,854 chunks (12.7%) were found to leak the answer and were removed. Without this step, models would be penalized for correctly identifying relevant information that was mislabeled as noise.

**Easy question filtering.** Each question was presented to an LLM without any context. 187 questions (2.5%) were answered correctly and removed. These questions test the model's parametric knowledge rather than its retrieval capability, which is not the purpose of a retrieval benchmark.

**Decoy generation.** For each query, the LLM rewrote the oracle chunk with the answer removed or replaced with incorrect information. This produced 7,349 adversarial decoy chunks that are semantically near-identical to the oracle but do not contain the correct answer. These are the hardest possible negatives for a retrieval model.

The final benchmark contains 7,373 validated QA pairs with 40,448 pool entries (7,373 oracle + 25,726 same-article noise + 7,349 LLM-generated decoys).


## How AZ-MIRAGE Differs from MIRAGE

| Aspect | MIRAGE | AZ-MIRAGE |
|--------|--------|-----------|
| Language | English | Azerbaijani |
| Source data | 5 English QA datasets (PopQA, NQ, TriviaQA, DROP, IfQA) | Native Azerbaijani QA (LDQuAd_v2) |
| Content origin | English Wikipedia | Azerbaijani Wikipedia |
| QA pairs | 7,560 | 7,373 |
| Document pool | 37,800 | 40,448 |
| Chunks per query | 5.0 | 5.5 |
| Noise source | Same-article chunks only | Same-article chunks + LLM-generated decoys |
| Noise validation | LLM labeling (Command-R) | LLM validation with leaked noise removal (12.7% removed) |
| Easy question filter | LLM inference validation (Llama-3.1-8B) | LLM closed-book check (2.5% removed) |
| Adversarial decoys | No | Yes -- LLM rewrites oracle without answer |
| Human validation | 100 samples, 95% agreement | Not yet conducted |
| Evaluation metrics | F1, NDCG, Precision, Recall | Precision, Recall, NDCG (corrected IDCG), MRR, F1 |

Key improvements over MIRAGE:

**LLM-generated adversarial decoys.** MIRAGE uses only same-article chunks as noise. AZ-MIRAGE adds decoy chunks where the LLM has rewritten the oracle text to remove the answer while preserving everything else. This tests whether models can distinguish between "text about the right topic" and "text that actually answers the question."

**Validated noise quality.** MIRAGE uses automated labeling but does not verify that noise chunks truly lack the answer. AZ-MIRAGE explicitly checks each noise chunk and removes those that contain the answer (12.7% of noise was mislabeled).

**Corrected NDCG computation.** The IDCG is computed from the true number of relevant documents in the collection, not from the retrieved set. This follows the standard definition and avoids inflated scores.


## Evaluation Metrics

AZ-MIRAGE reports the following retrieval metrics at k = 1, 3, 5, 10:

- **Precision@k**: fraction of top-k results that are relevant
- **Recall@k**: fraction of relevant documents found in top-k
- **NDCG@k**: normalized discounted cumulative gain with binary relevance and correct IDCG
- **MRR@k**: mean reciprocal rank -- average of 1/position of first relevant result
- **F1@k**: harmonic mean of Precision@k and Recall@k

Ground truth is defined by `mapped_id` (linking chunk to query) and `support == 1` (marking oracle chunks) in `doc_pool.json`.



## Usage

### Requirements

```
pip install datasets sentence-transformers torch numpy tqdm
```

### Evaluate retriever models

```
python evaluate.py --models models.txt --batch_size 64
```

Arguments:
- `--models`: path to text file with HuggingFace model IDs, one per line (lines starting with # are ignored)
- `--batch_size`: encoding batch size (default: 64, automatically halved on GPU OOM)
- `--top_k`: number of chunks to retrieve per query (default: 10)
- `--data_dir`: path to benchmark data directory (default: dataset)
- `--results_dir`: path to save evaluation results (default: evaluation_results)

Features:
- Automatic CUDA detection with CPU fallback
- Dynamic max_seq_length from model config
- OOM recovery with automatic batch size reduction
- Per-model result caching (delete evaluation_results/ to recompute)
- Final comparison table sorted by MRR


## Citation

If you use AZ-MIRAGE in your research, please cite:

```bibtex
@misc{az-mirage-2025,
  title={AZ-MIRAGE: Azerbaijani Retrieval-Augmented Generation Evaluation Benchmark},
  year={2025},
  url={https://github.com/LocalDoc-Azerbaijan/AZ-MIRAGE}
}
```

This work builds upon the MIRAGE benchmark:

```bibtex
@inproceedings{park2025mirage,
  title={MIRAGE: A Metric-Intensive Benchmark for Retrieval-Augmented Generation Evaluation},
  author={Park, Chanhee and Moon, Hyeonseok and Park, Chanjun and Lim, Heuiseok},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={2883--2900},
  year={2025}
}
```


## License

The benchmark data and code is released under the Apache 2.0 license.
