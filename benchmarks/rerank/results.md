# Reranker results

Completed 129 of 129 task snapshots. First stage: hips CodeRankEmbed + BM25 (`hybrid_full`, weighted alpha 0.15, 200 semantic candidates, exact vectors). `rerank_full` is the legacy `--mode rerank` (cosine on BM25 candidates).

Better/worse counts compare each query's reciprocal file rank with `hybrid_full`.

## agent queries (n=37)

| Variant | R@1 | R@5 | R@10 | MRR | Better / worse vs hybrid | Rerank ms (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.405 | 0.703 | 0.757 | 0.536 | 5 / 9 | — |
| semantic_full | 0.378 | 0.676 | 0.757 | 0.525 | 4 / 10 | — |
| hybrid_full | 0.486 | 0.703 | 0.838 | 0.600 | — | — |
| rerank_full | 0.486 | 0.676 | 0.811 | 0.587 | 2 / 2 | — |
| gte-modernbert | bm25 top20 | 0.378 | 0.703 | 0.784 | 0.502 | 6 / 10 | 2332 |
| jina-v1-turbo | bm25 top20 | 0.514 | 0.703 | 0.784 | 0.599 | 9 / 7 | 631 |
| jina-v2-base | bm25 top20 | 0.486 | 0.676 | 0.784 | 0.571 | 8 / 8 | 1542 |
| minilm-l6 | bm25 top20 | 0.378 | 0.703 | 0.784 | 0.514 | 6 / 11 | 230 |
| mxbai-base | bm25 top20 | 0.432 | 0.622 | 0.730 | 0.531 | 4 / 12 | 5364 |
| mxbai-xsmall | bm25 top20 | 0.432 | 0.649 | 0.784 | 0.546 | 7 / 10 | 2574 |
| gte-modernbert | hybrid top20 | 0.378 | 0.730 | 0.838 | 0.533 | 6 / 7 | 2523 |
| jina-v1-turbo | hybrid top20 | 0.459 | 0.784 | 0.838 | 0.595 | 9 / 7 | 578 |
| jina-v2-base | hybrid top20 | 0.486 | 0.730 | 0.811 | 0.593 | 8 / 5 | 1522 |
| minilm-l6 | hybrid top20 | 0.351 | 0.784 | 0.838 | 0.524 | 6 / 10 | 231 |
| mxbai-base | hybrid top20 | 0.432 | 0.676 | 0.811 | 0.549 | 3 / 10 | 5371 |
| mxbai-xsmall | hybrid top20 | 0.432 | 0.703 | 0.838 | 0.553 | 8 / 9 | 2463 |
| gte-modernbert | hybrid top20 rrf | 0.432 | 0.730 | 0.811 | 0.569 | 3 / 5 | 2523 |
| jina-v1-turbo | hybrid top20 rrf | 0.459 | 0.730 | 0.838 | 0.593 | 6 / 5 | 578 |
| jina-v2-base | hybrid top20 rrf | 0.459 | 0.730 | 0.811 | 0.579 | 4 / 5 | 1522 |
| minilm-l6 | hybrid top20 rrf | 0.459 | 0.730 | 0.838 | 0.586 | 7 / 6 | 231 |
| mxbai-base | hybrid top20 rrf | 0.486 | 0.730 | 0.811 | 0.596 | 4 / 7 | 5371 |
| mxbai-xsmall | hybrid top20 rrf | 0.432 | 0.703 | 0.838 | 0.564 | 4 / 8 | 2463 |
| gte-modernbert | hybrid top50 | 0.378 | 0.703 | 0.784 | 0.513 | 4 / 12 | 5717 |
| jina-v1-turbo | hybrid top50 | 0.486 | 0.757 | 0.865 | 0.602 | 10 / 7 | 1293 |
| jina-v2-base | hybrid top50 | 0.486 | 0.703 | 0.865 | 0.597 | 10 / 6 | 3920 |
| minilm-l6 | hybrid top50 | 0.378 | 0.757 | 0.784 | 0.526 | 6 / 11 | 583 |
| mxbai-base | hybrid top50 | 0.459 | 0.757 | 0.811 | 0.590 | 6 / 10 | 14373 |
| mxbai-xsmall | hybrid top50 | 0.432 | 0.703 | 0.811 | 0.564 | 8 / 10 | 6058 |
| gte-modernbert | hybrid top50 rrf | 0.432 | 0.703 | 0.811 | 0.568 | 2 / 6 | 5717 |
| jina-v1-turbo | hybrid top50 rrf | 0.459 | 0.757 | 0.838 | 0.595 | 7 / 5 | 1293 |
| jina-v2-base | hybrid top50 rrf | 0.459 | 0.703 | 0.811 | 0.583 | 7 / 4 | 3920 |
| minilm-l6 | hybrid top50 rrf | 0.459 | 0.757 | 0.811 | 0.585 | 6 / 7 | 583 |
| mxbai-base | hybrid top50 rrf | 0.486 | 0.703 | 0.811 | 0.583 | 5 / 9 | 14373 |
| mxbai-xsmall | hybrid top50 rrf | 0.432 | 0.703 | 0.838 | 0.547 | 5 / 10 | 6058 |

## idq queries (n=129)

| Variant | R@1 | R@5 | R@10 | MRR | Better / worse vs hybrid | Rerank ms (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.256 | 0.550 | 0.682 | 0.390 | 34 / 30 | — |
| semantic_full | 0.217 | 0.419 | 0.527 | 0.309 | 14 / 39 | — |
| hybrid_full | 0.256 | 0.527 | 0.620 | 0.372 | — | — |
| rerank_full | 0.256 | 0.535 | 0.659 | 0.375 | 13 / 3 | — |
| gte-modernbert | bm25 top20 | 0.302 | 0.605 | 0.698 | 0.434 | 41 / 24 | 2332 |
| jina-v1-turbo | bm25 top20 | 0.326 | 0.620 | 0.705 | 0.443 | 45 / 21 | 631 |
| jina-v2-base | bm25 top20 | 0.302 | 0.597 | 0.682 | 0.434 | 35 / 24 | 1542 |
| minilm-l6 | bm25 top20 | 0.264 | 0.589 | 0.682 | 0.402 | 41 / 23 | 230 |
| mxbai-base | bm25 top20 | 0.341 | 0.581 | 0.674 | 0.443 | 43 / 26 | 5364 |
| mxbai-xsmall | bm25 top20 | 0.287 | 0.612 | 0.698 | 0.430 | 39 / 26 | 2574 |
| gte-modernbert | hybrid top20 | 0.287 | 0.543 | 0.636 | 0.405 | 30 / 20 | 2523 |
| jina-v1-turbo | hybrid top20 | 0.326 | 0.566 | 0.636 | 0.431 | 35 / 18 | 578 |
| jina-v2-base | hybrid top20 | 0.310 | 0.566 | 0.636 | 0.421 | 32 / 19 | 1522 |
| minilm-l6 | hybrid top20 | 0.279 | 0.535 | 0.643 | 0.405 | 33 / 19 | 231 |
| mxbai-base | hybrid top20 | 0.302 | 0.519 | 0.597 | 0.397 | 33 / 24 | 5371 |
| mxbai-xsmall | hybrid top20 | 0.279 | 0.550 | 0.612 | 0.391 | 28 / 26 | 2463 |
| gte-modernbert | hybrid top20 rrf | 0.264 | 0.550 | 0.636 | 0.390 | 25 / 14 | 2523 |
| jina-v1-turbo | hybrid top20 rrf | 0.318 | 0.550 | 0.620 | 0.418 | 29 / 14 | 578 |
| jina-v2-base | hybrid top20 rrf | 0.256 | 0.543 | 0.643 | 0.380 | 28 / 17 | 1522 |
| minilm-l6 | hybrid top20 rrf | 0.287 | 0.535 | 0.620 | 0.401 | 25 / 13 | 231 |
| mxbai-base | hybrid top20 rrf | 0.248 | 0.512 | 0.628 | 0.368 | 22 / 22 | 5371 |
| mxbai-xsmall | hybrid top20 rrf | 0.271 | 0.535 | 0.605 | 0.387 | 23 / 17 | 2463 |
| gte-modernbert | hybrid top50 | 0.295 | 0.512 | 0.620 | 0.402 | 32 / 25 | 5717 |
| jina-v1-turbo | hybrid top50 | 0.318 | 0.543 | 0.628 | 0.417 | 33 / 24 | 1293 |
| jina-v2-base | hybrid top50 | 0.318 | 0.543 | 0.605 | 0.416 | 29 / 25 | 3920 |
| minilm-l6 | hybrid top50 | 0.264 | 0.527 | 0.628 | 0.387 | 32 / 27 | 583 |
| mxbai-base | hybrid top50 | 0.318 | 0.519 | 0.589 | 0.407 | 33 / 26 | 14373 |
| mxbai-xsmall | hybrid top50 | 0.279 | 0.566 | 0.628 | 0.390 | 31 / 31 | 6058 |
| gte-modernbert | hybrid top50 rrf | 0.264 | 0.558 | 0.628 | 0.387 | 27 / 19 | 5717 |
| jina-v1-turbo | hybrid top50 rrf | 0.310 | 0.550 | 0.620 | 0.417 | 33 / 17 | 1293 |
| jina-v2-base | hybrid top50 rrf | 0.248 | 0.550 | 0.612 | 0.378 | 29 / 19 | 3920 |
| minilm-l6 | hybrid top50 rrf | 0.295 | 0.543 | 0.636 | 0.408 | 29 / 19 | 583 |
| mxbai-base | hybrid top50 rrf | 0.248 | 0.519 | 0.612 | 0.365 | 24 / 20 | 14373 |
| mxbai-xsmall | hybrid top50 rrf | 0.264 | 0.558 | 0.620 | 0.382 | 27 / 21 | 6058 |

## issue queries (n=129)

| Variant | R@1 | R@5 | R@10 | MRR | Better / worse vs hybrid | Rerank ms (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.209 | 0.473 | 0.550 | 0.330 | 14 / 66 | — |
| semantic_full | 0.287 | 0.605 | 0.721 | 0.420 | 35 / 46 | — |
| hybrid_full | 0.341 | 0.659 | 0.791 | 0.488 | — | — |
| rerank_full | 0.364 | 0.690 | 0.752 | 0.502 | 15 / 10 | — |
| gte-modernbert | bm25 top20 | 0.302 | 0.519 | 0.574 | 0.401 | 23 / 50 | 2332 |
| jina-v1-turbo | bm25 top20 | 0.279 | 0.519 | 0.574 | 0.389 | 24 / 54 | 631 |
| jina-v2-base | bm25 top20 | 0.349 | 0.535 | 0.581 | 0.438 | 26 / 50 | 1542 |
| minilm-l6 | bm25 top20 | 0.287 | 0.527 | 0.581 | 0.382 | 24 / 60 | 230 |
| mxbai-base | bm25 top20 | 0.287 | 0.558 | 0.589 | 0.395 | 27 / 53 | 5364 |
| mxbai-xsmall | bm25 top20 | 0.256 | 0.535 | 0.581 | 0.378 | 26 / 54 | 2574 |
| gte-modernbert | hybrid top20 | 0.403 | 0.674 | 0.783 | 0.527 | 41 / 27 | 2523 |
| jina-v1-turbo | hybrid top20 | 0.333 | 0.667 | 0.791 | 0.474 | 39 / 35 | 578 |
| jina-v2-base | hybrid top20 | 0.411 | 0.705 | 0.775 | 0.542 | 44 / 28 | 1522 |
| minilm-l6 | hybrid top20 | 0.287 | 0.643 | 0.752 | 0.434 | 34 / 44 | 231 |
| mxbai-base | hybrid top20 | 0.349 | 0.659 | 0.791 | 0.488 | 37 / 39 | 5371 |
| mxbai-xsmall | hybrid top20 | 0.287 | 0.713 | 0.806 | 0.472 | 41 / 37 | 2463 |
| gte-modernbert | hybrid top20 rrf | 0.349 | 0.705 | 0.791 | 0.498 | 34 / 26 | 2523 |
| jina-v1-turbo | hybrid top20 rrf | 0.326 | 0.690 | 0.783 | 0.495 | 32 / 20 | 578 |
| jina-v2-base | hybrid top20 rrf | 0.388 | 0.721 | 0.791 | 0.530 | 37 / 17 | 1522 |
| minilm-l6 | hybrid top20 rrf | 0.295 | 0.698 | 0.791 | 0.460 | 33 / 33 | 231 |
| mxbai-base | hybrid top20 rrf | 0.310 | 0.705 | 0.806 | 0.482 | 34 / 31 | 5371 |
| mxbai-xsmall | hybrid top20 rrf | 0.333 | 0.713 | 0.798 | 0.503 | 38 / 25 | 2463 |
| gte-modernbert | hybrid top50 | 0.357 | 0.620 | 0.713 | 0.478 | 35 / 37 | 5717 |
| jina-v1-turbo | hybrid top50 | 0.271 | 0.589 | 0.713 | 0.407 | 33 / 49 | 1293 |
| jina-v2-base | hybrid top50 | 0.403 | 0.682 | 0.767 | 0.530 | 48 / 29 | 3920 |
| minilm-l6 | hybrid top50 | 0.264 | 0.605 | 0.713 | 0.402 | 36 / 49 | 583 |
| mxbai-base | hybrid top50 | 0.310 | 0.612 | 0.698 | 0.437 | 35 / 53 | 14373 |
| mxbai-xsmall | hybrid top50 | 0.240 | 0.643 | 0.744 | 0.424 | 40 / 44 | 6058 |
| gte-modernbert | hybrid top50 rrf | 0.388 | 0.667 | 0.760 | 0.523 | 36 / 27 | 5717 |
| jina-v1-turbo | hybrid top50 rrf | 0.295 | 0.713 | 0.783 | 0.470 | 38 / 28 | 1293 |
| jina-v2-base | hybrid top50 rrf | 0.411 | 0.721 | 0.767 | 0.538 | 40 / 19 | 3920 |
| minilm-l6 | hybrid top50 rrf | 0.310 | 0.667 | 0.775 | 0.458 | 36 / 37 | 583 |
| mxbai-base | hybrid top50 rrf | 0.326 | 0.674 | 0.760 | 0.485 | 36 / 37 | 14373 |
| mxbai-xsmall | hybrid top50 rrf | 0.341 | 0.698 | 0.791 | 0.503 | 40 / 27 | 6058 |

## Models

| Name | Hugging Face | Params | Device |
| --- | --- | ---: | --- |
| minilm-l6 | `cross-encoder/ms-marco-MiniLM-L6-v2` | 23M | mps float16 |
| jina-v1-turbo | `jinaai/jina-reranker-v1-turbo-en` | 38M | mps float16 |
| mxbai-xsmall | `mixedbread-ai/mxbai-rerank-xsmall-v1` | 71M | cpu float32 |
| mxbai-base | `mixedbread-ai/mxbai-rerank-base-v1` | 184M | cpu float32 |
| gte-modernbert | `Alibaba-NLP/gte-reranker-modernbert-base` | 150M | mps float16 |
| jina-v2-base | `jinaai/jina-reranker-v2-base-multilingual` | 278M | mps float16 |

Latency is PyTorch (sentence-transformers) on an Apple M3, scoring all pool chunks in one batch at max 512 tokens; it is not a ggml implementation.
