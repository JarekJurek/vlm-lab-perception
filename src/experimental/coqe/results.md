Computed from the two `predictions.jsonl` files using the current strict parser.

| Model | Task | Processed | Valid parsed | Invalid | Full test size | Status |
|---|---|---:|---:|---:|---:|---|
| 8B | Volume | 3,000 | 3,000 | 0 | 3,000 | complete |
| 8B | Content | 3,000 | 3,000 | 0 | 3,000 | complete |
| 8B | Comparative | 3,000 | 3,000 | 0 | 3,000 | complete |
| 8B | Pouring | 15,000 | 15,000 | 0 | 15,000 | complete |
| 30B | Volume | 3,000 | 2,854 | 146 | 3,000 | complete |
| 30B | Content | 3,000 | 2,997 | 3 | 3,000 | complete |
| 30B | Comparative | 3,000 | 2,999 | 1 | 3,000 | complete |
| 30B | Pouring | 15,000 | 14 | 14,986 | 15,000 | complete |

| Task | Metric | Qwen 8B | Qwen 30B | Paper best model | Best |
|---|---|---:|---:|---:|---|
| Volume | Avg per-class accuracy | 24.95% | **25.97%** | 17.79% | **Qwen 30B** |
| Content | Avg per-class accuracy | **38.00%** | 29.95% | 32.01% | **Qwen 8B** |
| Comparative | Avg per-class accuracy | 41.13% | 34.86% | **49.81%** | **Paper best model** |
| Pouring | Exact sequence match / edit distance 0 | 5.67% | 0.00% | **30.13%** | **Paper best model** |
| Pouring | Edit distance <= 1 | 5.77% | 0.00% | **36.38%** | **Paper best model** |
| Pouring | Edit distance <= 2 | 5.96% | 0.00% | **39.96%** | **Paper best model** |
| Pouring | Edit distance <= 3 | 7.09% | 0.00% | **42.90%** | **Paper best model** |
| Pouring | Edit distance <= 4 | 21.92% | 0.01% | **51.74%** | **Paper best model** |