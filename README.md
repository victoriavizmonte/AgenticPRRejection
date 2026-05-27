# Predicting Agentic Pull Request Rejection: An Empirical Comparison of NLP Approaches

**Author:** Victoria Angela Vizmonte  

This repository contains the full code for the thesis. Three NLP approaches are compared for predicting whether an AI-agent-authored pull request will be rejected at submission time, using the AIDev dataset (10,648 PRs from December 2024 to July 2025).

---

## Final Notebooks

| Name | Notebook | Description | Remarks |
|------|----------|-------------|---------|
| Exploratory Data Analysis | `notebooks/final/final_eda.ipynb` | Analyzes the final study population across class distribution, temporal trends, agent subgroups, task types, programming languages, and metadata features. | CPU only |
| Multimodal Logistic Regression Classifier | `notebooks/final/multimodal_lr.ipynb` | Full pipeline: preprocessing, CodeBERT diff embedding extraction, sBERT text encoding, metadata encoding, logistic regression training, evaluation, and ablation. | T4 GPU recommended |
| Few-Shot Standard LLM Classifier (Qwen3-32B) | `notebooks/final/qwen3_standard_llm.ipynb` | Few-shot PR rejection classification using Qwen3-32B in non-thinking mode via DSPy and the OpenRouter API. Includes full val and test inference with cache-guarded API calls. | Requires OpenRouter API key; saved predictions included in repo |
| Few-Shot Reasoning LLM Classifier (QwQ-32B) | `notebooks/final/qwq_reasoning_llm.ipynb` | Few-shot PR rejection classification using QwQ-32B with native chain-of-thought reasoning traces. Identical DSPy signature and demos as the standard LLM classifier for controlled comparison. | Not fully reproducible — model removed from OpenRouter as of May 2026; saved outputs included in repo |
| Statistical Tests | `notebooks/final/results_statistical_tests.ipynb` | Cross-model statistical comparison using bootstrap CI, pairwise bootstrap PR-AUC tests, McNemar's test, and Holm-Bonferroni correction across all three classifiers. | CPU only; runs directly from saved prediction CSVs |

---

## Prerequisites

- A **Google account** with access to Google Drive and Google Colab
- An **OpenRouter API key** for the few-shot standard LLM classifier only (sign up at [openrouter.ai](https://openrouter.ai))
- No local Python installation required — all execution happens in Google Colab

---
## Getting Started

Clone this repository first:
```bash
git clone https://github.com/victoriavizmonte/AgenticPRRejection.git
```
Then follow Steps 1–4 below to set up the data and run the notebooks.

## Step 1 — Get the Data

The AIDev data files are too large for GitHub and are provided separately via a shared Google Drive folder. Prediction CSVs, metrics JSONs, and charts are included directly in this repository under `results/final/`.

**Data folder:** https://drive.google.com/drive/folders/1EopKj73bVw4_12owBMVMFHls6Cw4fOuQ?usp=sharing

Follow these steps to set up the data:

1. Open the shared folder link above
2. Click **Add shortcut to Drive** (or right-click the folder and select the same option)
3. Place the shortcut anywhere on your Google Drive — you will tell each notebook where it is in Step 2

The Data folder contains:

| Path | Size | Description |
|------|------|-------------|
| `data/processed/pr_closed.parquet` | 10.8 MB | Filtered 10,648-PR study population from AIDev with all metadata features. Produced by running `multimodal_lr.ipynb` through the preprocessing cell. |
| `data/raw/pr_commit_details.parquet` | 462 MB | File-level code diffs from AIDev. Required by all model notebooks for diff text extraction. |
| `data/raw/pull_request.parquet` | 16.1 MB | Primary PR metadata table. |
| `data/raw/pr_commits.parquet` | 7.9 MB | Commit-level metadata. |
| `data/raw/pr_task_type.parquet` | 2.9 MB | Auto-assigned task type labels. |
| `data/raw/related_issue.parquet` | 74 KB | PR to GitHub issue mappings. |
| `data/raw/repository.parquet` | 163 KB | Repository-level attributes. |

> **Note:** All raw AIDev tables are included in the Data folder above, 
> so running `notebooks/utils/download_aidev_data_colab.ipynb` is not 
> required. It is provided only for reference or if you prefer to 
> download directly from HuggingFace at the pinned commit 
> `512e07014b7b6e34cc1080372caa1c2bc054369d`.

---

## Step 2 — Configure Each Notebook

Every notebook has a `DRIVE_BASE` variable at the first cell. Set this to the path on your own Google Drive where you placed the Data folder shortcut.

For example, if you placed the folder at the root of your Drive:
```python
DRIVE_BASE = Path('/content/drive/MyDrive/AgenticPRRejection')
```

If you placed it inside a subfolder:
```python
DRIVE_BASE = Path('/content/drive/MyDrive/MyFolder/AgenticPRRejection')
```

The first cell will successfully print `Drive mounted` if the path is correct. Otherwise, update `DRIVE_BASE` and re-run the cell before proceeding.

---

## Step 3 — Run the Notebooks

All notebooks run in **Google Colab**. Open each one via `File > Open notebook > Google Drive` or by uploading from this repository.

**Always run the first cell** at the start of every Colab session. It mounts Drive, adds `src/` to the Python path, and installs dependencies from `requirements.txt`.

> **Running from scratch:** By default, each notebook skips steps where cached outputs already exist. To force a full re-run, delete the relevant cached files before running:
> - Embeddings: `results/final/multimodal_lr/embeddings/`
> - Trained model: `results/final/multimodal_lr/*.joblib`, `results/final/multimodal_lr/*.json`, and `results/final/multimodal_lr/*_predictions.csv`
> - LLM predictions: `results/final/qwen3_standard_llm/`
> - Statistical test outputs: `results/final/statistical_tests/`
>
> Note: the few-shot reasoning LLM classifier cannot be directly re-run regardless, so do NOT delete its saved outputs.

### Recommended run order

#### 1. Exploratory Data Analysis (optional)
**Notebook:** `notebooks/final/final_eda.ipynb`  
**Hardware:** CPU  
**Runtime:** ~5 min

No setup beyond the first cell. Produces exploratory charts saved to `results/final/eda/`.

---

#### 2. Multimodal Logistic Regression Classifier
**Notebook:** `notebooks/final/multimodal_lr.ipynb`  
**Hardware:** T4 GPU recommended (`Runtime > Change runtime type > T4 GPU`)  
**Runtime:** ~25 min on first run; ~5 min on re-run if embeddings are cached

Run cells in order. The CodeBERT embedding extraction step dominates runtime and writes `.npy` files to `results/final/multimodal_lr/embeddings/`. If these files already exist from the shared Drive, the GPU step is skipped automatically.

Outputs saved to `results/final/multimodal_lr/`:
- `logistic_regression_multimodal.joblib` — trained classifier
- `feature_pipeline.joblib` — fitted feature pipeline
- `multimodal_lr_metrics.json` — train/val/test metrics
- Prediction CSVs and evaluation charts

---

#### 3. Few-Shot Standard LLM Classifier (Qwen3-32B)
**Notebook:** `notebooks/final/qwen3_standard_llm.ipynb`  
**Hardware:** CPU (API-based inference)  
**Runtime:** ~10 hours for full inference; skipped if prediction CSVs already exist

**API key setup — do this before running the first cell:**
1. In Colab, click the key icon in the left sidebar (**Secrets**)
2. Add a secret named `OPENROUTER_API_KEY` with your OpenRouter key as the value
3. Toggle on **Notebook access** for that secret

The notebook reads the key via `userdata.get('OPENROUTER_API_KEY')` and will raise an error if it is missing. If prediction CSVs already exist in `results/final/qwen3_standard_llm/`, all API calls are skipped automatically.

Outputs saved to `results/final/qwen3_standard_llm/`.

---

#### 4. Few-Shot Reasoning LLM Classifier (QwQ-32B)
**Notebook:** `notebooks/final/qwq_reasoning_llm.ipynb`  
**Status: NOT directly re-runnable.** As of May 2026, QwQ-32B (`qwen/qwq-32b`) has been removed from OpenRouter.

The notebook can be opened to review the implementation and reasoning trace examples. All saved outputs (predictions, metrics, charts) are included in this repository under `results/final/qwq_reasoning_llm/` and are used directly by the statistical tests notebook.

---

#### 5. Statistical Tests
**Notebook:** `notebooks/final/results_statistical_tests.ipynb`  
**Hardware:** CPU  
**Runtime:** ~5 min

Reads saved prediction CSVs from all models. Pre-saved outputs from all models are included in this repository, so this notebook can be run immediately without running any model notebook first.

Outputs saved to `results/final/statistical_tests/`.

---

#### 6. Repo-Context Feature Ablation — supplementary (optional)
**Notebook:** `notebooks/exploratory/exploratory_multimodal_lr_repo_context.ipynb`  
**Hardware:** T4 GPU recommended  
**Runtime:** ~10 min

Tests whether adding repository-level context features (agent and repository rejection rates) improves over the baseline multimodal logistic regression classifier. Requires the cached CodeBERT embeddings from the multimodal logistic regression classifier run to be present in `results/final/multimodal_lr/embeddings/` on Drive.

Outputs saved to `results/exploratory/exploratory_multimodal_lr_repo_context/`.

---

## Step 4 — Get the Model Artifacts (optional)

This step is only needed if you want to re-run the multimodal logistic regression classifier notebook without retraining from scratch.

**Results folder:** https://drive.google.com/drive/folders/1CekyfXuAe5kDYLEkibYpfFGln5WjnodJ?usp=sharing

1. Open the Results folder link above
2. Click **Add shortcut to Drive** and place it under the **same parent folder** as the Data folder from Step 1

The Results folder contains:

| Path | Size | Description |
|------|------|-------------|
| `results/final/multimodal_lr/feature_pipeline.joblib` | ~88 MB | Fitted feature pipeline used by the multimodal logistic regression classifier. |
| `results/final/multimodal_lr/logistic_regression_multimodal.joblib` | ~88 MB | Trained multimodal logistic regression classifier. |

All other model outputs from the above folder (prediction CSVs, metrics JSONs, and charts) are also included directly in this repository under `results/final/` and do not require any Drive setup.

---

## Repository Structure

```
AgenticPRRejection/
├── notebooks/
│   ├── final/
│   │   ├── final_eda.ipynb
│   │   ├── multimodal_lr.ipynb
│   │   ├── qwen3_standard_llm.ipynb
│   │   ├── qwq_reasoning_llm.ipynb
│   │   └── results_statistical_tests.ipynb
│   ├── exploratory/
│   │   └── exploratory_multimodal_lr_repo_context.ipynb    # Supplementary: repo-context feature ablation
│   └── utils/
│       └── download_aidev_data_colab.ipynb                 # Optional: full reproduction from raw AIDev tables
├── src/
│   ├── config.py                            # RANDOM_SEED, paths, constants
│   ├── data_loader.py                       # Environment-aware parquet loader
│   ├── preprocessing.py                     # Label creation, filtering, feature merging
│   ├── splits.py                            # Temporal 70/15/15 split
│   ├── model1_embedder.py                   # CodeBERT GPU embedding extraction
│   ├── model1_trainer.py                    # Logistic regression training and tuning
│   ├── evaluator.py                         # Metrics, plots, subgroup reports for all models
│   ├── statistical_tests.py                 # Bootstrap CI, McNemar, Holm-Bonferroni
│   ├── repo_context_features.py             
│   └── features/
│       ├── feature_pipeline.py              
│       ├── text_features.py                 
│       ├── metadata_features.py            
│       └── diff_features.py                 
├── data/
│   ├── processed/
│   │   └── pr_closed.parquet                
│   └── raw/
│       └── pr_commit_details.parquet        
├── results/
│   ├── final/                               
│   │   ├── multimodal_lr/                   # Multimodal logistic regression classifier outputs (and .joblib artifacts via Results Drive folder - see Step 4)
│   │   ├── qwen3_standard_llm/              # Few-shot standard LLM classifier outputs
│   │   ├── qwq_reasoning_llm/               # Few-shot reasoning LLM classifier outputs (pre-saved only)
│   │   ├── statistical_tests/               # Cross-model comparison results
│   │   └── eda/                             # Chart and CSV outputs from EDA
│   └── exploratory/                                     
│       ├── exploratory_multimodal_lr_repo_context/      # Repo-context ablation outputs
│       ├── reasoning_llm_tuning/                        # Reasoning LLM 50-PR tuning prediction CSVs
│       └── standard_llm_tuning/                         # Standard LLM 50-PR tuning prediction CSVs
├── requirements.txt
└── README.md
```

---

## Reproducibility Notes

- All stochastic operations use `RANDOM_SEED = 42` (defined in `src/config.py`).
- The temporal split (70% train / 15% val / 15% test by PR creation date) is deterministic given the same dataset.
- The study population is derived from `hao-li/AIDev` on HuggingFace, accessed on February 21, 2026, at commit `512e07014b7b6e34cc1080372caa1c2bc054369d` ([link](https://huggingface.co/datasets/hao-li/AIDev/commit/512e07014b7b6e34cc1080372caa1c2bc054369d)). The dataset covers December 2024 to July 2025 and has been updated since this access date. Use the pinned commit to reproduce the exact study population.
- Multimodal logistic regression classifier CodeBERT embeddings are cached as `.npy` files. Re-running the embedding extraction step from scratch requires a T4 GPU session (~45 min).
- Few-shot standard LLM classifier inference is cache-guarded. Re-running from scratch requires an OpenRouter API key and will consume approximately 11,000 API calls (with US$ cost) per split.
- Few-shot reasoning LLM classifier (QwQ-32B) was run via the SiliconFlow provider on OpenRouter. The model has since been removed from OpenRouter and cannot be re-run. Saved predictions and outputs in this repository are the only available source.
- The 50-PR hyperparameter tuning samples for the LLM classifiers used `val_df.sample(n=50, random_state=42)` and are excluded from full validation inference via the `already_predicted` merge pattern in each notebook.
- `requirements.txt` pins exact package versions for all dependencies.

---

## Citation

If you use this replication package or the AIDev dataset, please cite the original AIDev dataset paper:

```
@misc{li2025aidev,
  title={The Rise of AI Teammates in Software Engineering (SE) 3.0: How Autonomous Coding Agents are Reshaping Software Engineering},
  author={Hao Li and Hongjun Zhang and Ahmed E. Hassan},
  year={2025},
  eprint={2507.15003},
  archivePrefix={arXiv},
  primaryClass={cs.SE}
}
```
