# Methodology — DR Estimation for Item-Level Benchmark Missingness

*This document describes the methods and design choices for the project. Written to feed directly into the final report.*

---

## 1. Problem Setting

Large-scale AI evaluation relies on benchmarks, but the matrix of (model × benchmark-item) scores is highly incomplete. Not every model is tested on every item of every benchmark. This missingness is plausibly **non-random**: stronger models tend to be evaluated more thoroughly, certain model families only appear on specific benchmarks, and newer benchmarks only have results for models that existed at their release.

If missingness is correlated with performance — a Missing Not At Random (MNAR) pattern — then naive aggregation of observed scores produces biased estimates of model capability.

We apply **doubly robust (DR) estimation** at the item level to correct for this bias.

### Research Questions
1. Can we predict which (model, item) entries are observed using model and item features?
2. Does correcting for non-random missingness via DR estimation meaningfully change benchmark aggregates and model rankings?
3. How sensitive are DR-corrected estimates to violations of the MAR assumption?

---

## 2. Data: measurement-db

We use the **measurement-db** dataset, which aggregates results from 16 diverse AI benchmarks into a unified format.

### Scale
- **909 models** (subject_id) spanning open-source, closed-source, and agent-based systems
- **103,983 items** (item_id) — individual questions/tasks across all benchmarks
- **16 benchmarks** covering: medical QA (afrimedqa), vision understanding (ai2d_test, mmbench_v11, mathvista_mini), coding (swebench, livecodebench), function calling (bfcl), math competition (matharena), safety (rewardbench), instruction following (mtbench, ultrafeedback), and more

### Missingness Landscape
Observation density varies dramatically across benchmarks:
- **6 benchmarks are fully observed** (100% density): afrimedqa, androidworld, cybench, mathvista_mini, rewardbench, swebench
- **5 benchmarks are near-complete** (85-100%): agentdojo, ai2d_test, bfcl, livecodebench, mmlupro
- **3 benchmarks have substantial missingness** (<40%): hle (39.2%), matharena (38.1%), ultrafeedback (23.5%)

### Benchmark-Level vs. Item-Level Missingness

We distinguish between two types of missingness in the evaluation matrix:

1. **Benchmark-level missingness**: A model was never evaluated on a given benchmark at all. This produces block-matrix patterns in the observation matrix — entire rows are missing for all items simultaneously. Benchmarks like hle (19 models, 39.2%), livecodebench, bfcl, and agentdojo exhibit this pattern. Benchmark-level selection (which models get run on which benchmarks) is a separate research question driven by factors like benchmark release date, compute cost, and model modality.

2. **Item-level missingness**: A model was evaluated on a benchmark, but some individual items are missing. This produces scattered or structured patterns *within* the observation matrix of models that did participate. This is the focus of our work — it reflects selective reporting or partial evaluation at the question level, and is where doubly robust correction at the item level can provide meaningful bias reduction.

After visualizing observation matrix heatmaps, we restrict our analysis to **5 benchmarks with genuine item-level missingness**:

| Benchmark | Density | Models × Items | Missingness Type |
|-----------|---------|---------------|------------------|
| **mmbench_v11** | 96.8% | 251 × 3,579 | Item-level (primary focus) |
| ai2d_test | 98.3% | 254 × 3,088 | Item-level |
| mmlupro | 85.7% | 48 × 13,542 | Item-level |
| matharena | 38.1% | 88 × 405 | Item-level |
| ultrafeedback | 23.5% | 17 × 63,932 | Item-level |

The near-complete benchmarks (mmbench_v11, ai2d_test) enable semi-synthetic experiments: we treat them as approximate ground truth, artificially remove entries following MNAR patterns, and evaluate whether DR estimation recovers the true aggregate.

### Evidence of MNAR
Spearman correlations between model performance and observation coverage suggest MNAR:
- **matharena**: ρ = 0.54 — stronger models are tested on more items
- **mmlupro**: ρ = −0.50 (inverse — weaker models evaluated more, likely open-model leaderboard effect)

---

## 3. Data Preprocessing (Step 0)

### 3a. Item Embeddings

We embed all 103,983 item content strings using **nomic-ai/nomic-embed-text-v1.5**:
- 768-dimensional embeddings
- 8,192 token context window (accommodates long coding problems and GitHub issues without truncation)
- Strong MTEB performance, Apache 2.0 license

Items are prefixed with `"search_query: "` per the model's specification. All embeddings are L2-normalized to unit length for cosine similarity computation. The 1,792 HLE items contain only opaque hash IDs (no question text) and produce low-information embeddings.

**Technical note:** Encoding was done on a Google Colab T4 GPU (15GB VRAM). To handle the wide content length distribution (median 307 chars, max 24,770 chars), we load the model in float16 precision and sort texts by length before batching. This prevents a single long item from forcing an entire batch to be padded to its length, which would cause out-of-memory errors.

### 3b. Model Features

For each of the 909 models, we extract features from four sources:

**Name parsing (100% coverage):**
- Provider family (17 categories: openai, anthropic, google, meta, mistral, deepseek, qwen, etc.)
- Architecture family (14 categories: gpt, claude, gemini, llama, mistral, qwen, etc.)
- Parameter count in billions (46% coverage — extracted via regex, including MoE patterns like `8x7B → 56B`)
- Binary flags: is_instruct, is_multimodal, is_closed, is_swebench_agent, is_reward_model, is_fc_variant, is_reasoning

**LMArena Elo scores (31% coverage):** Matched via normalized name fuzzy matching.

**HuggingFace metadata (18% coverage):** Downloads, likes, creation date for models with identifiable HF repos.

**OpenRouter catalog (21% coverage):** Context window length and pricing for models available through OpenRouter.

The boolean features from name parsing have 100% coverage and are the most predictive for propensity modeling. External features provide additional signal where available; tree-based models (GBM) handle missing values natively.

### 3c. Observation Matrices

For each benchmark, we construct:
- **Response matrix M** ∈ ℝ^{n_models × n_items}: mean response across trials, normalized to [0,1]
  - mtbench: (x-1)/9 (likert 1-10)
  - ultrafeedback: (x-1)/4 (likert 1-5)
  - All others: already in [0,1] (binary or fraction)
- **Observation matrix O** ∈ {0,1}^{n_models × n_items}: O[i,j] = 1 iff M[i,j] is observed

**Test condition handling:** Many benchmarks have multiple experimental conditions (e.g., different attack types in agentdojo, different evaluation aspects in ultrafeedback). We select the primary condition per benchmark to avoid double-counting items:
- afrimedqa: `source=afrimedqa-v2|prompt=base`
- agentdojo: `metric=utility|attack=important_instructions`
- livecodebench: `source=submissions`
- matharena: null test_condition rows (per-attempt binary outcomes)
- ultrafeedback: `aspect=helpfulness`

**Item-level statistics computed:**
- Item difficulty: mean response across models (for binary items, this is fraction correct)
- Item discrimination: Pearson correlation between item response and model total score
- Number of models observed per item

**Model-level statistics computed:**
- Model mean score across observed items
- Number of items observed, observation rate

---

## 4. Missingness Prediction (Step 1) — COMPLETE

For each of the 5 focus benchmarks, we model the propensity π̂(O[i,j] = 1 | X_model_i, X_item_j) using logistic regression.

**Feature construction:**
- **Model features** (~25-35 dim): 7 boolean flags (is_instruct, is_multimodal, is_closed, is_swebench_agent, is_reward_model, is_fc_variant, is_reasoning), 4 numeric features (log_param_count, arena_elo, hf_downloads, openrouter_context_length) with median imputation, and one-hot encoded provider and arch_family (drop_first=True).
- **Item features** (23 dim): item_difficulty, item_discrimination, content_length from item_stats.csv, plus top-20 PCA components of the 768-dim nomic embeddings (PCA fit per benchmark).
- **Pair features**: For each (model, item) pair, the feature vector is the concatenation of the model and item feature vectors.

**Training:** Logistic regression (C=1.0, solver='lbfgs', max_iter=500) with StandardScaler preprocessing. 5-fold stratified cross-validation (random_state=42). For benchmarks with >500K pairs (mmlupro, ultrafeedback), pair matrices are constructed in chunks of 200K to manage memory.

**Ablations:** Model-only and item-only feature sets reveal whether missingness is driven by which model is evaluated, which item is asked, or their interaction.

**Results:**

| Benchmark | AUC | Model-only | Item-only | Dominant mechanism |
|-----------|-----|------------|-----------|-------------------|
| ai2d_test | 0.874 | 0.869 | 0.539 | Model-driven (architecture, multimodality) |
| mmbench_v11 | 0.831 | 0.813 | 0.610 | Model-driven + some item signal |
| mmlupro | 0.757 | 0.579 | 0.702 | Item-driven (question content, context length) |
| matharena | 0.720 | 0.656 | 0.642 | Mixed — both model strength and item type |
| ultrafeedback | 0.550 | 0.551 | 0.494 | ~MCAR — no predictable pattern |

Missingness is clearly non-random in 4/5 benchmarks. The ablation reveals distinct mechanisms: vision benchmarks (ai2d, mmbench) have model-architecture-driven missingness (multimodal models are more thoroughly evaluated), while mmlupro has item-driven missingness (specific question types are selectively missing). Ultrafeedback serves as a null case where missingness is essentially unpredictable.

---

## 5. Semi-Synthetic DR Experiments (Step 2) — COMPLETE

We test whether doubly robust estimation can recover unbiased model scores under non-random missingness, using a semi-synthetic approach: learn the masking pattern from real data, add additional missingness following that pattern, then evaluate recovery against ground truth.

**Dropped ultrafeedback:** 68% of its scores are non-binary (likert 1–5), producing poor IRT fits. The remaining 4 benchmarks (mmbench_v11, ai2d_test, mmlupro, matharena) all have binary items. Paper figures focus on **mmlupro** and **matharena**.

### 5a. Learning the Masking Pattern from Real Data

For each benchmark, we learn how the real observation matrix O_orig was generated. We fit a logistic regression on all (model, item) pairs within O_orig:

P(O_orig[i,j] = 1) = σ(w^T · x_{ij})

where x_{ij} is the concatenation of model features and item features (same features as Step 1). This captures the real-world relationship between model/item characteristics and observation probability.

**Two masking modes:**

1. **Features only**: x_{ij} = [model_features, item_features]. Missingness depends on observable metadata — which model architecture, what kind of question, etc. This covers scenarios like: multimodal models are evaluated more on vision benchmarks, long questions get skipped, etc.

2. **Features and score**: x_{ij} = [model_features, item_features, M[i,j]]. Missingness also depends on the actual score. Since M[i,j] is unobserved where O_orig = 0, we first fit IRT on O_orig to impute these values, then include the (actual or imputed) score as an additional feature. This captures scenarios where easy items for a model are more likely to be reported.

### 5b. Generating New Observation Matrices

Using the learned masking model, we generate O_new ⊂ O_orig at various sparsity levels. For each target keep_rate ∈ {0.5, 0.6, 0.7, 0.8, 0.9}:

1. Compute z_ij = logit(P̂(O_orig[i,j] = 1)) for all originally-observed pairs
2. Center z to z_centered = z − mean(z)
3. Binary search for α such that σ(α + z_centered) achieves the target keep_rate
4. Draw O_new[i,j] ~ Bernoulli(σ(α + z_centered[i,j])) for each O_orig[i,j] = 1

This ensures O_new preserves the *structure* of missingness from O_orig (same features drive observation probability) while controlling the overall observation rate.

**Ground truth**: M_true[i,j] for all originally-observed entries allows bias-free evaluation.

### 5c. Estimators

On each O_new, we fit and evaluate four estimators of model-level mean scores:

**1. Naive**: Mean of observed entries per model.
  V̂_naive(m) = (1/n_obs(m)) Σ_{j: O_new[m,j]=1} M[m,j]

**2. IRT (Rasch model)**: Fit a 1PL model on O_new:
  P(Y=1 | θ_m, β_k) = σ(θ_m − β_k)

  Parameters θ_m (model ability) and β_k (item difficulty) are estimated via gradient descent (200 iterations, lr=0.1, L2 regularization λ=0.01). Model score estimate:
  V̂_IRT(m) = (1/K) Σ_k σ(θ̂_m − β̂_k)

**3. IPW (Inverse Propensity Weighting)**: Fit a propensity model on O_new using features only (never the score M):
  π̂(i,j) = P̂(O_new[i,j] = 1 | model_features, item_features)

  using logistic regression (C=1.0, max_iter=500, solver='lbfgs'), clipped to [0.05, 0.95]. Model score estimate:
  V̂_IPW(m) = Σ_{j: O_new[m,j]=1} M[m,j]/π̂(m,j) / Σ_{j: O_new[m,j]=1} 1/π̂(m,j)

**4. DR+IRT (Doubly Robust)**: Combines IRT outcome model R̂ with propensity correction:
  V̂_DR(m) = (1/K) Σ_k [R̂(m,k) + O_new[m,k]/π̂(m,k) · (M[m,k] − R̂(m,k))]

  This has the double robustness property: the estimate is consistent if *either* the IRT model or the propensity model is correctly specified.

### 5d. Evaluation Metrics

For each estimator, we compute per-model score estimates V̂(m) and compare against ground truth V_true(m) = mean of all originally-observed items:

- **Bias**: mean(V̂ − V_true) — systematic over/underestimation
- **RMSE**: √mean((V̂ − V_true)²) — overall accuracy
- **Rank correlation**: Spearman ρ between V̂ and V_true — preservation of model ordering

### 5e. Key Results

**Features_only mode** (propensity correctly specified — same features drive masking and correction):

| Benchmark | Naive bias | IRT bias | IPW bias | DR bias |
|-----------|-----------|----------|----------|---------|
| mmlupro (keep=0.7) | +0.015 | +0.007 | −0.005 | **+0.001** |
| matharena (keep=0.7) | −0.009 | −0.001 | +0.001 | **−0.001** |

DR corrects to near-zero bias across all keep rates. This is expected: the propensity model can capture the masking mechanism because it uses the same feature space.

**Features_and_score mode** (propensity misspecified — can't observe score M):

| Benchmark | Naive bias | IRT bias | IPW bias | DR bias |
|-----------|-----------|----------|----------|---------|
| mmlupro (keep=0.7) | +0.008 | **−0.004** | −0.013 | −0.006 |
| matharena (keep=0.7) | +0.035 | **+0.017** | +0.030 | +0.019 |

When masking depends on the score itself, the propensity model (which never sees M) is misspecified. IPW and DR cannot fully correct the bias. IRT is actually the best corrector here because it directly models the score structure via latent ability parameters.

### 5f. Interpretation

These results demonstrate three key findings:

1. **When missingness depends on observable features, DR works.** The features_only case shows that propensity-based correction can eliminate bias when the propensity model is correctly specified — the classic MAR (Missing At Random conditional on observed features) result.

2. **When missingness also depends on the outcome, IRT helps more.** The features_and_score case reveals the fundamental MNAR limitation: propensity methods fail when the missing data mechanism depends on unobserved quantities. IRT succeeds because it models the data-generating process (ability × difficulty) rather than the selection mechanism.

3. **Real-world missingness is likely a mix.** In practice, some missingness is driven by observable features (model architecture, item type) and some by scores (easy items more likely reported). DR + IRT together provide the most robust correction across both regimes.

---

## 6. Future Extensions

### Held-Out Model Prediction

A natural real-world application: hold out an entire model (row) from the matrix, fit IRT+DR on the remaining models, then predict the held-out model's full-benchmark score from its partially observed items. This tests whether the DR+IRT framework improves over naive averaging for a genuinely new model entering the benchmark.

### Sensitivity Analysis

Apply Tan's marginal sensitivity model to bound DR estimates under MNAR violations. For a sensitivity parameter Γ ≥ 1, compute upper/lower bounds on model scores and find the tipping point Γ* at which conclusions change.
