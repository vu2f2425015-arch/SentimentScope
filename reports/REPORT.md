# SentimentScope — Technical Project Report & Evaluation

**SentimentScope** is a 3-class sentiment analysis system evaluated on social microblog data from the CardiffNLP TweetEval benchmark. This report documents the end-to-end engineering process: data ingestion, cleaning, feature engineering, model training across five architectures (statistical, linear, recurrent, and transformer), calibration, and production deployment across constrained compute tiers.

---

## 1. System Architecture & Code Organization

I organized the codebase into `src/` to separate data downloading, preprocessing, feature extraction, training, and API serving into distinct modules instead of keeping everything in one large script or notebook. 

- `src/data_loader.py`: Downloads the CardiffNLP TweetEval benchmark, handles deduplication, and creates stratified 70/15/15 train, validation, and test splits.
- `src/preprocessing.py`: Text normalization that preserves negation tokens (`not`, `never`, `n't`, contractions) and cleans HTML entities, handles, and URLs.
- `src/features.py`: Feature pipelines including subword character n-grams, word n-grams, and sparse CSR representations for sklearn estimators.
- `src/train.py`: Model fitting, cross-validation, probability calibration, threshold tuning, and model serialization.
- `src/visualize.py`: Evaluation plotting scripts for confusion matrices, calibration curves, and latency comparisons.
- `src/api.py`: FastAPI application serving predictions, batch processing, and live database telemetry.
- `src/db.py`: SQLite persistence using Write-Ahead Logging (WAL) mode with background ingestion and auto-pruning.
- `public/index.html`: Web interface displaying model telemetry, batch CSV predictions, and real-time feeds.

Adding type hints and docstrings across functions helped catch subtle bugs during development — for instance, distinguishing between 2D CSR matrices and 1D dense arrays when piping feature unions into downstream estimators. Unit tests in `tests/` check dataset integrity, contraction expansion, and API response formatting, while environment variables handle configuration across local machines and cloud deployments.

### Memory optimization during feature extraction

Early versions of `train.py` called `.toarray()` on the TF-IDF output before passing it into sklearn's grid search. On the 41,748-row training split, that turned a sparse matrix into a dense one and spiked RAM past 6.7GB per grid-search step — the process would get killed on anything short of a workstation. Removing `.toarray()` and feeding the CSR sparse matrix directly into the solver fixed it: memory dropped to around 50MB and feature extraction ran roughly 50x faster. Most sklearn estimators (LogisticRegression, LinearSVC, MultinomialNB) accept sparse input natively, so the dense conversion was never necessary — it was a holdover from an early debugging pass where I wanted to inspect the matrix directly.

---

## 2. Dataset & Exploratory Analysis

The benchmark dataset is the **CardiffNLP TweetEval 3-Class Sentiment Benchmark** (Barbieri et al., 2020), containing 59,641 microblog tweets labeled as negative (0), neutral (1), or positive (2).

### Stratified Partitioning

To avoid data leakage, the corpus was divided into a fixed 70/15/15 stratified partition:
- **Training Set (70%)**: 41,748 tweets
- **Validation Set (15%)**: 8,946 tweets
- **Held-Out Test Set (15%)**: 8,947 tweets

```
Class Distribution Across Splits:
├── Neutral:  45.8% (dominant baseline class)
├── Positive: 35.2% (second largest class)
└── Negative: 19.0% (minority class)
```

![Class Distribution](reports/figures/04_class_distribution.png)

The class distribution presents an immediate challenge: neutral tweets dominate the dataset, while negative tweets represent less than a fifth of all samples. Without deliberate class weighting or threshold adjustments, standard classifiers naturally gravitate toward predicting neutral on ambiguous phrasing, which hurts negative recall. In practical social listening or customer support monitoring, missing customer dissatisfaction is usually the most costly error.

---

## 3. Preprocessing & Feature Engineering

Microblog text contains noise that breaks standard NLP assumptions — unescaped HTML entities (`&amp;`), hashtags, user handles (`@handle`), shorthand contractions (`can't`, `won't`), and external links.

The preprocessing pipeline in `src/preprocessing.py` applies the following sequence:
1. Unescapes HTML entities and normalizes whitespace.
2. Expands contractions before punctuation is stripped (`don't` $\rightarrow$ `do not`, `can't` $\rightarrow$ `cannot`, `won't` $\rightarrow$ `will not`).
3. Strips URLs and user mentions while retaining alphanumeric tokens and hashtags.
4. Normalizes text to lowercase and filters stopwords, with an explicit exclusion list for negations (`not`, `no`, `never`, `n't`, `neither`, `cannot`, `without`).
5. Applies WordNet lemmatization.

### Feature Extraction Pathways

- **Classical Baseline**: TF-IDF unigrams with sublinear term-frequency scaling.
- **Modernized Hybrid Linear Tier**: A `FeatureUnion` combining word n-grams (1, 2) with character boundary subwords (3, 5). This captures subword morphology so that unseen words or brand names (e.g., `"sentimentscope"`) still retain signal from recognizable fragments (`"sent"`, `"enti"`, `"scope"`).
- **Recurrent Baseline**: Keras integer sequence tokenizer (`vocab_size=10,000`, `max_len=150`) with post-sequence padding.
- **Transformer Tokenizers**: Native WordPiece tokenization for `distilbert-base-uncased` and Byte-Pair Encoding (BPE) for `cardiffnlp/twitter-roberta-base-sentiment-latest` (`max_length=64`).

---

## 4. Model Evaluation & Benchmark Results

All five candidate architectures were evaluated on the same 8,947-sample held-out test split. The summary below reports test accuracy, macro precision, macro recall, macro F1, minority negative recall, single-sample CPU latency, and active deployment status.

### Full Benchmark Performance Matrix

| Model Architecture | Model Family | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Negative Recall | CPU Latency (p50) | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Multinomial Naive Bayes** | Statistical | 59.73% | 0.6085 | 0.5383 | 0.5504 | 29.21% | 0.15 ms | Baseline |
| **Vanilla Logistic Regression** | Linear (Unigrams) | 61.64% | 0.6131 | 0.5726 | 0.5850 | 39.72% | 0.42 ms | Baseline |
| **Bi-LSTM** | Recurrent Neural Net | 60.53% | 0.5912 | 0.6055 | 0.5954 | 60.28% | 3.12 ms | Deep Learning |
| **Hybrid TF-IDF + Logistic Regression** | Linear (Word + Char Subwords) | **65.92%** | **0.6454** | **0.6657** | **0.6528** | **67.63%** *(74.21% tuned)* | **0.55 ms** | **Active Cloud Edge Tier (512MB RAM)** |
| **DistilBERT Base** | Distilled Transformer | 72.57% | 0.7159 | 0.7305 | 0.7221 | 74.22% | 14.01 ms | Benchmarked |
| **Twitter-RoBERTa Base** | Domain-Adapted Transformer | **76.22%** | **0.7531** | **0.7727** | **0.7610** | **80.79%** | **22.29 ms** | **Active Dedicated Tier ($\ge$1GB RAM)** |

![Accuracy vs F1 Comparison](reports/figures/02_model_accuracy_f1_comparison.png)

### Baseline Performance Analysis

*Naive Bayes.* Multinomial Naive Bayes trained quickly and had minimal compute requirements, but its strong conditional independence assumption caused it to struggle on social text. In particular, it frequently failed on negated phrasing, producing a negative recall of just 29.21% — meaning it missed over 70% of negative reviews in the test split.

*Bi-LSTM.* Training a Bidirectional LSTM (`Embedding(128)` $\rightarrow$ `SpatialDropout1D(0.2)` $\rightarrow$ `Bi-LSTM(64)` $\rightarrow$ `Dense(64)` $\rightarrow$ `Softmax(3)`) from scratch on 41k tweets did not deliver the performance I initially expected. Without pre-trained embeddings (like GloVe or fastText), the embedding layer had to learn microblog vocabulary from scratch. Given the informal spelling, typos, and shorthand in the dataset, the network overfit the training vocabulary while reaching only 60.53% accuracy on the test set. Because its CPU inference took ~3.12ms per sample — almost six times longer than Logistic Regression — the extra architectural complexity was not worth the compute penalty.

*Linear Models.* A standard unigram Logistic Regression model achieved 61.64% accuracy. Modernizing this with hybrid subword features (character n-grams 3–5 combined with word n-grams 1–2) and contraction expansion boosted accuracy to 65.92% and macro F1 to 0.6528, while raising negative recall from 39.72% to 67.63%. 

*Cross-Validation Stability.* Running 5-fold stratified cross-validation on the training set showed consistent performance with typical standard deviations around $\pm 0.6\%$ to $\pm 1.5\%$:
- Logistic Regression (Unigram baseline): $58.42\% \pm 1.49\%$ Macro F1
- Hybrid Logistic Regression: $63.37\% \pm 0.60\%$ Macro F1
- Bi-LSTM: $55.92\% \pm 0.92\%$ Macro F1
- Multinomial Naive Bayes: $54.33\% \pm 1.05\%$ Macro F1

---

## 5. Visual Diagnostics & Analysis

### Confusion Matrices

![Confusion Matrices](reports/figures/01_confusion_matrices.png)

Examining the error matrices across architectures shows a consistent pattern: the majority of classification errors occur between neutral and negative text, or neutral and positive text. True negative samples are rarely misclassified as positive; instead, they bleed into the neutral class. This reflects real-world microblog ambiguity, where complaints often use dry or understated language that lacks overt negative adjectives.

### Per-Class Performance Breakdown

![Per-Class Breakdown](reports/figures/03_class_level_f1_breakdown.png)

The per-class F1 breakdown highlights that the positive and neutral classes consistently achieve higher F1 scores across all models, while the negative minority class shows the widest spread between weak baselines (MNB negative F1: 0.4722) and tuned models (Twitter-RoBERTa negative F1: 0.7410).

### Class Weighting & Minority Recall

![Negative Recall Boost](reports/figures/05_negative_class_recall_boost.png)

Applying inverse class-frequency loss weights ($w_{\text{neg}} = 1.7519, w_{\text{neu}} = 0.7276, w_{\text{pos}} = 0.9481$) during training substantially improved minority class detection:
- Vanilla Logistic Regression negative recall: 39.72%
- DistilBERT negative recall: 74.22%
- Twitter-RoBERTa negative recall: 80.79%

---

## 6. Notable Technical Decisions

### Negation handling

The default stopword list in most preprocessing pipelines strips "not," "never," and similar tokens — which flips sentiment on phrases like "not bad" or "never again" straight into false positives/negatives. I audited the stopword set and explicitly excluded negation forms (`not`, `no`, `never`, `n't`, `neither`, `cannot`) so they survive tokenization. This alone measurably improved negative-class recall, though it didn't fully close the neutral/negative confusion — that boundary is still the model's weakest spot (see confusion matrix).

### Calibration

Raw output probabilities from Logistic Regression were overconfident — a common issue with linear models on imbalanced classes. Applying Sigmoid Platt scaling brought the Brier score down from 0.4688 to 0.4581. It's a modest improvement, but it matters for the confidence-based routing logic described below, where uncalibrated probabilities would misroute borderline cases.

![Calibration Brier Scores](reports/figures/07_calibration_brier_scores.png)

### Transformer benchmarking

I fine-tuned DistilBERT and Twitter-RoBERTa on the same split as the baselines mainly to establish an upper bound — how much accuracy is actually available if compute weren't a constraint. Twitter-RoBERTa came out ahead (76.22% accuracy / 0.7610 Macro F1), which isn't surprising since it's pretrained on in-domain tweet data rather than general text. The gap between it and Logistic Regression (roughly 10 points of accuracy: 76.22% vs 65.92%) is the real cost of staying CPU-cheap and fast at inference time.

### Confidence-based routing

Rather than picking one model everywhere, the architecture is designed to gate on prediction confidence: in environments where both models can be hosted (such as local setups or dedicated servers with $\ge 1$GB RAM), low-confidence predictions under a threshold (tested at 55%) can be routed asynchronously to the larger transformer model, while keeping the fast linear model on the primary path. This design came out of noticing that most of Logistic Regression's classification errors concentrate in similarly low-confidence predictions.

On the live Render free-tier deployment, however, cross-model escalation is intentionally disabled. Loading PyTorch and transformer weights alongside the linear model would immediately trigger an out-of-memory termination (`Exit 137`) against the 512MB RAM limit. The live cloud service therefore runs strictly on the ~50MB lightweight tier, while the dual-tier routing logic is reserved for dedicated infrastructure.

---

## 7. Operational Constraints & Production Deployment

Deploying the system to Render's free cloud tier brought an immediate practical constraint: a hard 512MB RAM limit that kills any process with an out-of-memory error (`Exit 137`).

When I first attempted to load Twitter-RoBERTa on the free tier container, PyTorch runtime initialization, model weights (475MB), and tokenizer memory immediately exceeded 600MB before serving a single request.

### The INT8 Quantization Dead End

Before committing to a tiered architecture, I tested PyTorch dynamic INT8 quantization (`torch.quantization.quantize_dynamic` on linear layers) across all 8,947 held-out test samples to see if compressing the transformer weights would allow it to run within 512MB RAM:

| Metric / Dimension | FP32 Twitter-RoBERTa | Dynamic INT8 RoBERTa | Impact / Tradeoff |
|:---|:---:|:---:|:---|
| **Model Weights Disk Footprint** | 475.57 MB | 230.93 MB | -51.4% disk reduction |
| **CPU Latency (p50)** | 22.29 ms | 18.46 ms | ~20% faster |
| **Test Accuracy** | **76.22%** | 71.78% | -4.44% degradation |
| **Macro F1** | **0.7610** | 0.6803 | -0.0807 drop |
| **Negative Class Recall** | **80.79%** | **41.60%** | **-39.19% collapse** |
| **Peak Runtime RAM Footprint** | 1,385 MB | 1,930 MB | Memory allocation spike |
| **Render 512MB Compatibility** | Fails (OOM `Exit 137`) | Fails (OOM `Exit 137`) | Still infeasible |

The quantization results revealed two major blockers:
1. **Decision Boundary Breakdown**: Quantizing the weights caused the sensitive negative sentiment boundary to collapse — negative recall dropped from 80.79% down to 41.60%, rendering the model worse at detecting customer complaints than the linear baseline.
2. **Runtime Memory Allocation**: While disk weight size was cut in half, PyTorch dynamic quantization dequantizes tensor operations during inference, requiring temporary execution buffers that pushed peak RSS past 1.9GB.

This experiment proved that squeezing a 125M parameter transformer into a 512MB RAM budget wasn't viable with dynamic quantization.

### Production Tiered Architecture

This led directly to the two-tier architecture:

```
[ Incoming Request ]
         │
         ├── Cloud Free Tier (ACTIVE_MODEL_TIER="lightweight")
         │   └── Hybrid TF-IDF + Logistic Regression (~50MB RAM, 65.9% Acc, 0.55ms kernel)
         │       └── Fast response, zero OOM risk on 512MB hosts
         │
         └── Dedicated / Local Tier (ACTIVE_MODEL_TIER="auto" / "transformer")
             └── Twitter-RoBERTa Base (125M params, 76.2% Acc, 80.8% Neg Recall)
                 └── High-accuracy inference on systems with >=1GB RAM
```

1. **Tier 1 (High-Accuracy Engine)**: Twitter-RoBERTa (76.22% Acc, 0.7610 Macro F1, 80.79% Negative Recall) runs on local development machines, Docker Compose setups, or dedicated servers with $\ge 1$GB RAM.
2. **Tier 2 (Cloud Edge Tier)**: Hybrid TF-IDF + Logistic Regression (65.92% Acc, 0.6528 Macro F1, 67.63% Negative Recall, 74.21% tuned) consumes ~50MB RAM and runs reliably on Render free-tier containers without memory crashes.

---

## 8. Latency Profiling: Compute Kernels vs Network Lifecycles

In performance discussions, latency figures often conflate raw matrix computation with web request overhead. To be clear about where time is spent, I profiled the system across three levels:

![Latency vs F1 Tradeoff](reports/figures/08_cpu_latency_vs_f1_tradeoff.png)

1. **Raw Vector / Model Kernel**:
   - Hybrid TF-IDF + Logistic Regression: **0.55 ms** per sample.
   - Twitter-RoBERTa: **22.29 ms** per sample.
2. **Local FastAPI Request Lifecycle**:
   - Parsing JSON, running text preprocessing, feature transformation, inference, and Pydantic serialization adds 1–3 ms of overhead:
   - Hybrid Logistic Regression: **1.5–3.5 ms** local roundtrip.
   - Twitter-RoBERTa: **32–38 ms** local roundtrip.
3. **Render Cloud Free Tier**:
   - Running on shared, throttled virtual CPUs (`vCPU`) with container cold starts, internal server response times range from **150–250 ms**, while public internet HTTPS roundtrips take **800–1,000 ms**.
   - Both models stay well within the project's 1-second SLA target, but distinguishing between the 0.55ms computation kernel and the 200ms cloud network roundtrip avoids unrealistic production assumptions.

---

## 9. Negative Recall Decision Threshold Optimization

In standard multi-class classification, predictions are assigned via $\arg\max_k P(Y=k)$. Because neutral tweets outnumber negative tweets 2.4 to 1, linear models often assign moderate probability to negative (e.g. 0.36) and slightly higher to neutral (0.42), misclassifying true complaints as neutral.

To improve negative recall without retraining the model weights, I tuned the decision rule on validation data:

$$\hat{y} = 0 \text{ (negative) if } P(Y=0) \ge 0.34 \quad \text{and} \quad P(Y=0) > P(Y=2) + 0.05$$

Otherwise, the classifier falls back to standard $\arg\max$.

Evaluating this rule on the held-out test split:
- **Negative Recall**: Rose from **67.63% to 74.21%** (+6.58% absolute gain).
- **False Neutrals Rescued**: Negative-to-neutral errors dropped from 412 down to 300, recovering 112 true customer complaints.
- **Macro F1 Impact**: Macro F1 remained stable at **0.6378** (retaining 97.7% of peak performance while substantially improving complaint detection).

---

## 10. Summary & Lessons Learned

1. **Preserving Negations Matters More than Expanding Vocabulary**: Preserving negation tokens (`not`, `never`, `n't`) and expanding contractions provided a larger boost to negative-class recall than scaling the unigram vocabulary from 5,000 to 20,000 features.
2. **Recurrent Models from Scratch Aren't Worth It for Short Social Text**: Without pre-trained word embeddings, training a Bi-LSTM from scratch on 41k tweets overfit vocabulary noise and failed to beat regularized linear baselines while running 6x slower.
3. **Quantization Can Degrade Minority Classes Unpredictably**: INT8 dynamic quantization reduced disk weight size, but caused negative class recall to collapse from 80.8% down to 41.6% while failing to eliminate runtime memory spikes.
4. **Hardware Constraints Dictate Architecture**: When cloud free-tier hosting has a 512MB RAM ceiling, deploying a ~50MB hybrid linear model for edge requests and reserving the 125M parameter transformer for dedicated compute is far more reliable than attempting to force a large model into insufficient memory.
