# SentimentScope: Multi-Model Sentiment Analysis & Telemetry Platform
**Concise Project Report & Technical Evaluation (2–3 Pages)**  
*CardiffNLP TweetEval 3-Class Sentiment Benchmark*

---

| Property | Value | Property | Value |
| :--- | :--- | :--- | :--- |
| **Dataset** | CardiffNLP TweetEval (59,641 Tweets) | **Evaluation Scope** | Stratified 70/15/15 Split (8,947 Held-out Test) |
| **Framework Stack** | Scikit-Learn, Keras, PyTorch, Hugging Face, FastAPI | **Production Edge** | Render Cloud Free Tier (Docker, 512MB RAM, <50MB RSS) |

---

## 1. Problem Statement

Social media microblog sentiment analysis (Twitter/X) is essential for brand reputation monitoring, automated customer support alerting, and consumer trend detection. However, microblog text presents severe linguistic and operational hurdles: heavy informal slang, non-standard spelling, contractions, emojis, and nuanced negation syntax (e.g., *"not terrible"*, *"never flying with them again"*).

Furthermore, social sentiment corpora exhibit pronounced class imbalance (Neutral tweets outnumber Negative tweets by 2.4 to 1). Standard machine learning models consistently suffer from **"neutral bleed"**—misclassifying subtle customer complaints as neutral. In customer success and brand protection, missing a customer complaint (a false neutral) is catastrophic, leading directly to customer churn and escalation, whereas neutral/positive boundary confusion carries a far lower commercial penalty.

Compounding this linguistic challenge is a stringent cloud engineering constraint: state-of-the-art transformer architectures (e.g., RoBERTa, DistilBERT) contain 110M–125M parameters, requiring over 450MB in static weights and spiking runtime memory past 1.5GB. On cloud free-tier hosting platforms (such as Render's 512MB RAM ceiling), loading a standard transformer triggers an immediate out-of-memory termination (`Exit 137`). The goal of **SentimentScope** is to engineer, calibrate, and benchmark a complete multi-model pipeline spanning statistical, recurrent, and transformer architectures, optimize negative-class recall through negation whitelisting and decision threshold tuning, and deploy a memory-safe dual-tier cloud architecture delivering sub-millisecond cloud edge inference within a 50MB RAM footprint.

---

## 2. Dataset Description (Source + Size + Features)

- **Source & Benchmark**: The system utilizes the **CardiffNLP TweetEval 3-Class Sentiment Benchmark** (Barbieri et al., SemEval / Hugging Face). The corpus comprises authentic, real-world microblog tweets categorized into three mutual classes: `Negative (0)`, `Neutral (1)`, and `Positive (2)`.
- **Dataset Size & Stratification**: The dataset contains **59,641 unique tweets**, partitioned into a strict, reproducible stratified 70/15/15 partition:
  - **Training Set**: 41,748 samples (Neutral: 19,126 [45.8%], Positive: 14,678 [35.2%], Negative: 7,944 [19.0%]).
  - **Validation Set**: 8,946 samples (Used for grid search, hyperparameter tuning, Platt calibration, and decision threshold selection).
  - **Test Set**: 8,947 samples (Held-out evaluation split used strictly for final comparative benchmark reporting).
- **Feature Engineering & Preprocessing Pathways**:
  1. *Text Normalization*: Strips URLs, user mentions (`@user`), unescapes HTML entities, expands contractions (e.g., `can't` $\rightarrow$ `cannot`), and converts text to lowercase.
  2. *Negation Whitelisting*: Standard NLP stopword filters discard negation tokens, causing catastrophic sentiment polarity flips on phrases like *"not bad"*. Our pipeline enforces an explicit negation whitelist (`not`, `no`, `never`, `n't`, `cannot`, `without`) paired with WordNet lemmatization.
  3. *Hybrid Feature Union (Linear Tier)*: Combines sublinear TF-IDF word n-grams (1, 2) (vocabulary 20,000) with character boundary subwords (3, 5). This captures internal morphological roots so that out-of-vocabulary terms and social typos (e.g., *"sooooo bad"*) retain strong predictive signal.
  4. *Recurrent Sequences*: Keras integer sequence tokenization (`vocab_size=10,000`, `max_len=150`) with post-sequence padding for deep recurrent networks.
  5. *Transformer Tokenizers*: Byte-Pair Encoding (BPE, `max_length=64`) for Twitter-RoBERTa and WordPiece for DistilBERT Base.

---

## 3. Methods Used (Algorithms & Models)

Six distinct model architectures spanning three algorithmic paradigms were implemented, trained, and benchmarked on identical splits:
1. **Multinomial Naive Bayes (MNB)**: Statistical probabilistic baseline utilizing Laplace smoothing ($\alpha=1.0$) over unigram TF-IDF vectors.
2. **Vanilla Logistic Regression**: Linear baseline using L2 regularization ($C=1.0$, L-BFGS solver) over standard word unigram TF-IDF features.
3. **Bidirectional LSTM (Bi-LSTM)**: Deep recurrent architecture: `Embedding(128) -> SpatialDropout1D(0.2) -> Bi-LSTM(64) -> Dense(64) -> Softmax(3)` trained from scratch on 41k tweets using Adam optimizer and categorical cross-entropy.
4. **Hybrid TF-IDF + Logistic Regression (Active Cloud Edge)**: Modernized linear classifier over word n-grams (1,2) and character subwords (3,5) trained with inverse class-frequency loss reweighting ($w_{\text{neg}}=1.7519, w_{\text{neu}}=0.7276, w_{\text{pos}}=0.9481$) to counteract majority neutral bias.
5. **DistilBERT Base**: 66M parameter distilled transformer fine-tuned using AdamW ($lr=2\times 10^{-5}$, linear warmup, weight decay 0.01).
6. **Twitter-RoBERTa Base (Active Dedicated Tier)**: 125M parameter transformer pre-trained on 124M in-domain tweets and fine-tuned on TweetEval.

### Production Engineering & Serving Architecture
- **Sparse Matrix Memory Optimization**: Passing dense arrays (`.toarray()`) during grid search caused 6.7GB RAM spikes. Eliminating dense conversion and feeding SciPy CSR sparse matrices directly into scikit-learn solvers reduced memory consumption to 50MB and accelerated throughput by 50x.
- **Sigmoid Platt Calibration**: Uncalibrated linear model probabilities were overconfident. Applying Sigmoid Platt scaling reduced the Brier error score from 0.4688 to 0.4581, yielding well-calibrated posterior probabilities essential for confidence-based routing.
- **Asymmetric Decision Thresholding**: Rather than default $\arg\max$, negative sentiment is assigned whenever $P(Y=\text{neg}) \ge 0.34$ and $P(Y=\text{neg}) > P(Y=\text{pos}) + 0.05$. This recovers 112 false neutral complaints without retraining model weights.
- **Dual-Tier Cloud Architecture**: Tier 1 (Twitter-RoBERTa, 76.2% Acc) serves high-accuracy inference for dedicated infrastructure ($\ge 1$GB RAM). Tier 2 (Hybrid Logistic Regression, 65.9% Acc, 0.55ms latency, ~50MB RAM) operates as the cloud edge model on Render's 512MB free tier.

---

## 4. Results & Insights

All candidate models were evaluated on the identical 8,947-sample held-out test split. The empirical performance matrix is summarized below:

| Model Architecture | Model Family | Test Acc. | Macro F1 | Neg. Recall | CPU p50 Latency | Deployment Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Multinomial Naive Bayes** | Statistical | 59.73% | 0.5504 | 29.21% | 0.15 ms | Baseline Benchmark |
| **Vanilla Logistic Regression** | Linear (Unigrams) | 61.64% | 0.5850 | 39.72% | 0.42 ms | Baseline Benchmark |
| **Bi-LSTM (From Scratch)** | Recurrent Neural Net | 60.53% | 0.5954 | 60.28% | 3.12 ms | Deep Learning Baseline |
| **Hybrid TF-IDF + Logistic Reg.** | Linear (Word + Char) | **65.92%** | **0.6528** | **67.63%**\* | **0.55 ms** | **Active Cloud Edge (512MB RAM)** |
| **DistilBERT Base** | Distilled Transformer | 72.57% | 0.7221 | 74.22% | 14.01 ms | Benchmarked (Dedicated) |
| **Twitter-RoBERTa Base** | Domain Transformer | **76.22%** | **0.7610** | **80.79%** | **22.29 ms** | **Active Dedicated Tier ($\ge$1GB RAM)** |

*\*Note: Hybrid Logistic Regression negative recall reaches **74.21%** under $\theta_{\text{neg}} = 0.34$ threshold tuning (Macro F1 = 0.6378).*

![Pareto Latency vs Macro F1](figures/08_cpu_latency_vs_f1_tradeoff.png)

### Key Empirical Insights
1. **In-Domain Pretraining is Decisive**: Twitter-RoBERTa achieves peak performance (**76.22% accuracy, 0.7610 Macro F1, 80.79% negative recall**), outperforming general DistilBERT by +3.65% accuracy and +6.57% negative recall. Pre-training on 124M tweets equips RoBERTa to decipher social slang and contextual nuance.
2. **Hybrid Subwords Substantially Rejuvenate Linear Models**: Augmenting Logistic Regression with character 3–5 n-grams and negation whitelisting lifted accuracy from 61.64% to 65.92% (+4.28%) and surged negative recall from 39.72% to 67.63% (+27.91%), bridging the gap to deep learning at 0.55ms latency.
3. **Recurrent Models From Scratch Fail on Short Social Text**: Bi-LSTM trained from scratch achieved only 60.53% accuracy and 0.5954 F1. Without pre-trained embeddings (GloVe/fastText), the embedding layer overfit noisy spelling while running 6x slower (3.12ms) than linear models.
4. **The INT8 Dynamic Quantization Dead End**: Empirically profiling PyTorch Dynamic INT8 quantization on RoBERTa compressed disk weights by 51.4% (475MB $\rightarrow$ 230MB), but collapsed negative recall from 80.79% to 41.60% (-39.2%) and spiked peak runtime RAM to 1,930 MB during dequantization. This proves post-training quantization alone cannot fit transformers into a 512MB RAM budget, validating the two-tier architectural split.
5. **Threshold Tuning Rescues Customer Complaints**: Asymmetric thresholding ($\theta_{\text{neg}} = 0.34$) recovers 112 false neutrals, raising negative recall to 74.21% while maintaining high overall F1 (0.6378).

---

## 5. Limitations & Future Improvements

### Identified System Limitations
1. **Negative-Neutral Decision Boundary Ambiguity**: Error matrix analysis shows that remaining classification errors concentrate along the neutral/negative boundary. Sarcastic or politely understated complaints lacking explicit negative vocabulary are occasionally absorbed into the neutral class.
2. **Hardware Ceiling on Free Cloud Tiers**: While the Hybrid Logistic Regression model runs flawlessly within 50MB RAM on Render's 512MB free tier, the 76.2% accurate Twitter-RoBERTa engine requires a dedicated server ($\ge 1$GB RAM) and cannot run concurrently on free cloud edge hosting.
3. **Monolingual English Restriction**: The current benchmark and feature pipelines focus strictly on English Twitter text and do not generalize to multilingual social streams or code-mixed dialects (e.g., Hinglish, Spanglish).

### Future Technical Improvements
1. **Quantization-Aware Training (QAT) & ONNX Runtime**: Implement Quantization-Aware Training during fine-tuning rather than dynamic post-training quantization, allowing RoBERTa to be exported to ONNX Runtime under 150MB without degrading negative-class decision boundaries.
2. **Live Telemetry Active Learning Loop**: Connect the real-time SQLite rolling store to an active learning workflow, capturing low-confidence predictions (<55% probability) for automated human-in-the-loop validation and periodic model re-fitting.
3. **Multilingual Foundation Backbones**: Integrate XLM-RoBERTa or lightweight multilingual adapters to expand SentimentScope into global enterprise social listening feeds across multiple languages.
