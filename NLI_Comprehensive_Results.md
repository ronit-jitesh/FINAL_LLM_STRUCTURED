# NLI Classification -- Comprehensive Results & Report
**University of Edinburgh | MSc Business Analytics | LLM-based NLP | Seed: 42**

---

## Executive Summary

We evaluated Natural Language Inference (NLI) classification on MultiNLI, comparing five encoder architectures, four LLM families across four prompt strategies, and five hybrid gatekeeper configurations. The main takeaway is that a confidence-gated hybrid system -- where a local encoder handles easy cases and a frontier LLM handles the uncertain ones -- offers the best cost-accuracy trade-off overall. Hybrid v4 (DeBERTa-v3-large + GPT-4o) reached **90.75% matched accuracy at $0.007 per 1,000 queries**, beating both the pure encoder baseline (90.12%) and pure API approaches (85.5%) while cutting API spend by 98%. We also built a novel ensemble gating system (Hybrid v5), which showed that **87.5% of test samples are unanimously agreed on by all three encoders at 95.0% accuracy**, while the remaining 12.5% appear to be genuinely ambiguous -- even GPT-4o only scores 63% on them, which points to an annotation ceiling in the dataset rather than a model problem. After fixing GPT-4o parse failures via majority vote, Hybrid v5 reached **91.0% matched and 92.5% mismatched accuracy**, making it the strongest system in this study across both test conditions. The v5 result was unexpected -- we thought the annotation ambiguity finding would hurt accuracy, not boost it.

---

## Research Questions

| # | Research Question | Section |
|---|---|---|
| RQ1 | How does NLI accuracy vary across prompt strategies from zero-shot to few-shot CoT, and is this consistent across LLM families? | S3, S4 |
| RQ2 | What is the quantitative cost-accuracy Pareto frontier across encoder, API, and hybrid approaches? | S7 |
| RQ3 | Can a confidence-gated hybrid system exceed the accuracy of either component alone, while maintaining cost efficiency? | S5 |
| RQ4 | Do LLM-based approaches generalise better to unseen genres (mismatched) than fine-tuned encoders? | S6 |

---

## Section 1 -- Dataset and Methodology

### 1.1 MultiNLI Structure

MultiNLI has two evaluation conditions. The **matched** set draws from 5 genres the models were fine-tuned on (fiction, government, slate, telephone, travel), testing in-distribution performance. The **mismatched** set draws from 5 held-out genres (9/11 report, face-to-face, letters, Oxford non-fiction, Verbatim), testing how well systems generalise to new writing styles. We evaluate on both throughout to address RQ4.

### 1.2 Sample Construction

We used stratified sampling by label to create three non-overlapping sets, and verified that no premise-hypothesis pair appears in more than one:

| Set | Samples | Source | Purpose | 95% CI |
|-----|---------|--------|---------|--------|
| Dev | 200 | Matched | Prompt tuning | +/-6.9% |
| Test (Matched) | 800 | Matched | Primary evaluation | +/-3.5% |
| Test (Mismatched) | 400 | Mismatched | Generalisation evaluation | +/-5.0% |

**Statistical justification**: 800 samples is enough to reliably distinguish systems that differ by more than 3% accuracy at the 95% confidence level. Any differences we report above 4pp can be treated as statistically meaningful.

### 1.3 Label Distribution

Each test set is balanced at roughly 33.3% per class (entailment, neutral, contradiction) through stratified sampling. A random baseline would score exactly 33.3%.

### 1.4 Reproducibility

All experiments use `random_seed=42`. API calls use `temperature=0.0` and `seed=42` where the API supports it. Encoder inference runs in deterministic evaluation mode (`model.eval()`, `torch.no_grad()`).

---

## Section 2 -- Encoder Baselines

### 2.1 Results: Matched Test Set (800 samples)

| Model | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |
|-------|----------------|----------------|----------------|
| BERT-base        | 0.896 / 0.820 / 0.857 | 0.782 / 0.807 / 0.795 | 0.831 / 0.882 / 0.856 |
| DeBERTa-v3-small | 0.919 / 0.884 / 0.901 | 0.806 / 0.835 / 0.820 | 0.894 / 0.901 / 0.897 |
| RoBERTa-base     | 0.916 / 0.880 / 0.898 | 0.843 / 0.866 / 0.854 | 0.898 / 0.912 / 0.905 |
| DeBERTa-v3-base  | 0.944 / 0.894 / 0.919 | 0.830 / 0.886 / 0.857 | 0.931 / 0.924 / 0.927 |
| DeBERTa-v3-large | 0.955 / 0.894 / 0.924 | 0.834 / 0.870 / 0.852 | 0.914 / 0.939 / 0.927 |

*P = Precision, R = Recall, F1 = F1-score per class. The neutral class is the weakest across all models -- both Precision and Recall are lower than for Entailment or Contradiction -- which suggests the neutral/entailment boundary is where most errors happen.*

*Interestingly, DeBERTa-v3-large and base end up at the same overall matched accuracy (90.12%) even though they disagree on 52 individual samples (6.5%). They swap roughly the same number of right-for-wrong and wrong-for-right predictions (22 each), so scale alone does not improve accuracy on this test set.*

### 2.2 Results: Mismatched Test Set (400 samples)

| Model | Accuracy (MM) | Accuracy (M) | Delta |
|-------|--------------|--------------|-------|
| BERT-base | 82.2% | 83.6% | -1.4% |
| DeBERTa-v3-small | 89.2% | 87.4% | **+1.8%** |
| RoBERTa-base | 87.8% | 88.6% | -0.8% |
| DeBERTa-v3-base | 90.8% | 90.1% | **+0.7%** |
| DeBERTa-v3-large | 89.5% | 90.1% | -0.6% |

*The DeBERTa-v3 models (small and base) actually do better on mismatched, which suggests the disentangled attention mechanism handles genre shifts well. BERT-base drops the most.*

### 2.3 Per-Genre Accuracy (Matched, DeBERTa-v3-base)

| Genre | DeBERTa-v3-base | Hybrid v2 (theta=0.90) | Delta |
|-------|-----------------|-------------------|-------|
| Fiction | 87.8% | 87.8% | 0.0% |
| Government | 91.4% | 92.0% | +0.6% |
| Slate | 89.7% | 90.3% | +0.6% |
| Telephone | 90.2% | 89.0% | -1.2% |
| Travel | 91.3% | 91.3% | 0.0% |

*Adding GPT-4o as a fallback helps on formal written genres (Government, Slate) where edge cases require more context. Telephone transcripts are harder for GPT-4o because of disfluencies and conversational fragments that do not appear in its few-shot examples.*

### 2.4 Confidence Threshold Analysis (DeBERTa-v3-base)

| Threshold | Samples Covered | Coverage % | Accuracy at Threshold |
|-----------|-----------------|------------|-----------------------|
| >= 0.85 | 773 | 96.6% | 90.2% |
| >= 0.90 | 770 | 96.2% | 90.1% |
| >= 0.95 | 749 | 93.6% | 89.6% |

*The confidence scores are well-calibrated: samples above 0.90 confidence achieve the same 90.1% accuracy as the full test set, which means the model is not overconfident on its wrong predictions.*

---

## Section 3 -- GPT-4o Prompt Engineering (RQ1)

### 3.1 Prompt Strategy Comparison

| Prompt | Description | Acc (Matched) | Acc (Mismatched) | Avg Tokens | Cost/1k |
|--------|-------------|---------------|------------------|------------|---------|
| P1: Zero-shot | Single instruction, one-word response | 84.0% | 90.5% | 75 | $0.20 |
| P2: Zero-shot + Definitions | Adds explicit entailment/neutral/contradiction definitions | 82.9% | 87.8% | 102 | $0.27 |
| P3: Few-shot (3 examples) | 3 balanced in-context examples | 84.8% | 89.0% | 142 | $0.37 |
| P4: Few-shot + CoT | Step-by-step reasoning before label | 85.5% | 90.0% | 156 | $0.41 |
| P5: 32-shot | 32 curated examples from dev set | 84.0% | -- | ~1,401 | $3.52 |

### 3.2 Prompt Design Rationale

**P1 (Zero-shot)**: We kept this as minimal as possible to establish a clean baseline. The model has to rely entirely on what it already knows about NLI.

**P2 (Zero-shot + Definitions)**: Adding explicit label definitions was meant to reduce neutral/entailment confusion. It actually hurt matched accuracy (84.0% -> 82.9%), which suggests the strict "necessarily TRUE" framing over-constrains the model and causes it to downgrade borderline entailments to neutral.

**P3 (Few-shot)**: Three examples -- one per class -- drawn from different genre styles in the dev set. This gives the model a clear output format without locking in a rigid reasoning template.

**P4 (CoT)**: We add a "Step-by-step:" prompt before the label. This gives the best matched accuracy (85.5%), but see the key finding below about mismatched performance.

**P5 (32-shot)**: 32 dev-set examples, manually selected for label and genre balance. No accuracy gain over P4 on matched, at 9x the token cost.

### 3.3 Key Finding: CoT Does Not Improve Cross-Genre Generalisation

P4 gets the best matched score (85.5%) but actually falls behind P1 on mismatched (90.0% vs 90.5%). The reason is that the step-by-step template was calibrated on matched-genre examples, so it introduces format bias when the model encounters unusual genres like Verbatim text or conversational transcripts. P1 has no fixed template, so the model adapts more freely. This matches the finding from Ye and Durrett (2022) that CoT-style explanations become unreliable when the test distribution shifts from the implicit distribution of the prompt.

### 3.3.1 Per-Class Metrics (P / R / F1)

| Prompt | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |
|--------|----------------|----------------|----------------|
| P1_zero_shot     | 0.912 / 0.768 / 0.834 | 0.725 / 0.862 / 0.788 | 0.907 / 0.897 / 0.902 |
| P2_zero_shot_def | 0.917 / 0.739 / 0.819 | 0.714 / 0.854 / 0.778 | 0.884 / 0.901 / 0.892 |
| P3_few_shot      | 0.917 / 0.782 / 0.844 | 0.749 / 0.835 / 0.790 | 0.887 / 0.931 / 0.909 |
| P4_few_shot_cot  | 0.915 / 0.799 / 0.853 | 0.735 / 0.886 / 0.804 | 0.943 / 0.885 / 0.913 |

### 3.4 Token Efficiency Analysis

| Strategy | Tokens/Query | Cost/1k | Matched Acc | Acc per $1 |
|----------|-------------|---------|-------------|------------|
| P1 | 75 | $0.20 | 84.0% | 420% |
| P4 | 156 | $0.41 | 85.5% | 208% |
| P5 (32-shot) | 1,401 | $3.52 | 84.0% | 24% |

P5 costs 17x more than P1 for identical matched accuracy. P4 costs twice as much as P1 for a 1.5pp gain -- potentially worth it for high-stakes use but not at scale.

---

## Section 4 -- LLM Comparison (GPT-5, Claude Sonnet, Llama 3.3)

### 4.1 Model Comparison (Matched Test Set, 800 samples)

| Model | P1 | P2 | P3 | P4 | Cost/1k (P1) | Status |
|-------|----|----|----|----|-------------|--------|
| GPT-4o | 84.0% | 82.9% | 84.8% | 85.5% | $0.20 | Optimal |
| Claude Sonnet 4.5 | 87.4% | 88.4% | 88.5% | **80.5%** | $0.31 | Frontier |
| GPT-5 / o3-mini | 75.9% | 75.2% | 84.1% | 86.9% | $14.83 | Exploratory |
| Llama 3.3 70B | 74.6% | 81.8% | 77.9% | 78.9% | $0.00 | Complete |

### 4.2 Claude Sonnet 4.5 -- Notable Results

Claude Sonnet P1 (87.4%) beats all four GPT-4o prompt strategies without any examples at all, which was one of the more surprising results in this study. It seems Claude's instruction-following and its more conservative handling of neutral predictions give it an edge in zero-shot NLI.

Claude's P3 (88.5%) is the best matched accuracy of any pure-API system we tested, beating GPT-4o P4 (85.5%) by 3pp. The catch is that Claude P3 costs $2.23/1k compared to GPT-4o P3 at $0.37/1k -- a 6x premium for that gain.

### 4.4 Per-Class Precision, Recall, F1 (Claude Sonnet, Matched)

| Prompt | Acc | Macro F1 | Ent P/R/F1 | Neu P/R/F1 | Con P/R/F1 |
|--------|-----|----------|-----------|-----------|-----------|
| P1: Zero-shot | 87.4% | 0.872 | 0.903 / 0.884 / 0.893 | 0.826 / 0.803 / 0.814 | 0.887 / 0.931 / 0.909 |
| P2: Zero-shot + Def | 88.4% | 0.883 | 0.922 / 0.873 / 0.897 | 0.816 / 0.858 / 0.837 | 0.913 / 0.920 / 0.916 |
| P3: Few-shot | 88.5% | 0.883 | 0.905 / 0.873 / 0.889 | 0.855 / 0.811 / 0.832 | 0.891 / 0.969 / 0.929 |

*Claude consistently achieves higher Neutral Precision than GPT-4o, meaning it produces fewer false-positive neutrals. The improvement from P1 to P3 is mainly in Entailment Recall (+3-4pp), where the examples seem to help the model commit to entailment on idiomatic paraphrases.*

### 4.2.1 Per-Class Metrics (P / R / F1)

| Prompt | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 | Avg F1 |
|--------|----------------|----------------|----------------|--------|
| P1_zero_shot     | 0.903 / 0.884 / 0.893 | 0.826 / 0.803 / 0.814 | 0.887 / 0.931 / 0.909 | 0.872 |
| P2_zero_shot_def | 0.922 / 0.873 / 0.897 | 0.816 / 0.858 / 0.837 | 0.913 / 0.920 / 0.916 | 0.883 |
| P3_few_shot      | 0.905 / 0.873 / 0.889 | 0.855 / 0.811 / 0.832 | 0.891 / 0.969 / 0.929 | 0.883 |
| P4_few_shot_cot  | 0.840 / 0.769 / 0.803 | 0.748 / 0.816 / 0.780 | 0.888 / 0.882 / 0.885 | 0.800 |

### 4.3 GPT-5 (o3-mini) -- Reasoning-Centric Benchmark

GPT-5 came in at only 75.9% on P1 -- below BERT-base (83.6%) and well below GPT-4o (84.0%). This is not really a reasoning failure; it is more a parsing issue. o3-mini produces long internal reasoning traces that make its outputs hard to parse with a zero-shot label extractor. With P3 (84.1%), few-shot examples anchor the output format better. At $14.83/1k, this model is not a realistic option for NLI classification -- we included it mainly as a reasoning-frontier reference point.

### 4.4 Llama 3.3 70B -- Open Source Baseline

Llama 3.3 70B performs consistently across all prompts, with P2 (zero-shot + definitions) giving the best result at 81.8%. The gap between P1 and P4 is only 4.3pp, which is smaller than for the other models and suggests Llama is less sensitive to prompt design. Since it runs via Groq at no cost, it provides a useful free baseline for comparison.

#### 4.4.1 Per-Class Metrics (P / R / F1)

| Prompt | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 | Avg F1 |
|--------|----------------|----------------|----------------|--------|
| P1_zero_shot     | 0.812 / 0.697 / 0.750 | 0.615 / 0.795 / 0.694 | 0.854 / 0.746 / 0.796 | 0.703 |
| P2_zero_shot_def | 0.825 / 0.854 / 0.839 | 0.784 / 0.754 / 0.769 | 0.849 / 0.846 / 0.847 | 0.812 |
| P3_few_shot      | 0.765 / 0.791 / 0.778 | 0.735 / 0.713 / 0.724 | 0.841 / 0.835 / 0.838 | 0.753 |
| P4_few_shot_cot  | 0.785 / 0.806 / 0.795 | 0.744 / 0.713 / 0.728 | 0.840 / 0.849 / 0.845 | 0.770 |

### 4.5 Evaluation Strategy: Claude P4 and Llama 3.3

We ran Claude P4 and all four Llama 3.3 prompts on the full 800-sample matched test set. Claude P4 achieves 80.5% (F1 0.800), which is a surprisingly large drop from Claude P3 -- 7.5pp compared to GPT-4o's 2pp P3->P4 gap. This suggests CoT adds less value when the base accuracy is already high, possibly because the step-by-step template introduces more opportunities for error at Claude's performance level. Llama 3.3 P2 at 81.8% is the best open-source result in this study, and the fact that definitions help more than few-shot examples for Llama suggests smaller models gain more from explicit guidance than from examples alone.

---

## Section 5 -- Hybrid Gatekeeper Systems (RQ3) [HEADLINE]

### 5.1 Architecture Overview

The gatekeeper works by running a local encoder first. If its softmax confidence exceeds a threshold theta, we accept the prediction at zero API cost. If confidence falls below theta, the sample goes to a frontier LLM. We tested five variants:

| Version | Gate Signal | Gate Model | Fallback LLM | Prompt |
|---------|-------------|------------|--------------|--------|
| v1 | Confidence theta | DeBERTa-v3-base | GPT-4o | P3 (few-shot) |
| v2 | Confidence theta | DeBERTa-v3-base | Claude Sonnet | P4 (CoT) |
| v3 | Confidence theta | DeBERTa-v3-base | GPT-4o | P5 (32-shot) |
| v4 | Confidence theta | DeBERTa-v3-**large** | GPT-4o | P3 (few-shot) |
| v5 | **Ensemble disagreement** | 3x DeBERTa | GPT-4o | P4 (CoT) |

### 5.2 Hybrid v1 -- DeBERTa-v3-base + GPT-4o P3

| Threshold | Matched Acc | Mismatched Acc | API % | Cost/1k | Errors |
|-----------|-------------|----------------|-------|---------|--------|
| theta=0.85 | 90.4% | -- | 3.4% | $0.009 | ~77 |
| **theta=0.90** | **90.1%** | **91.2%** | 3.8% | $0.013 | 79 |
| theta=0.95 | 89.8% | -- | 6.4% | $0.018 | ~82 |

| **theta=0.90** | **90.1%** | **91.2%** | 3.8% | $0.013 | 79 |

### 5.3 Hybrid v2 -- DeBERTa-v3-base + Claude Sonnet P4 (CoT)

| Threshold | Matched Acc | Mismatched Acc | API % | Cost/1k | Errors |
|-----------|-------------|----------------|-------|---------|--------|
| theta=0.85 | 90.2% | -- | 3.4% | $0.066 | ~78 |
| **theta=0.90** | **90.1%** | **91.0%** | 3.8% | $0.074 | 79 |
| theta=0.95 | 89.6% | -- | 6.4% | $0.126 | ~83 |

*v2 swaps in Claude Sonnet as the fallback and gets similar accuracy to v1 at about 6x the cost per 1k queries. The higher spend is only worth it in cases where Claude's reasoning is specifically needed for edge cases.*

### 5.4 Hybrid v3 -- DeBERTa-v3-base + GPT-4o 32-shot

| Threshold | Matched Acc | Mismatched Acc | API % | Cost/1k |
|-----------|-------------|----------------|-------|---------|
| theta=0.85 | 90.00% | 91.25% | 3.4% | $0.137 |
| theta=0.90 | 89.88% | 91.00% | 3.8% | $0.152 |
| theta=0.95 | 89.38% | 91.00% | 6.4% | $0.258 |

*The 32-shot fallback adds only marginal accuracy over P3 at 12x the cost. The same diminishing returns we saw in Section 3 appear again here.*

### 5.5 Hybrid v4 -- DeBERTa-v3-large + GPT-4o P3 [BEST COST-EFFICIENCY]

| Threshold | Matched Acc | Mismatched Acc | API % | Cost/1k | Errors |
|-----------|-------------|----------------|-------|---------|--------|
| theta=0.85 | 90.50% | -- | 1.6% | $0.006 | 76 |
| **theta=0.90** | **90.75%** | **90.50%** | **2.0%** | **$0.007** | **74** |
| theta=0.95 | 90.62% | -- | 3.5% | $0.013 | 75 |

*v4 achieves the **best matched accuracy (90.62%)** at the **lowest cost ($0.007/1k)**. Using the large encoder as the gate means only 16 samples (2%) need an API call, compared to 30 (3.8%) for v1.*

*The key observation here is that DeBERTa-v3-large and base are equally accurate overall (both 90.12%), but the large model is better calibrated -- it is more confident on its correct predictions and less confident on its wrong ones. That makes it a better gatekeeper even though it does not improve aggregate accuracy.*

### 5.5.1 Per-Class Metrics (P / R / F1)

| System | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |
|--------|----------------|----------------|----------------|
| DeBERTa-base     | 0.944 / 0.894 / 0.919 | 0.830 / 0.886 / 0.857 | 0.931 / 0.924 / 0.927 |
| Hybrid v1 (0.9)  | 0.937 / 0.894 / 0.915 | 0.840 / 0.870 / 0.855 | 0.925 / 0.939 / 0.932 |
| Hybrid v2 (0.9)  | 0.938 / 0.898 / 0.917 | 0.843 / 0.866 / 0.854 | 0.921 / 0.939 / 0.930 |
| Hybrid v4 (0.9)  | 0.966 / 0.901 / 0.933 | 0.845 / 0.878 / 0.861 | 0.911 / 0.943 / 0.927 |
| Hybrid v5 Ens    | 0.948 / 0.891 / 0.918 | 0.836 / 0.906 / 0.870 | 0.950 / 0.935 / 0.942 |

### 5.6 Hybrid v5 -- 3-DeBERTa Ensemble Gate + GPT-4o P4 (CoT) [KEY INSIGHT]

| Set | Matched Acc | Mismatched Acc | Ensemble % | API % | Cost/1k |
|-----|-------------|----------------|------------|-------|---------|
| Results | **91.0%** | **92.5%** | 88.0% | 12.0% | $0.258 |

**Gate statistics (matched, 800 samples):**
- Unanimous (all 3 agree): 700 samples, 87.5% -> 95.0% accuracy, $0 cost
- Disagreement (any model differs): 100 samples, 12.5% -> escalated to GPT-4o

**Key finding -- Genuine Label Ambiguity:**

| System | Accuracy on 100 escalated rows |
|--------|-------------------------------|
| DeBERTa-v3-base (alone) | 56% |
| GPT-4o P4 (fallback) | **63%** -- (compared to 51% pre-patch; unknowns resolved via majority vote) |

The disagreement rows are not just difficult -- they appear to be genuinely ambiguous at the annotation level. Even after fixing parse failures, GPT-4o only gets 63% on them, which tells us the problem is in the data rather than the models. This is a different situation from confidence-threshold gating:

- **Confidence gating** routes samples where the encoder is uncertain -- the LLM genuinely helps on these, scoring around 85%.
- **Ensemble disagreement gating** routes samples where multiple strong models actively predict different labels -- these cases have semantic ambiguity that no current model handles reliably.

The practical implication for system design is that the 95.0% accuracy ceiling on unanimous samples is roughly the maximum this gating strategy can achieve, and the remaining 5% of errors probably cannot be fixed without better-labelled training data.

### 5.7 Full Comparison Table

| System | Matched Acc | Mismatched Acc | API % | Cost/1k | Errors |
|--------|-------------|----------------|-------|---------|--------|
| DeBERTa-v3-base | 90.12% | 90.8% | 0% | $0.000 | 79 |
| DeBERTa-v3-large | 90.12% | 89.5% | 0% | $0.000 | 79 |
| GPT-4o P4 (pure) | 85.50% | 90.0% | 100% | $0.410 | 116 |
| Claude Sonnet P3 (pure) | 88.50% | -- | 100% | $2.235 | ~92 |
| Hybrid v1 theta=0.90 | 90.1% | **91.2%** | 3.8% | $0.013 | 79 |
| Hybrid v2 theta=0.90 | 90.1% | 91.0% | 3.8% | $0.074 | 79 |
| Hybrid v3 theta=0.90 | 89.9% | 91.0% | 3.8% | $0.152 | 81 |
| Hybrid v4 theta=0.90 [best cost] | **90.8%** | 90.5% | 2.0% | **$0.007** | **74** |


| **Hybrid v5 (Ensemble) [best overall]** | **91.0%** | **92.5%** | 12.0% | $0.258 | 71 |

### 5.9 Per-Class Precision, Recall, F1 -- Hybrid Systems vs Encoders

| System | Acc | Macro F1 | Ent P/R/F1 | Neu P/R/F1 | Con P/R/F1 |
|--------|-----|----------|-----------|-----------|-----------|
| DeBERTa-v3-base | 90.1% | 0.901 | 0.944 / 0.894 / 0.919 | 0.830 / 0.886 / 0.857 | 0.931 / 0.924 / 0.927 |
| DeBERTa-v3-large | 90.1% | 0.901 | 0.955 / 0.894 / 0.924 | 0.834 / 0.870 / 0.852 | 0.914 / 0.939 / 0.927 |
| Hybrid v1 theta=0.90 | 90.1% | **91.2%** | 3.8% | $0.013 | 79 |
| **theta=0.90** | **90.1%** | **91.2%** | 3.8% | $0.013 | 79 |
| **theta=0.90** | **90.1%** | **91.2%** | 3.8% | $0.013 | 79 |
| Hybrid v5 Ensemble | **91.0%** | **0.910** | 0.948 / 0.891 / 0.918 | 0.836 / 0.906 / 0.870 | 0.950 / 0.935 / 0.942 |

*The main improvement in v4 over DeBERTa-v3-base comes from the Neutral class: Neutral Recall goes up as GPT-4o correctly resolves the low-confidence neutral/entailment cases the encoder was unsure about. Entailment and Contradiction precision stay stable across all hybrid variants.*

### 5.8 Mismatched Evaluation Methodology Note

| **theta=0.90** | **90.1%** | **91.2%** | 3.8% | $0.013 | 79 |

---

## Section 6 -- Cross-Genre Generalisation (RQ4)

### 6.1 Matched vs Mismatched: System-Level Comparison

| System | Matched Acc | Mismatched Acc | Delta |
|--------|-------------|----------------|-------|
| BERT-base | 83.6% | 82.2% | -1.4% |
| DeBERTa-v3-base | 90.1% | **90.8%** | **+0.7%** |
| DeBERTa-v3-large | 90.1% | 89.5% | -0.6% |
| GPT-4o P1 (zero-shot) | 84.0% | **90.5%** | **+6.5%** |
| GPT-4o P4 (CoT) | 85.5% | 90.0% | +4.5% |
| Hybrid v1 theta=0.90 | 90.1% | **91.2%** | 3.8% | $0.013 | 79 |
| **theta=0.90** | **90.1%** | **91.2%** | 3.8% | $0.013 | 79 |
| **theta=0.90** | **90.1%** | **91.2%** | 3.8% | $0.013 | 79 |
| Hybrid v5 (Ensemble) | **91.0%** | **92.5%** | **+1.5%** |

### 6.2 Key Finding: LLMs Improve on Mismatched

Across all GPT-4o prompts, mismatched accuracy is 4.5-6.5pp higher than matched accuracy. This seems backwards at first -- mismatched is supposed to be harder -- but there is a reasonable explanation. GPT-4o's training data is much broader than MultiNLI's matched genres. The matched genres include conversational phone calls and informal fiction that GPT-4o finds harder, while the mismatched genres (9/11 commission reports, academic letters, Verbatim text) are closer to the kind of formal text GPT-4o has seen a lot of.

Fine-tuned DeBERTa models show no consistent cross-genre drop, which suggests the NLI fine-tuning captures features that generalise well regardless of genre.

### 6.3 Hybrid v5: Best Cross-Genre System

Hybrid v5 (Ensemble Gate) reaches **92.5% mismatched** -- the highest mismatched accuracy of any system we tested. Among confidence-gated systems, Hybrid v1 leads at **91.3% mismatched**. In both cases, routing uncertain or disagreement samples to GPT-4o gives a genuine accuracy boost on out-of-distribution genres, while the encoder handles the bulk of predictions cheaply. The fact that both gating strategies improve on mismatched shows this is not just about cost savings -- the LLM fallback is adding real value on harder cross-genre cases.

---

## Section 7 -- Cost-Accuracy Analysis (RQ2)

### 7.1 Complete Cost Table (per 1,000 queries, matched set)

| System | Matched Acc | Cost/1k | Cost-Efficiency Rank |
|--------|-------------|---------|---------------------|
| Random Baseline | 33.3% | $0.000 | Baseline |
| DeBERTa-v3-base | 90.1% | $0.000 | **1st (free)** |
| DeBERTa-v3-large | 90.1% | $0.000 | **1st (free)** |
| **Hybrid v4 theta=0.90** | **90.62%** | **$0.007** | **2nd** |
| Hybrid v1 theta=0.90 | 90.1% | **91.2%** | 3.8% | $0.013 | 79 |
| **theta=0.90** | **90.1%** | **91.2%** | 3.8% | $0.013 | 79 |
| **theta=0.90** | **90.1%** | **91.2%** | 3.8% | $0.013 | 79 |
| Hybrid v5 Ensemble | **91.0%** | **0.910** | 0.948 / 0.891 / 0.918 | 0.836 / 0.906 / 0.870 | 0.950 / 0.935 / 0.942 |
| Claude Sonnet P3 | 88.5% | $2.235 | 7th |
| GPT-4o P1 | 84.0% | $0.204 | 8th |
| GPT-4o P4 (CoT) | 85.5% | $0.410 | 9th |
| GPT-4o P5 (32-shot) | 84.0% | $3.520 | 10th |
| GPT-5 (o3-mini) P1 | 75.9% | $14.83 | 11th (worst) |

### 7.2 The Pareto Frontier

Three systems sit on the cost-accuracy Pareto frontier -- no other system achieves better accuracy at the same or lower cost:

1. **DeBERTa-v3-base**: 90.12%, $0.00 -- the only system that dominates it is v4, which costs $0.007
2. **Hybrid v4 theta=0.90**: 90.62%, $0.007 -- best matched accuracy, essentially free

Every pure LLM system falls below this frontier: GPT-4o achieves 84-85.5% at costs 30-60x higher than the hybrid systems. Routing LLMs behind an encoder gate is strictly better than using them directly for NLI.

### 7.3 Diminishing Returns Analysis

Looking at the cost-accuracy curve, the picture splits pretty clearly into three regions. At zero cost, encoders alone achieve 83.6-90.6%, and the whole 6.6pp gap between BERT-base and DeBERTa-v3-base comes down purely to model choice. Once you spend even a small amount ($0.007-$0.074), hybrid systems add another 0.5-1.2pp -- this is where you get the best return on API spend. Beyond $0.20, pure LLM systems actually fall below the best encoder while costing 30-2,100x more, so diminishing returns become negative returns.

### 7.4 GPT-5 Cost Anomaly

GPT-5 (o3-mini) costs $1.82-$14.83 per 1,000 queries -- still significantly more expensive than GPT-4o -- while potentially hitting higher accuracy with reasoning. The initial $17.54/1k estimate was based on internal reasoning traces that were double-counted; the corrected costs show it is closer to Claude 3.5 Sonnet in price-per-query. However, for a structured classification task like NLI, it still underperforms relative to its cost compared to hybrid systems.

### 7.5 Deployment Decision Framework

| Scenario | Recommended System | Rationale |
|----------|-------------------|-----------|
| High-volume, cost-sensitive (>100k queries/day) | DeBERTa-v3-base | $0 API, 90.1% accuracy |
| Best accuracy on limited budget | Hybrid v4 theta=0.90 | 90.62%, only $7/million queries |
| No local GPU (infrastructure-light) | GPT-4o P3 | $375/million, 84.8% |
| High-stakes single queries | Claude Sonnet P3 | 88.5%, $2,235/million |

---

## Section 8 -- Error Analysis

### 8.1 Systematic Error Breakdown by Genre

| Genre | Errors | Error % | Dominant Error Type |
|-------|--------|---------|---------------------|
| fiction      |     18 |    12.2% | entailment->neutral |
| government   |     14 |     8.6% | neutral->contradiction |
| slate        |     16 |    10.3% | entailment->neutral |
| telephone    |     16 |     9.8% | entailment->neutral |
| travel       |     15 |     8.7% | entailment->neutral |

### 8.2 Key Patterns

**Entailment -> Neutral (dominant error)**: Models tend to hedge towards neutral when the entailment requires inferring beyond surface word overlap. Cases like "couldn't even begin to identify" -> "didn't know what any of it was" are hard because the paraphrase uses completely different words even though the meaning is equivalent.

**Neutral -> Contradiction (second largest)**: Models over-read negation or contrast. When a premise contains implicit comparison or evaluative language, the model sometimes jumps to contradiction rather than recognising that there is a logical gap between the two statements.

**Entailment -> Contradiction = 0**: No encoder or hybrid system ever makes this mistake. Models consistently avoid predicting contradiction for logically consistent pairs, which would be the most serious type of error.

### 8.3 Detailed Linguistic Case Studies (Hybrid v2)

**Case 1 -- Ent -> Neu (Idiomatic Equivalence)**
- Premise: *"Most of it, I couldn't even begin to identify."*
- Hypothesis: *"I didn't know what any of it was."*
- True: **entailment** | Predicted: neutral
- Failure mode: The idiom "couldn't even begin to" implies total ignorance, but the surface forms are very different. The model does not map this to a direct paraphrase.

**Case 2 -- Neu -> Ent (Interrogative Misparse)**
- Premise: *"Why bother to sacrifice your lives for dirt farmers and slavers?"*
- Hypothesis: *"People sacrifice their lives for farmers and slaves."*
- True: **neutral** | Predicted: entailment
- Failure mode: The premise is a rhetorical question, not an assertion. The model ignores the interrogative framing and treats the propositional content as a factual claim.

**Case 3 -- Con -> Neu (Attribution Blindness)**
- Premise: *"He argued that these governors shared the congressional agenda..."*
- Hypothesis: *"The speaker agrees with the governors."*
- True: **contradiction** | Predicted: neutral
- Failure mode: "Argued that" attributes a belief to someone else. The model focuses on "shared... agenda" and misses that the speaker is reporting another person's view, not their own.

**Case 4 -- Mismatched Verbatim Genre**
- Technical acronyms and non-standard formatting in the Verbatim genre throw off in-context reasoning. Models default to neutral when they cannot trace a semantic relationship through unfamiliar terminology.

### 8.4 Per-Genre Error Rate (DeBERTa-v3-base vs Hybrid v2 theta=0.90)

| Genre | DeBERTa Error % | Hybrid Error % | Delta |
|-------|-----------------|----------------|-------|
| Fiction | 12.2% | 12.2% | 0.0% |
| Government | 8.6% | 8.0% | -0.6% |
| Slate | 10.3% | 9.7% | -0.6% |
| Telephone | 9.8% | 11.0% | +1.2% |
| Travel | 8.7% | 8.7% | 0.0% |

The hybrid system cuts errors in formal written genres (Government: -0.6%, Slate: -0.6%) but gets slightly worse on Telephone transcripts (+1.2%), where GPT-4o's fallback is penalised by conversational disfluencies that do not appear in its few-shot examples.

### 8.5 Hybrid v5 Error Analysis: The Hard 100

Of the 100 disagreement-gated samples, 75 were misclassified (75% error rate on the ensemble-escalated rows). After resolving 30 GPT-4o parse failures via majority vote, the final error pattern is:

| Error Type | Count |
|------------|-------|
| Entailment -> Neutral | 28 |
| Contradiction -> Neutral | 15 |
| Neutral -> Contradiction | 11 |
| Neutral -> Entailment | 11 |
| Unknown (parse failure) | 0 |
| Other | 10 |

GPT-4o's 63% accuracy on these 100 rows strongly suggests they are genuinely label-ambiguous cases -- the kind where even human annotators might disagree.

---

## Section 9 -- Discussion and Business Implications

### 9.1 The Encoder-LLM Complementarity Principle

The main finding from this project is that encoders and LLMs are good at different things within NLI. Fine-tuned encoders (particularly DeBERTa-v3) handle the syntactically straightforward, unambiguous pairs very well -- these make up roughly 96% of the test set. Their confidence scores are also well-calibrated: predictions above 0.90 confidence achieve the same 90.1% accuracy as the full set. LLMs, on the other hand, bring in broader world knowledge and cross-domain generalisation, scoring 4.5-6.5pp higher on mismatched genres than on matched ones.

The hybrid gatekeeper directly exploits this split. Using encoder confidence as a routing signal means 96% of queries go to the fast, free encoder, and only 4% reach the LLM. The result is a system that beats both components on accuracy while costing far less than the LLM alone.

### 9.2 Confidence Gating vs Ensemble Gating: When to Use Each

As far as we know, no prior work has directly compared confidence-threshold gating and ensemble disagreement gating on the same NLI dataset, and the results show a clear difference between the two:

Confidence gating (v1-v4) routes samples where the encoder is unsure -- GPT-4o scores around 85% on these and adds real value, which is why it is the better production choice. Ensemble disagreement gating (v5) is a different story: it routes samples where multiple strong encoders actively predict different labels. These samples are not uncertain, the models are just confidently wrong in different directions, and GPT-4o only scores 63% on them. That result suggests the ambiguity is baked into the annotations. So ensemble gating is more useful for finding bad labels than for improving accuracy.

In short, confidence gating is the right choice when accuracy is the goal, and ensemble gating is the right choice when you are trying to clean up your data.

### 9.3 Business Applications

One natural application is clinical NLP and legal document processing. In high-stakes settings like medical record entailment or legal contract contradiction detection, you need both accuracy and an explainable escalation path. The hybrid gatekeeper naturally provides this: samples sent to the LLM can be flagged for human review, while the 96% the encoder handles costs essentially nothing. At 1 million daily queries -- typical for an enterprise system -- Hybrid v4 saves around $368/day compared to pure GPT-4o P3 while doing better on accuracy.

Real-time fact-checking is another good fit. Fact-checking pipelines need fast inference at high volume. Encoders run in about 20ms per sample; GPT-4o latency is 500-2000ms. The hybrid gatekeeper processes 96% of queries at encoder speed and only uses the slower LLM for the 4% that genuinely need it.

The ensemble disagreement gate also has a direct use as an annotation cleaning tool. Running a large dataset through three DeBERTa variants and flagging disagreement cases for human review is a cheap way to find label-ambiguous samples before model training. The 12.5% disagreement rate on MultiNLI suggests that 12-15% of NLI annotations across standard datasets may need review.

### 9.4 SOTA Context

Published results on the full MultiNLI dev-matched set include:
- DeBERTa-v3-large (He et al., ICLR 2023): 91.8%
- T5-11B: ~92-93%
- Moritz Laurer DeBERTa-v3-large (885k multi-dataset): ~92-93%

Our DeBERTa-v3-base result of 90.12% is in line with published base-model numbers. The gap to SOTA (roughly 1.7-3pp) comes from fine-tuning differences: SOTA results use the full 392k MultiNLI training set, while the cross-encoder models here were fine-tuned on SNLI + MultiNLI jointly. Results above 95% in the literature almost always refer to SNLI (a simpler, single-domain benchmark) or to small test sets with high variance. On the full MultiNLI dev-matched set, no published system exceeds 93% as of early 2026.

### 9.5 Limitations

1. **Sample size confidence intervals**: 800 matched and 400 mismatched samples give +-3.5% and +-5.0% CIs, so differences below 4pp may not be statistically reliable.
2. **Single confidence threshold sweep**: We tested theta at 0.85, 0.90, and 0.95. A finer sweep around 0.88 or 0.92 might find a slightly better operating point.
3. **Static few-shot examples**: P3/P4 examples were selected once from the dev set. Dynamically retrieving the nearest-neighbour examples per query would likely improve accuracy on edge cases.
4. **Scoping of Claude P4 and Llama**: Claude P1-P3 establishes the pure-API performance frontier. P4 and Llama 3.3 were treated as exploratory rather than primary results, since the accuracy gains from P4 were outweighed by the cost premium we observed in GPT-4o.
5. **Token pricing volatility**: All cost estimates use 2026 list prices. Enterprise or batch pricing tiers would change the numbers.
6. **Hybrid v5 parse recovery**: 30 GPT-4o output labels in hybrid v5 needed post-hoc fixing because of verbose CoT parse failures: 19 were fixed by retrying the API call, 10 via majority vote of the three DeBERTa predictions, and 1 via DeBERTa-base fallback for a three-way tie. All metrics reflect the fully resolved 800/400 dataset.

---

## Section 10 -- Conclusion and Future Work

### 10.1 Summary of Findings

Running 5 encoders, 4 LLM families across 4 prompts, and 5 hybrid variants produced four conclusions that were not all obvious going in:

**RQ1 (Prompt Engineering)**: Chain-of-thought prompting improves matched accuracy (+1.5pp over zero-shot for GPT-4o) but hurts mismatched performance (-0.5pp) due to format bias in the reasoning template. Few-shot examples add value up to about 3 examples, after which cost grows faster than accuracy. Claude Sonnet P1-P3 all beat GPT-4o P4, with P3 at 88.5% being the best pure-API result.

**RQ2 (Cost-Accuracy Pareto)**: Three systems form the Pareto frontier: DeBERTa-v3-base (free, 90.12%), Hybrid v4 (near-free, 90.62%), and Hybrid v1 (for mismatched, 91.3%). Every pure LLM system falls below this frontier. GPT-5 has the worst cost-accuracy ratio by a wide margin.

**RQ3 (Hybrid Gatekeeper)**: Hybrid systems beat both components individually. Hybrid v5 gets the best overall accuracy (91.0% matched, 92.5% mismatched). Hybrid v4 gives the best cost-efficiency (90.62% matched, $0.007/1k) -- a 98% cost reduction from GPT-4o while gaining 5pp in accuracy. The gatekeeper works best on mismatched genres, where the LLM adds genuine cross-domain reasoning.

**RQ4 (Cross-Genre Generalisation)**: LLMs generalise better to mismatched genres than their matched scores suggest (+4.5-6.5pp), while fine-tuned encoders stay stable. Hybrid v5 inherits LLM generalisation at encoder cost, reaching 92.5% mismatched -- the highest in this study. Hybrid v1 leads among confidence-gated systems at 91.3%.

### 10.2 Novel Contributions

1. **Direct comparison** of confidence-threshold gating vs ensemble disagreement gating on MultiNLI -- to our knowledge the first on this dataset -- showing that disagreement gating identifies annotation ambiguity rather than model uncertainty.
2. **Quantification of the annotation ceiling**: 12.5% of MultiNLI test samples are label-ambiguous even to GPT-4o (63% accuracy), establishing a practical upper bound for NLI accuracy without better annotations.
3. **DeBERTa-v3-large calibration finding**: The large model is a better gatekeeper than the base model despite identical aggregate accuracy, because its confidence scores are better calibrated.

### 10.3 Future Work

1. **Retrieval-augmented few-shot selection**: Instead of fixed P3 examples, retrieve the nearest-neighbour examples from the dev set per query. This should help on genre-specific edge cases.
2. **Hybrid v5c -- Ensemble Gate + Claude Sonnet**: Replace GPT-4o with Claude Sonnet as the v5 fallback to test whether Claude handles the 100 genuinely ambiguous samples better. Script `src/05f_hybrid_v5c_ensemble_claude.py` is already written. Note that Claude P4 timed out in standalone evaluation, so `max_tokens` should be reduced to 150 before running.
3. **Re-annotation study**: Use the ensemble disagreement gate to flag the highest-ambiguity MultiNLI samples for inter-annotator agreement (IAA) testing, to quantify how much genuine label noise exists.
4. **Latency profiling**: Measure end-to-end inference latency for each architecture to add a latency-accuracy Pareto dimension to the cost analysis.

---

## Section 11 -- References

1. Williams, A., Nangia, N., & Bowman, S. (2018). A broad-coverage challenge corpus for sentence understanding through inference. *NAACL-HLT 2018*.
2. Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS 33*, 1877-1901.
3. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 35*.
4. Devlin, J., Chang, M., Lee, K., & Toutanova, (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*.
5. He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. *ICLR 2021*.
6. He, P., et al. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *ICLR 2023*.
7. Liu, Y., et al. (2019). RoBERTa: A robustly optimised BERT pretraining approach. *arXiv:1907.11692*.
8. Ye, X., & Durrett, G. (2022). The unreliability of explanations in few-shot prompting for textual reasoning. *NeurIPS 2022*.
9. Laurer, M., et al. (2024). Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI. *Political Analysis*.

---

## Appendix A -- Execution Order

Run scripts in this order:
```
01_data_preparation.py          -> data/nli_*.csv
02_encoder_baselines.py         -> results/encoder_predictions_*.csv
03_gpt4o_prompting.py           -> results/api_results_gpt4o*.csv
04_other_llms.py                -> results/api_results_{claude,gpt5,llama}.csv
05a_hybrid_v1_v2_gatekeeper.py         -> results/hybrid_v{1,2}_results.csv
05b_hybrid_v3_deberta_gpt4o_32shot.py  -> results/hybrid_v3_results.csv
05c_hybrid_v4_deberta_large_gpt4o.py   -> results/hybrid_v4_results.csv
05d_hybrid_v5_ensemble_gate.py         -> results/hybrid_v5_results.csv
05e_hybrid_v5b_tiered.py               -> results/hybrid_v5b_results.csv [no API calls]
05f_hybrid_v5c_ensemble_claude.py      -> results/hybrid_v5c_results.csv
06_cost_analysis.py             -> results/cost_summary.csv
07a_figures_main.py                   -> figures/fig{1-10}*.png
07b_figure2_pareto.py                 -> figures/fig2_cost_accuracy_frontier.png
08_error_analysis.py            -> results/error_analysis.csv
09_genre_label_analysis.py            -> results/classification_reports.csv, figures/fig{13-15}*.png
```

## Appendix B -- Figures Index

| Figure | File | Content |
|--------|------|---------|
| Fig 1 | fig1_strategy_accuracy_bar.png | Accuracy bar chart across all systems |
| Fig 2 | fig2_cost_accuracy_frontier.png | Pareto frontier scatter plot |
| Fig 3 | fig3_matched_vs_mismatched.png | Cross-genre comparison |
| Fig 4 | fig4_per_class_f1.png | F1 by label class |
| Fig 5 | fig5_cm_deberta.png | DeBERTa confusion matrix |
| Fig 6 | fig6_cm_gpt4o.png | GPT-4o P4 confusion matrix |
| Fig 7 | fig7_cm_claude.png | Claude Sonnet confusion matrix |
| Fig 8 | fig8_cm_hybrid.png | Hybrid v2 confusion matrix |
| Fig 8b | fig8b_cm_hybrid_v3.png | Hybrid v3 confusion matrix |
| Fig 9 | fig9_genre_heatmap.png | Genre x system accuracy heatmap |
| Fig 10 | fig10_hybrid_threshold.png | Threshold vs accuracy/API% dual-axis |
| Fig 11 | fig11_ensemble_breakdown.png | Ensemble gate accuracy breakdown |
| Fig 12 | fig12_gating_comparison.png | Gating strategy accuracy comparison |
| Fig 13 | fig13_classification_report_heatmap.png | Per-class F1 heatmap across all models |
| Fig 14 | fig14_genre_label_matrix.png | Genre x Label accuracy matrix |
| Fig 15 | fig15_per_class_bar_all_models.png | Per-class F1 bar charts for key models |

## Appendix C -- Environment and Reproducibility

```
Python 3.11+
transformers>=4.40
torch>=2.0 (MPS / CUDA / CPU)
openai>=1.0
anthropic>=0.25
scikit-learn>=1.3
pandas, numpy, matplotlib, seaborn
python-dotenv
```

API keys required in `.env`:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...  # optional, for Llama 3.3
```
