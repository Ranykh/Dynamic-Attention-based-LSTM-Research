# Dynamic Attention based LSTM Research
We propose a sample-wise dynamic, attention-enhanced LSTM that adapts its computation per input to predict human decisions in text-based persuasion settings. While our approach is principled and efficient, experiments show a strong simple LSTM baseline remains competitive, motivating new directions for dynamic sequence modeling. 

<p align="center">
  <a href="LSTM_Research.pdf"><b>📄 Read the Full Report (PDF)</b></a>
</p>
---
## Problem & Motivation
Predicting human decisions from conversational text is crucial for human–AI interaction, safety, and adaptive assistance. Prior work (Shapira et al., 2023) found a simple LSTM surprisingly outperformed transformer baselines on this task. We revisit this setting and investigate whether attention and dynamic computation can close the gap while remaining efficient.
---
## Dataset
- **Source:** Human decisions from a mobile “Travel or Trouble” persuasion game + simulated player decisions (combined to improve generalization).
- **Scale:** 87,204 decisions by 245 people across multi-round interactions with strategy-driven agent messages.
---
## Approach

1) Start with a strong static baseline
We first fine-tune a static Attention-based LSTM to establish a high-quality base model. 


2) Masked self-attention (no future peeking)
Naïve attention can leak future tokens. We implement masked attention so each step only attends to current/past states—preserving causal forecasting. 

3) Sample-wise dynamic policy
A lightweight policy network (fully connected layer) selects, per input, whether each layer:

- runs LSTM+Attention,

- runs LSTM only, or

- is skipped entirely.

We apply selective gradient updates so inactive components don’t receive gradients.


---
## Experiments (Summary)
- Static model tuning: We tuned layers ∈ {2,4,6}, hidden sizes ∈ {32,64,128}, and LR ∈ {1e-2, 1e-3, 1e-4} using W&B sweeps on human-only data for faster iteration. 

- Final training: Converted to dynamic model and trained 30 epochs on the combined human + simulated dataset; evaluated across 15 random seeds as in the original study. 

- Outcome: Despite architectural sophistication, the simple LSTM remained slightly better on average, highlighting the strength of streamlined recurrent baselines for this task and the need for refined dynamic policies.

---

## Key Contributions
- Causality-preserving masked attention layered over LSTM for persuasion-sequence prediction. 

- Dynamic, sample-wise policy that adapts layer usage and reduces unnecessary compute, trained with selective gradient updates. 

- Empirical insight: In this domain, a well-optimized simple LSTM is a tough baseline—even against attention + dynamic computation.

---
## Tech Stack

- Python, PyTorch/Keras, NumPy, pandas
- Training tooling: Weights & Biases for sweeps
- Modeling: LSTM, masked self-attention, dynamic policy network (FC), softmax classification head
