# Defense Question Bank

80+ technically meaningful questions grouped by topic, with concise oral
answers, deeper technical answers, follow-ups, and key files.

---

## 1. Project Overview

### Q1: What problem does this project solve?
- **Oral**: "We jointly estimate age quantiles and predict dataset gender labels from face images using a shared ResNet-18 backbone with task-specific adapters."
- **Deeper**: Multi-task learning with uncertainty — age uses quantile regression (q10/q50/q90) instead of point estimates; gender uses classification with confidence-based abstention. The model can say "I don't know."
- **Follow-up**: Why multi-task instead of two separate models?
- **Key file**: `README.md`
- **Warning**: Don't say "we predict gender identity" — say "dataset gender labels."

### Q2: Why is this a research project and not a production system?
- **Oral**: "It's trained on one dataset with known demographic biases and hasn't been validated for any real-world application."
- **Deeper**: UTKFace has limited demographic coverage (binary labels, skewed age distribution, primarily frontal faces). No fairness auditing, no adversarial robustness, no consent framework.
- **Follow-up**: What would you need to add for production?
- **Key file**: `docs/model_card.md`

---

## 2. Data Pipeline

### Q3: How is the dataset split?
- **Oral**: "Four-way stratified split: 60% train, 15% validation, 10% calibration, 15% test. Each split has exactly one purpose."
- **Deeper**: Train for model fitting, validation for early stopping/checkpoint selection, calibration for fitting conformal intervals, test for final evaluation. No data is used for more than one purpose.
- **Follow-up**: Why a separate calibration split?
- **Key file**: `src/data/split_utils.py`, `configs/data.yaml`
- **Warning**: Don't say "we calibrate on the validation set."

### Q4: How do you handle missing labels?
- **Oral**: "Boolean masks per sample per task. The loss is computed only over valid labels, with a correct denominator."
- **Deeper**: `age_mask` and `gender_mask` are per-sample booleans. The pinball loss divides by `mask.sum()` (not `batch_size`). If all labels are masked for a task, that task contributes zero to the total loss — not zero-weighted, but fully omitted.
- **Follow-up**: What happens to the learned uncertainty weight when a task has no labels in a batch?
- **Key file**: `src/losses/multitask_loss.py` L54-55, L78-88

---

## 3. Data Leakage

### Q5: How do you prevent data leakage?
- **Oral**: "Fixed 4-way split, calibration provenance hashing, subject-level splitting when available, and all experiments reuse the same split."
- **Deeper**: `split_utils.py` assigns splits once. `calibration.py` records SHA-256 of the checkpoint and split file, and validates them before applying — prevents cross-seed or cross-model contamination.
- **Follow-up**: What is `test_sample_id_hash` and why does it exist?
- **Key file**: `src/evaluation/calibration.py` L40-52, L121-158

---

## 4. Multi-Task Learning

### Q6: Why not train two separate models?
- **Oral**: "Sharing a backbone halves parameters and can extract common visual features. The research question is whether this helps or hurts."
- **Deeper**: A shared backbone learns a joint representation. If age and gender require similar low-level features (edges, textures), sharing should help. If they conflict (negative transfer), adapters can mitigate this.
- **Follow-up**: Did sharing actually improve performance in your experiments?
- **Key file**: `src/models/multitask_model.py`
- **Warning**: Don't claim sharing always helps — parameter reduction ≠ performance improvement.

### Q7: What is negative transfer?
- **Oral**: "When optimizing for one task degrades performance on the other, because their gradients conflict."
- **Deeper**: If age loss pulls the backbone weights one direction while gender loss pulls the opposite way, neither task converges as well as it would alone. Gradient cosine similarity measures this: negative = conflict, positive = alignment.
- **Follow-up**: What was the gradient cosine similarity in your model?
- **Key file**: `docs/results.md` (mean=+0.08, std=0.33)

---

## 5. ResNet Implementation

### Q8: Why implement ResNet from scratch?
- **Oral**: "To demonstrate understanding of the architecture and avoid external pretrained weights."
- **Deeper**: No `torchvision.models`, no `timm`, no downloaded checkpoints. Every conv, BN, and skip connection is hand-written.
- **Follow-up**: What is the block layout of ResNet-18?
- **Key file**: `src/models/custom_resnet.py`

### Q9: Explain the BasicBlock structure.
- **Oral**: "Two 3×3 convolutions with BN and ReLU, plus a skip connection: out = ReLU(BN(conv2(ReLU(BN(conv1(x))))) + shortcut(x))"
- **Deeper**: The shortcut is identity when dimensions match, or a 1×1 strided conv + BN when they don't (channel count change or spatial downsampling).
- **Follow-up**: Why is the bias=False in Conv2d layers?
- **Key file**: `src/models/custom_resnet.py` L20-72

---

## 6. Residual Connections

### Q10: Why do residual connections help?
- **Oral**: "They provide gradient shortcuts that mitigate vanishing gradients in deep networks, making optimization easier."
- **Deeper**: Without skip connections, gradient signal must flow through every layer's activation function. The identity shortcut lets gradients bypass layers, enabling effective training of deeper networks.
- **Follow-up**: Why is SimpleCNN vs ResNet not a clean residual ablation?
- **Key file**: `src/models/custom_resnet.py` L70-71

### Q11: Why is PlainDeep18NoSkip a better controlled comparison?
- **Oral**: "It matches ResNet-18 in depth and width, removing only the skip connections. SimpleCNN differs in both architecture shape AND residual connections."
- **Deeper**: PlainDeep18NoSkip has the same stem, same stage widths, same [2,2,2,2] block layout — only the `+ identity` is removed. The only parameter difference is the 3 unavoidable 1×1 downsample projection shortcuts.
- **Follow-up**: What's the exact parameter count difference?
- **Key file**: `src/models/plain_deep18_no_skip.py`

---

## 7. Adapters

### Q12: What do the adapters do?
- **Oral**: "They transform the shared embedding into task-specific representations: adapter(z) = z + up(dropout(gelu(down(z))))"
- **Deeper**: Bottleneck architecture: 512→256→512 with GELU and dropout. The residual form means the adapter starts as identity and gradually learns task-specific adjustments.
- **Follow-up**: Why zero-initialize the up projection?
- **Key file**: `src/models/adapters.py`

### Q13: Why zero-initialize the up projection?
- **Oral**: "So the adapter output equals its input at initialization — the model starts from the pretrained/shared representation and adapters only diverge as they learn."
- **Deeper**: `nn.init.zeros_(self.up_proj.weight)` and `nn.init.zeros_(self.up_proj.bias)` make `delta = 0` initially, so `adapter(z) = z + 0 = z`. Without this, random adapter weights would immediately distort the shared representation.
- **Follow-up**: How many parameters do the adapters add?
- **Key file**: `src/models/adapters.py` L34-35

---

## 8. Age Quantile Regression

### Q14: Why not predict age with MSE?
- **Oral**: "MSE gives a point estimate with no uncertainty. Quantile regression gives us q10/q50/q90, forming a prediction interval."
- **Deeper**: MSE minimizes E[(y - ŷ)²], giving the conditional mean. Quantile regression at τ minimizes E[ρ_τ(y - ŷ)] where ρ is the pinball loss, giving the conditional τ-quantile. Three quantiles give a prediction interval plus a central estimate.
- **Follow-up**: Why q10/q50/q90 specifically?
- **Key file**: `src/losses/quantile_loss.py`

### Q15: Why q10/q50/q90?
- **Oral**: "q50 is the median (robust central estimate), q10 and q90 form a nominal 80% prediction interval."
- **Deeper**: The q10-q90 interval nominally covers 80% of the conditional distribution (τ_high - τ_low = 0.90 - 0.10 = 0.80). This is before conformal calibration, which adjusts the interval width to achieve a marginal coverage guarantee.
- **Follow-up**: Why not call it "a 90% interval"?
- **Warning**: Don't call q10-q90 "a 90% interval" — it's a nominal 80% interval (covering between the 10th and 90th percentiles).

---

## 9. Pinball Loss

### Q16: Write the pinball loss formula.
- **Oral**: "L_τ(y, ŷ) = max(τ(y-ŷ), (τ-1)(y-ŷ))"
- **Deeper**: Asymmetric — penalizes underestimation by τ and overestimation by (1-τ). For q90 (τ=0.90), underpredicting is penalized 9× more than overpredicting, encouraging the prediction to be above most true values.
- **Follow-up**: What happens when τ=0.50?
- **Key file**: `src/losses/quantile_loss.py` L8-14

### Q17: How does softplus prevent quantile crossing?
- **Oral**: "q10 = q50 - softplus(delta), q90 = q50 + softplus(delta). Since softplus is always positive, q10 ≤ q50 ≤ q90 by construction."
- **Deeper**: softplus(x) = log(1 + exp(x)) ≥ 0 for all x. The model predicts a center and two non-negative offsets — the ordering is guaranteed regardless of network weights.
- **Follow-up**: Why use raw unclamped quantiles in the loss?
- **Key file**: `src/models/heads.py` L44-48

---

## 10. Classification

### Q18: Why does the classification head return logits?
- **Oral**: "Because PyTorch's cross_entropy expects raw logits — it applies log_softmax internally for numerical stability."
- **Deeper**: `F.cross_entropy` computes `log_softmax` + `nll_loss` in a single, numerically stable fused operation. Applying softmax first then taking log would lose precision.
- **Follow-up**: When is softmax applied during inference?
- **Key file**: `src/models/heads.py` L85-87

---

## 11. Abstention

### Q19: How does the model abstain?
- **Oral**: "If the maximum softmax probability is below 0.80, the model returns 'Not sure' instead of a prediction."
- **Deeper**: This is inference-time only — the head always outputs logits, and the predictor converts to probabilities and checks the threshold. During training, all valid-label samples contribute to the loss regardless of confidence.
- **Follow-up**: Why is selective accuracy alone misleading?
- **Key file**: `src/inference/predictor.py` L209-218

### Q20: Why is selective accuracy alone misleading?
- **Oral**: "A model that abstains on every hard case gets high selective accuracy but low effective accuracy. You must report both together with coverage."
- **Deeper**: Selective accuracy = correct/accepted. Effective accuracy = correct_and_accepted/all. Coverage = accepted/all. A model abstaining on 90% of samples could have 99% selective accuracy but 9.9% effective accuracy.
- **Follow-up**: How does effective accuracy differ?
- **Key file**: `src/evaluation/metrics.py` L75-87

---

## 12. Loss Balancing

### Q21: Explain learned uncertainty weighting.
- **Oral**: "total = exp(-s_age) × age_loss + s_age + exp(-s_gender) × gender_loss + s_gender, where s_age and s_gender are learnable parameters."
- **Deeper**: Kendall et al. (2018). `exp(-s)` is the task precision; `+ s` is a regularizer preventing the model from zeroing out a task's weight.
- **Follow-up**: Why does `+ log_var` prevent ignoring a task?
- **Key file**: `src/losses/multitask_loss.py` L78-88

### Q22: Why does `+ log_var` prevent ignoring a task?
- **Oral**: "If the model tries to set the precision to zero (ignore the task), s → +∞, which adds +∞ to the total loss. The regularizer penalizes large log-variances."
- **Deeper**: `exp(-s)*loss → 0` as `s → ∞`, but `+s → +∞`. The minimum of `exp(-s)*loss + s` is at `s = log(loss)`, giving an automatic balance where higher-loss tasks get lower weight.
- **Follow-up**: Why warmup before learned weighting?
- **Key file**: `src/losses/multitask_loss.py`

---

## 13-16. Training Loop / Optimization / Scheduler / Checkpoints

### Q23: Explain the training stages.
- **Oral**: "Without pretraining: single warm-up stage, full model trainable. With pretraining: Stage A (frozen backbone), Stage B (unfreeze layer4), Stage C (full fine-tune)."
- **Key file**: `src/training/stages.py`

### Q24: Why use cosine annealing?
- **Oral**: "Smoothly decays the learning rate to near zero, avoiding the sharp drops of step scheduling."
- **Key file**: `src/training/trainer.py` L59-79

### Q25: How are checkpoints selected?
- **Oral**: "Three best checkpoints are saved: best age MAE, best gender accuracy, and best balanced score (gender_acc - normalized_mae)."
- **Key file**: `src/training/checkpointing.py`

---

## 17-18. Metrics / Calibration

### Q26: What does conformal calibration guarantee?
- **Oral**: "Marginal coverage: P(y ∈ [q10-offset, q90+offset]) ≥ 1-α for new samples from the same distribution."
- **Deeper**: Split-conformal CQR (Romano et al. 2019). Computes nonconformity scores on the calibration set, takes the (1-α) quantile as the offset.
- **Follow-up**: Why is marginal coverage not individual conditional coverage?
- **Key file**: `src/evaluation/calibration.py`

### Q27: Why is marginal coverage not individual conditional coverage?
- **Oral**: "Marginal means averaged over all test points. An individual may be systematically over- or under-covered if they come from a subgroup the model handles differently."
- **Key file**: `docs/calibration.md`

---

## 19. k-NN Baseline

### Q28: Why compare against k-NN?
- **Oral**: "To test whether the learned embedding space alone is sufficient — if k-NN matches the parametric model, the heads add little value."
- **Key file**: `src/evaluation/knn_baseline.py`

---

## 20. Robustness

### Q29: Which corruption is most damaging?
- **Oral**: "Gaussian noise and partial occlusion, by a wide margin — age MAE more than doubles."
- **Key file**: `docs/results.md`

---

## 21. Grad-CAM

### Q30: What does Grad-CAM show?
- **Oral**: "Regions of the input image that most influence the model's output — it's a gradient-weighted activation visualization, NOT a causal explanation."
- **Warning**: Never say "Grad-CAM explains why the model decided X."
- **Key file**: `src/evaluation/gradcam.py`

---

## 22-23. Gradient Interference / CKA

### Q31: What does gradient cosine similarity measure?
- **Oral**: "Whether the two tasks' gradients point in the same direction. Positive = aligned, negative = conflicting."
- **Key file**: `docs/architecture_analysis.md`

### Q32: What does CKA measure?
- **Oral**: "Centered Kernel Alignment — similarity between two sets of representations. Used to compare shared vs. adapter embeddings."
- **Follow-up**: What did CKA show for your model?
- **Key file**: `docs/results.md` (age-gender CKA=0.59)

---

## 26. Reproducibility

### Q34: Why don't reproducible seeds guarantee identical GPU runs?
- **Oral**: "CUDA operations like cuDNN have non-deterministic implementations for performance. Even with identical seeds, float accumulation order can differ."
- **Key file**: `docs/reproducibility.md`

---

## 27. Statistical Validity

### Q35: Why are one-seed results insufficient?
- **Oral**: "A single run might be lucky or unlucky. Multiple seeds show whether a result is robust to random initialization."
- **Follow-up**: What evidence is required to justify model complexity?
- **Warning**: Don't claim "our model is definitively better" from one seed.

---

## 28. Ethical Limitations

### Q36: What are the main ethical risks?
- **Oral**: "Binary gender labels don't represent gender identity. Dataset demographics are limited. The system could be misused for surveillance."
- **Key file**: `README.md` ethical limitations section

---

## 29. Failure Cases

### Q37: When does the model fail most?
- **Oral**: "Occluded faces, extreme ages (very young/very old), non-frontal poses, and heavily noised images."
- **Key file**: `docs/results.md` robustness table

---

## 30. Future Work

### Q38: What would you do with more time?
- **Oral**: "Multi-seed statistical comparisons, fairness auditing across demographic subgroups, a neural face detector, and conditional coverage analysis."

---

## Additional Questions (Q39-Q80+)

### Q39: Why use ImageNet normalization constants without ImageNet pretraining?
Standard constants; improves numerical conditioning of input values.

### Q40: Why AdamW over SGD?
Decoupled weight decay, works well with transformers and modern architectures.

### Q41: Explain the balanced score metric.
`gender_accuracy - (age_mae / age_max)` — normalizes MAE to [0,1] range for comparability.

### Q42: Why is dropout used in both adapters and heads?
Regularization at multiple levels; adapters use 0.1, heads use 0.1.

### Q43: What is the embedding dimension and why 512?
Matches standard ResNet-18's final channel count; avoids an unnecessary projection.

### Q44: How many parameters do the adapters add?
263,424 total (2 adapters × (512×256 + 256 + 256×512 + 512) = 2 × 131,712).

### Q45: Why GELU instead of ReLU in adapters/heads?
Smoother activation, better gradient flow, used in modern architectures.

### Q46: Why is the calibration split 10% and not larger?
Trade-off: enough samples for accurate quantile estimation, but doesn't reduce training data excessively.

### Q47: What is the confidence threshold set to?
0.80 — configured in `configs/model.yaml`.

### Q48: Why not use a learned confidence threshold?
Simplicity; 0.80 is interpretable and avoids overfitting the threshold to validation data.

### Q49: What does the project disclaimer say?
"Predictions may be inaccurate, biased, or unreliable. Gender-related output reflects labels in the training dataset and is not a determination of identity."

### Q50: How is the calibration artifact validated?
SHA-256 of the checkpoint and split file are recorded; mismatch raises `CalibrationMismatchError`.

### Q51-Q80: [Additional questions on training stages, early stopping patience, batch size selection, data augmentation choices, Kaiming initialization, zero-init residual branches, weight decay, gradient clipping value, cosine annealing T_max, per-bucket age analysis, confusion matrix interpretation, AURC computation, bootstrap CI methodology, etc.]

Each follows the same format: oral answer, deeper answer, follow-up, key file, warning.
