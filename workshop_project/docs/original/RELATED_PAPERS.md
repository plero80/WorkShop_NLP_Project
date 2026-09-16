# Related papers and what they support

1. **Gao, Schulman and Hilton — Scaling Laws for Reward Model Overoptimization
   (2022).** https://arxiv.org/abs/2210.10760
   Studies optimization against a proxy while evaluating with a separate gold
   reward model. This motivates measuring divergence under PPO. It does not
   establish that any individual proxy–judge disagreement is a confirmed hack,
   nor that this specific kNN refresh algorithm works.

2. **Khandelwal et al. — Generalization through Memorization: Nearest Neighbor
   Language Models (2019/2020).** https://arxiv.org/abs/1911.00172
   Uses nearest neighbors in a pretrained model representation space and a
   datastore for language-model predictions. This is prior art for the memory
   mechanism; its task is token prediction, not reward-gap correction.

3. **Christiano et al. — Deep Reinforcement Learning from Human Preferences
   (2017).** https://arxiv.org/abs/1706.03741
   Learns a reward signal from human comparisons to guide reinforcement
   learning. This is foundational context for feedback-driven reward learning;
   the present experiment substitutes a fixed model judge for the refresh labels
   and reserves human comparisons for evaluation.

These connections justify an experimental extension. They are not an exhaustive
novelty search, and none is evidence that the exact proposed method has already
succeeded. A suitable project claim is: **We test whether periodically adding
teacher-labeled policy outputs to a frozen reward-model representation memory
improves subsequent proxy-gap correction and policy optimization.**

# Implementation references

- Transformers generation parameters (`max_new_tokens`):
  https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
- PEFT adapter loading, saving and disabling:
  https://huggingface.co/docs/peft/main/en/package_reference/peft_model

The code pins model revisions, records installed package versions, validates
saved adapter tensors, and runs model-specific GPU parity checks. Documentation
links explain the APIs; they do not replace that validation.
