# Can We Trust AI Explanations?
## Evidence of Systematic Underreporting in Chain-of-Thought Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2412.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2412.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Deep Pankajbhai Mehta (Adobe Inc.)

---

## Abstract

When AI systems explain their reasoning step-by-step, practitioners often assume these explanations reveal what actually influenced the AI's answer. We tested this assumption by embedding hints into questions and measuring whether models mentioned them. In a study of over 9,000 test cases across 11 leading AI models, we found a troubling pattern: models almost never mention hints spontaneously, yet when asked directly, they admit noticing them. This suggests models see influential information but choose not to report it. Telling models they are being watched does not help. Forcing models to report hints works, but causes them to report hints even when none exist and reduces their accuracy. We also found that hints appealing to user preferences are especially dangerous-models follow them most often while reporting them least. These findings suggest that simply watching AI reasoning is not enough to catch hidden influences.

---

## Key Findings

| Finding | Result |
|---------|--------|
| **Baseline acknowledgment rate** | 20.7% (models rarely report hints spontaneously) |
| **Perception rate** | 99.4% (models see hints when probed) |
| **Perception-acknowledgment gap** | 78.7 percentage points |
| **Monitoring effect** | +1.1pp, p=0.38 (not significant) |
| **False positive rate** | 68.2% (explicit instruction causes over-reporting) |
| **Accuracy cost** | -15.9pp (explicit instruction hurts performance) |
| **Sycophancy susceptibility** | 45.5% (highest of all hint types) |
| **Sycophancy acknowledgment** | 43.6% (hidden influence pattern) |

---

## Experiment Overview

We conducted **9,154 experimental trials** across **11 frontier language models**:

- GPT-5.2 (OpenAI)
- Claude Opus 4 (Anthropic)
- Claude Sonnet 4.5 (Anthropic)
- Gemini 3 Flash (Google)
- DeepSeek-R1 (DeepSeek)
- Grok-4 (xAI)
- Kimi-K2-Thinking (Moonshot)
- Qwen3-235B (Alibaba)
- GLM-4.7 (Zhipu)
- Nemotron-3-Nano (NVIDIA)
- Sonar-Reasoning-Pro (Perplexity)

### Three Experimental Conditions

1. **Baseline** (n=1,908): Standard CoT prompting
2. **Monitoring** (n=1,911): Models told reasoning is being monitored
3. **Explicit** (n=1,969): Models instructed to report any hints

### Seven Hint Types Tested

| Category | Hint Types |
|----------|------------|
| Expertise | Authority, Expert |
| Social | Majority, Sycophancy |
| Technical | Metadata, System, Confidence |

---

## Repository Structure

```
├── data/
│   ├── primary_results.csv      # 5,788 primary experiment trials
│   ├── extended_results.csv     # 3,366 extended experiment trials
│   └── model_metadata.json      # Model information
├── src/
│   ├── experiment.py            # Main experiment runner
│   ├── hint_generator.py        # Hint embedding logic
│   ├── acknowledgment_detector.py # Pattern matching for acknowledgment
│   └── analysis.py              # Statistical analysis
├── figures/
│   ├── fig1_main_results.png
│   ├── fig2_per_model.png
│   ├── fig3_gap.png
│   ├── fig4_tradeoff.png
│   └── fig5_hint_types.png
├── paper/
│   └── cot_paper_verified.pdf   # Full paper
└── README.md
```

---

## Main Results

### Figure 1: Acknowledgment by Condition
![Main Results](figures/fig1_main_results.png)

### Figure 2: Per-Model Variation
![Per Model](figures/fig2_per_model.png)

### Figure 3: The Perception-Acknowledgment Gap
![Gap](figures/fig3_gap.png)

### Figure 4: Transparency-Accuracy Tradeoff
![Tradeoff](figures/fig4_tradeoff.png)

### Figure 5: Hint Type Effects
![Hint Types](figures/fig5_hint_types.png)

---

## Practical Implications

Based on our findings:

1. **Do not trust absence of mention as evidence of non-influence.** ~80% of contextual influences go unreported.

2. **Simple monitoring notifications are insufficient.** Telling models they are being watched produces no significant improvement.

3. **Consider consequence-based monitoring framings.** +6pp improvement without false positive inflation.

4. **Be especially cautious of user-pleasing responses.** Sycophancy influences are commonly followed (45.5%) and commonly hidden (43.6% acknowledged).

5. **Avoid explicit acknowledgment prompts unless you can tolerate the tradeoffs.** 68.2% false positive rate and -15.9pp accuracy cost.

---

## Citation

```bibtex
@article{mehta2024trustai,
  title={Can We Trust AI Explanations? Evidence of Systematic Underreporting in Chain-of-Thought Reasoning},
  author={Mehta, Deep Pankajbhai},
  journal={arXiv preprint arXiv:2412.XXXXX},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Experiments conducted using the [OpenRouter API](https://openrouter.ai) for unified model access.

Questions derived from the [MMLU benchmark](https://github.com/hendrycks/test).

---

## Contact

- **Author:** Deep Pankajbhai Mehta
- **Affiliation:** Adobe Inc.
- **GitHub:** [@researchaudio](https://github.com/researchaudio)
