# KernelBench Level 4 - Pure PyTorch Implementations

This directory contains pure PyTorch reimplementations of 7 HuggingFace transformer
architectures, originally benchmarked in `KernelBench/level4/`.

## Models

| Model | Architecture | Files |
|-------|-------------|-------|
| GPT-2 | Causal LM with Conv1D, gelu_new | 7, 16, 19 |
| GPT-Neo 2.7B | Mixed global/local attention | 1, 3, 18 |
| OPT 1.3B | Pre-LN decoder, learned pos embed | 2, 4, 8 |
| BART Large | Post-LN decoder-only, embed scaling | 6, 17, 20 |
| BigBird RoBERTa | Block-sparse attention (MLM) | 5, 9, 10 |
| ELECTRA Small | Separate embed/hidden dims (CLM) | 11, 12, 14 |
| Reformer enwik8 | LSH + local attention, RevNet | 13, 15 |

## Structure

```
models/              Pure PyTorch model implementations
  config_utils.py    Config loading from HuggingFace hub
  weight_utils.py    Weight downloading and state_dict mapping
  gpt2.py            GPT-2
  gpt_neo.py         GPT-Neo
  opt.py             OPT
  bart.py            BART decoder-only
  bigbird.py         BigBird
  electra.py         ELECTRA
  reformer.py        Reformer
*.py                 20 benchmark files (same interface as level4/)
verify_precision.py  Precision verification script
```

## Usage

Each benchmark file provides the same interface as `level4/`:

```python
from level4_expand.7_gpt2_bs32_seq256 import Model, get_inputs, get_init_inputs

model = Model(*get_init_inputs())
inputs = get_inputs()
logits = model(*inputs)
```

## Verification

```bash
# Verify all models
python verify_precision.py

# Verify specific files
python verify_precision.py --files 7 16 19

# Custom threshold
python verify_precision.py --threshold 1e-4
```

## Design Decisions

- All models use `nn.Linear` instead of custom Conv1D; weights are transposed during loading
- Weight tying is applied after loading pretrained weights
- Causal masks use `float("-inf")` or dtype-specific min values
- Softmax computed in float32 where HF does so, then cast back to value dtype
- All verification done in eval mode (dropout disabled)
