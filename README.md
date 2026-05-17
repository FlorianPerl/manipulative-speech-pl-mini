# Manipulative Speech Detection (Polish) — LoRA PoC

Binary classification of Polish text for propaganda and disinformation using parameter-efficient fine-tuning on Llama 3 8B.

## What it does
- Labels Polish text as `MANIPULACJA` or `NEUTRALNY`
- Provides a one-sentence justification per classification
- Targets manipulation patterns: fear appeals, false dilemmas, conspiracy framing, ad hominem, catastrophism

## Stack
| Component | Detail |
|---|---|
| Base model | `unsloth/llama-3-8b-Instruct-bnb-4bit` — Llama 3 8B, pre-quantized to 4-bit |
| Fine-tuning | LoRA (`r=16`) via Hugging Face PEFT — ~40M trainable out of 8.04B parameters |
| Optimization | Unsloth — 2–5x faster training, reduced VRAM draw |
| Environment | Google Colab, T4 GPU (15GB VRAM) |

## How to run
1. Open `manipulative-speech-pl-mini.ipynb` in Google Colab
2. Set runtime: `Runtime → Change runtime type → T4 GPU`
3. Run cells in order (Steps 1–6)

## Pipeline
| Step | Description |
|---|---|
| 1 | Install dependencies |
| 2 | Load model and tokenizer |
| 3 | Configure LoRA adapters |
| 4 | Format Polish dataset with Llama 3 chat template |
| 5 | Run supervised fine-tuning (SFT) |
| 6 | Inference on unseen text |

## Dataset
- 20 labeled Polish examples (10 manipulation, 10 neutral)
- Each label includes an annotation of the specific technique used
- Mock dataset — sized to validate the pipeline, not for production accuracy

## Example output
```
Input:
"Tylko ślepi zwolennicy obecnego rządu nie widzą, że te decyzje
doprowadzą nasz kraj do absolutnej ruiny gospodarczej."

Output:
MANIPULACJA (Stygmatyzacja zwolenników rządu, katastrofizm,
fałszywa pewność co do przyszłości).
```

## Notes
- PoC scope: 6 training steps on 20 examples — validates the pipeline only
- For real use: scale to thousands of examples, replace `max_steps` with `num_train_epochs`
- Adapter weights saved to `lora_model/` (not tracked — push to Hugging Face Hub to share)
