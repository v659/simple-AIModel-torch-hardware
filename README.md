
# 🧠 Word Transformer Language Model

This project trains a word-level Transformer on the [WikiText-2 raw dataset](https://huggingface.co/datasets/wikitext) using PyTorch. It supports continued training, generation with top-k/top-p sampling, repetition penalties, and CPU optimization.

---

## 🚀 Features

- Word-level tokenization from raw WikiText-2
- Transformer Encoder model with positional embeddings
- Sampling via `top_k`, `top_p` (nucleus), and `repetition_penalty`
- Autocast mixed precision (`bfloat16` on CPU)
- `torch.compile` optimization (PyTorch ≥ 2.0)
- Automatically saves best model checkpoint
- Continues training from last checkpoint
- CLI-ready for both training and generation

---

## 🧾 Requirements

- Python 3.9+
- PyTorch ≥ 1.13 (for `torch.compile`, recommend 2.0+)
- `datasets` from HuggingFace

Install with:

```bash
pip install torch datasets
```

---

## 🧠 Model Architecture

- 4 TransformerEncoder layers
- Embedding size: 128
- Heads: 4
- Feedforward hidden size: 512
- Block size (context window): 32
- Trained on full wikiText2 dataset

---

## 🧪 Usage

### 🔧 Train the model

```bash
python NeuralModel.py
```

By default, it:
- Loads `wikitext-2-raw-v1` from Hugging Face
- Builds a word vocabulary (if not cached)
- Trains for 20000 steps (configurable)

If a model checkpoint exists at `~/Desktop/word_transformer.pt`, training resumes from there.

---

### ✨ Generate from a prompt

You can modify the bottom of `NeuralModel.py` or create a script to load and generate:

```python
from NeuralModel import load_and_generate

prompt = "the secrets of the universe lie hidden"
load_and_generate(prompt)
```

---

## 🧠 Sampling Controls

You can tweak the following inside `generate()`:

```python
model.generate(prompt_ids, max_new_tokens=50,
               temperature=0.8,
               top_k=50,
               top_p=0.9,
               repetition_penalty=1.2)
```

---

## 📂 Saved Files

- `word_transformer.pt` — model weights
- `word_transformer.pt.vocab` — vocabulary + encoded data

---

## 🧠 Example Output

```text
Step 0000 | Loss: 11.1250
Generated: deep in the himalayas an explorer uncovers an ancient temple containing artifacts linked to the lost city of atlantis ...
```
---

## 📝 License

MIT License. Attribution appreciated but not required.
