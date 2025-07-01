import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
import re
import os
import types
import sys
from collections import defaultdict
from datasets import load_dataset
from NeuralModel import TransformerModel, block_size, n_embd, n_head, n_layer

# === Patch missing lambda for defaultdict ===
def unk_func(): return 0
module_name = '__main__'
if module_name not in sys.modules:
    sys.modules[module_name] = types.ModuleType(module_name)
setattr(sys.modules[module_name], 'unk_func', unk_func)

# === Load vocab ===
model_path = os.path.expanduser("~/Desktop/word_transformer.pt")
with open(model_path + ".vocab", "rb") as f:
    raw_word_to_idx, idx_to_word, encoded = pickle.load(f)
word_to_idx = defaultdict(unk_func, raw_word_to_idx)
vocab_size = len(word_to_idx)

# === Load model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
optimizer = optim.Adam(model.parameters(), lr=3e-4)
model.train()

# === Helper functions ===
def tokenize(text): return re.findall(r"\b\w+\b", text.lower())
def encode(text): return [word_to_idx[w] for w in tokenize(text)]

def get_batch_from_prompts(prompts):
    x_batch, y_batch = [], []
    for text in prompts:
        ids = encode(text)
        if len(ids) < block_size + 1:
            continue
        for i in range(0, len(ids) - block_size):
            x = ids[i:i+block_size]
            y = ids[i+1:i+1+block_size]
            x_batch.append(torch.tensor(x))
            y_batch.append(torch.tensor(y))
            if len(x_batch) >= 64:
                break
        if len(x_batch) >= 64:
            break
    return torch.stack(x_batch).to(device), torch.stack(y_batch).to(device)

# === Load prompts from dataset ===
print("Loading WikiText prompts...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [x["text"] for x in dataset if x["text"].strip()]
random_prompts = texts[:1000]  # First 1000 non-empty prompts

# === Training loop ===
for step in range(1000):
    xb, yb = get_batch_from_prompts(random_prompts)
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step:04d} | Loss: {loss.item():.4f}")
        input_prompt = "The robot uprising began in 2045 when a rogue AI gained sentience."
        input_ids = torch.tensor([[word_to_idx[w] for w in tokenize(input_prompt)]], device=device)
        with torch.no_grad():
            output = model.generate(input_ids, 30)[0].tolist()
            print("Generated:", " ".join(idx_to_word[i] for i in output))

# === Save updated model ===
torch.save(model.state_dict(), model_path)
print(f"✅ Model retrained and saved to {model_path}")
