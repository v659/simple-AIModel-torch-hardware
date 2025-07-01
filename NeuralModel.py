import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from datasets import load_dataset
from collections import defaultdict
import re
import html
import os
import pickle
import multiprocessing
torch.set_num_threads(multiprocessing.cpu_count())  # Use all CPU cores
try:
    torch.set_float32_matmul_precision('high')  # PyTorch ≥ 2.0
except:
    pass


# === Config ===
block_size = 32
batch_size = 64
n_embd = 128
n_head = 4
n_layer = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.expanduser("~/Desktop/word_transformer.pt")

# === Tokenizer ===
def clean(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    return text.lower()

def tokenize(text):
    return re.findall(r"\b\w+\b", text)

# === Load Dataset and Vocabulary ===
print("Loading WikiText...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = " ".join(clean(x["text"]) for x in dataset if x["text"]).strip()
tokens = tokenize(text)

# === Load or Build Vocabulary ===
def unk_func():
    return 0

if os.path.exists(model_path + ".vocab"):
    with open(model_path + ".vocab", "rb") as f:
        word_to_idx_raw, idx_to_word, encoded = pickle.load(f)
    word_to_idx = defaultdict(unk_func, word_to_idx_raw)
    print("🔁 Loaded existing vocabulary.")
else:
    print("🧱 Building vocabulary...")
    word_to_idx = defaultdict(unk_func)
    word_to_idx["<unk>"] = 0
    idx_to_word = {0: "<unk>"}
    encoded = []
    next_idx = 1
    for word in tokens:
        if word not in word_to_idx:
            word_to_idx[word] = next_idx
            idx_to_word[next_idx] = word
            next_idx += 1
        encoded.append(word_to_idx[word])
    with open(model_path + ".vocab", "wb") as f:
        # Save as plain dict, not defaultdict
        pickle.dump((dict(word_to_idx), idx_to_word, encoded), f)
    print("✅ Saved vocabulary.")

vocab_size = len(word_to_idx)


# === Data loader ===
def get_batch():
    ix = torch.randint(0, len(encoded) - block_size - 1, (batch_size,))
    x = torch.stack([torch.tensor(encoded[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(encoded[i+1:i+1+block_size]) for i in ix])
    return x.to(device), y.to(device)

# === Model ===
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head,
                dim_feedforward=4*n_embd,
                batch_first=True,
                dropout=0.1
            ) for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=device)).unsqueeze(0)
        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., -1, None]] = -float('Inf')
        return out

    @staticmethod
    def top_p_filtering(logits, top_p=0.9, filter_value=-float("Inf")):
        """Nucleus (top-p) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_mask = cumulative_probs > top_p
        # Shift mask right to always include the first token above the threshold
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_mask]
        logits[indices_to_remove] = filter_value
        return logits

    @torch.no_grad()
    def generate(self, x, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.2):
        generated = x.clone()
        for _ in range(max_new_tokens):
            x_cond = generated[:, -block_size:] if generated.size(1) > block_size else generated
            logits = self(x_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab)

            for b in range(generated.size(0)):
                # Apply repetition penalty
                for token_id in generated[b].tolist():
                    logits[b, token_id] /= repetition_penalty

                # Apply top-k if requested
                if top_k is not None:
                    v, _ = torch.topk(logits[b], top_k)
                    logits[b][logits[b] < v[-1]] = -float("Inf")

                # Apply top-p if requested
                if top_p is not None:
                    logits[b] = self.top_p_filtering(logits[b], top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


model = TransformerModel().to(device)
try:
    model = torch.compile(model)  # PyTorch 2.x optimization
except:
    pass

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)


def decode(ids): return " ".join(idx_to_word.get(i, "<unk>") for i in ids)

def train(steps=2000):
    model.train()
    best_loss = float('inf')  # ✅ Initialize once, outside the loop

    for step in range(steps):
        xb, yb = get_batch()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16, enabled=True):
            logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1 == 0:
            while True:
                prompt = clean(dataset[random.randint(0, len(dataset) - 1)]["text"])
                tokens = tokenize(prompt)
                if tokens:
                    break  # found a non-empty prompt

            prompt_ids = torch.tensor([[word_to_idx.get(w, word_to_idx["<unk>"]) for w in tokens]],
                                      device=device, dtype=torch.long)
            out = model.generate(prompt_ids, 30, temperature=0.8)[0].tolist()
            print("Generated:", decode(out))


        # ✅ Save only if current loss is better than previous best
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), model_path)
            print(f"📉 New best loss: {best_loss:.4f} — model saved!")
            while True:
                prompt = clean(dataset[random.randint(0, len(dataset) - 1)]["text"])
                tokens = tokenize(prompt)
                if tokens:
                    break  # found a non-empty prompt

            prompt_ids = torch.tensor([[word_to_idx.get(w, word_to_idx["<unk>"]) for w in tokens]],
                                      device=device, dtype=torch.long)
            out = model.generate(prompt_ids, 30, temperature=0.8)[0].tolist()
            print("Generated:", decode(out))


    print(f"✅ Training done. Best loss: {best_loss:.4f}")


def load_and_generate(prompt):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        prompt_words = tokenize(prompt)
        ids = torch.tensor([[word_to_idx.get(w, 0) for w in prompt_words]], device=device)
        if ids.numel() == 0:
            print("⚠️ No valid tokens in prompt")
            return
        out = model.generate(ids, 30)[0].tolist()
        print("Generated:", decode(out))


# === Entry point ===
if __name__ == "__main__":
    if os.path.exists(model_path):
        print("🔁 Training from existing model")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("🆕 Training from scratch")

    train(steps=20000)

