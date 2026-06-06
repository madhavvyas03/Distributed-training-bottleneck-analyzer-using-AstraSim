import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from datasets import load_dataset
import os

os.makedirs("traces", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("criteo_1tb", "day_0", split="train[:1%]")  
dense_cols = [f"I{i}" for i in range(1, 14)]
cat_cols = [f"C{i}" for i in range(1, 27)]

def preprocess(batch):
    dense = torch.tensor([[float(v) if v != '' else 0.0 for v in batch[c]] for c in dense_cols]).T
    cat = torch.tensor([[hash(v) % 10000 for v in batch[c]] for c in cat_cols]).T
    labels = torch.tensor(batch["label"]).float().unsqueeze(1)
    return {"dense": dense, "cat": cat, "label": labels}

dataset = dataset.map(preprocess, batched=True, batch_size=256)
dataset.set_format("torch", columns=["dense", "cat", "label"])
train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

class DLRMLike(nn.Module):
    def __init__(self, embed_dim=16, vocab_size=10000, num_dense=13, num_sparse=26):
        super().__init__()
        self.bottom_mlp = nn.Sequential(
            nn.Linear(num_dense, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for _ in range(num_sparse)])
        self.top_mlp = nn.Sequential(
            nn.Linear(embed_dim * (num_sparse + 1), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, dense_x, cat_x):
        dense_out = self.bottom_mlp(dense_x)
        sparse_outs = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        concat = torch.cat([dense_out] + sparse_outs, dim=1)
        return self.top_mlp(concat)

model = DLRMLike().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
steps = 100
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler("./traces/dlrm_trace"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for i, batch in enumerate(train_loader):
        if i >= steps:
            break
        dense_x = batch["dense"].to(device)
        cat_x = batch["cat"].to(device)
        target = batch["label"].to(device)
        optimizer.zero_grad()
        output = model(dense_x, cat_x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        prof.step()

prof.export_chrome_trace("./traces/dlrm_trace.json")
torch.save(prof.key_averages(), "./traces/dlrm_trace.pt")
print("✅ DLRM profiling complete. Traces saved in ./traces/")
