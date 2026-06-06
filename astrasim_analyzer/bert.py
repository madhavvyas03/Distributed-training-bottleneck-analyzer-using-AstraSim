import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import os

os.makedirs("traces", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Dataset (SST-2) =====
dataset = load_dataset("glue", "sst2")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=64)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=8, shuffle=True)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

model.train()
steps = 100
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler("./traces/bert_trace"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for i, batch in enumerate(train_loader):
        if i >= steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        prof.step()

prof.export_chrome_trace("./traces/bert_trace.json")
torch.save(prof.key_averages(), "./traces/bert_trace.pt")
print("✅ BERT profiling complete. Traces saved in ./traces/")
