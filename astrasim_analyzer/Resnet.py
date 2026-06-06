import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import os

os.makedirs("traces", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

model = torchvision.models.resnet50(weights=None, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model.train()
steps = 150 
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler("./traces/resnet50_trace"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for i, (inputs, labels) in enumerate(trainloader):
        if i >= steps:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prof.step()

prof.export_chrome_trace("./traces/resnet50_trace.json")
torch.save(prof.key_averages(), "./traces/resnet50_trace.pt")
print("✅ ResNet-50 profiling complete. Traces saved in ./traces/")
