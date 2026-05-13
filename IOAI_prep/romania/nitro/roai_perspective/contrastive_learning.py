import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel
from tqdm import tqdm

# ============================================================
# Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 20
LR = 3e-4
TEMPERATURE = 0.5
IMAGE_SIZE = 224
EMBED_DIM = 256

MODEL_NAME = "google/vit-base-patch16-224"

# ============================================================
# Data Augmentation
# ============================================================

class ContrastiveTransform:
    """
    Creates two augmented views of the same image.
    """

    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# ============================================================
# Dataset
# ============================================================

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=ContrastiveTransform(size=IMAGE_SIZE)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

# ============================================================
# Projection Head
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, output_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# Contrastive ViT Model
# ============================================================

class ContrastiveViT(nn.Module):
    def __init__(self, model_name, projection_dim):
        super().__init__()

        self.encoder = ViTModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.projector = ProjectionHead(
            input_dim=hidden_size,
            output_dim=projection_dim
        )

    def forward(self, x):

        outputs = self.encoder(pixel_values=x)

        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0]

        projections = self.projector(cls_embedding)

        projections = F.normalize(projections, dim=1)

        return projections

# ============================================================
# NT-Xent Loss (InfoNCE)
# ============================================================

def nt_xent_loss(z1, z2, temperature=0.5):

    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)

    similarity = torch.matmul(z, z.T)

    mask = torch.eye(2 * batch_size, device=z.device).bool()

    similarity = similarity / temperature

    similarity.masked_fill_(mask, -1e9)

    positives = torch.cat([
        torch.diag(similarity, batch_size),
        torch.diag(similarity, -batch_size)
    ], dim=0)

    numerator = torch.exp(positives)

    denominator = torch.exp(similarity).sum(dim=1)

    loss = -torch.log(numerator / denominator)

    return loss.mean()

# ============================================================
# Initialize Model
# ============================================================

model = ContrastiveViT(
    model_name=MODEL_NAME,
    projection_dim=EMBED_DIM
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

# ============================================================
# Training Loop
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    progress = tqdm(loader)

    for (x1, x2), _ in progress:

        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)

        z1 = model(x1)
        z2 = model(x2)

        loss = nt_xent_loss(
            z1,
            z2,
            temperature=TEMPERATURE
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress.set_description(
            f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.4f}"
        )

    avg_loss = total_loss / len(loader)

    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

# ============================================================
# Save Model
# ============================================================

torch.save(model.state_dict(), "contrastive_vit.pth")

print("Model saved to contrastive_vit.pth")

# ============================================================
# Example: Extract Embeddings
# ============================================================

model.eval()

sample_loader = DataLoader(dataset, batch_size=8)

(x1, _), _ = next(iter(sample_loader))

x1 = x1.to(DEVICE)

with torch.no_grad():
    embeddings = model(x1)

print("Embeddings shape:", embeddings.shape)