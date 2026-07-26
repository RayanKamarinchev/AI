# Plots

-  sns.pairplot
- sns.heatmap(corr, xticklabels=ids, yticklabels=ids)
- 
```
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(max_depth=3)
tree.fit(x, y)

plot_tree(tree)
plt.show()
```

### Decision bounary
```
model = SVC(kernel='rbf', C=1.0, gamma="scale")
model.fit(x, y)

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min()-1, x[:, 0].max()+1, 300),
    np.linspace(x[:, 1].min()-1, x[:, 1].max()+1, 300)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors="k")
plt.show()
```
# Feature

- SimpleImputer(strategy="median")
- dorp null > 0.8, binary mask of isnull if >0.2, else imputer
- xgb handles nulls
- [[multispectral_segmentation/sol.ipynb]]
- 
```
from sklearn.feature_selection import mutual_info_classif
import numpy as np

imgs  = torch.stack([img  for img, _    in raw_train]).numpy()   # [N, 12, 30, 30]
masks = torch.stack([mask for _,   mask in raw_train]).numpy()   # [N, 30, 30]

X = imgs.transpose(0, 2, 3, 1).reshape(-1, 12)   # [N*900, 12]
y = masks.flatten()                                # [N*900]

mi = mutual_info_classif(X, y, discrete_features=False)
for i, score in enumerate(mi):
    print(f"Channel {i:2d}:  MI = {score:.4f}")
```

```
from itertools import combinations

def nd(a, b, eps=1e-8):
    d = a + b
    d = np.where(np.abs(d) < eps, eps, d)
    return (a - b) / d

# Build all 66 normalised-difference features at pixel level
nd_features = {}
for i, j in combinations(range(12), 2):
    key = f"ND({i},{j})"
    nd_features[key] = nd(X[:, i], X[:, j])

ND_matrix = np.column_stack(list(nd_features.values()))  # [N*900, 66]

# Score each one by mutual information with labels
mi_nd = mutual_info_classif(ND_matrix, y, discrete_features=False)

# Top 3
top_idx = np.argsort(mi_nd)[::-1][:3]
for rank, idx in enumerate(top_idx):
    key = list(nd_features.keys())[idx]
    print(f"Rank {rank+1}: {key}  MI={mi_nd[idx]:.4f}")
```

### Selection
- Random forest
- PCA
- Mutual Information
- recursive feature elimination RFE
# Optimization
- AdamW for the win
- optuna xgboost for ml
- initialization is VERY important, start with mean
- add penalties for too big parameters or model cheats that come to mind
- [[shadows/main.ipynb]]

```
from tqdm.notebook import tqdm
def compute_kernel() -> torch.Tensor:
    kernel = nn.Parameter(torch.ones(9, 9, device=DEVICE, requires_grad=True))

    kernel = kernel.to(DEVICE)
    optim = torch.optim.Adam([kernel], lr=5e-5)
    criterion = nn.MSELoss()

    num_epochs = 500

    for epoch in range(num_epochs):
        train_loss, val_loss = 0.0, 0.0
        
        for sample in train_ds:
            original = sample['image_original'].to(DEVICE)
            corrupted = sample['image_corrupted'].to(DEVICE)

            reconstructed = apply_kernel(corrupted, kernel)
            optim.zero_grad()
            loss = criterion(reconstructed, original)
            loss.backward()
            optim.step()

            train_loss += loss.item()

        with torch.no_grad():
            for sample in val_ds:
                original = sample['image_original'].to(DEVICE)
                corrupted = sample['image_corrupted'].to(DEVICE)

                reconstructed = apply_kernel(corrupted, kernel)
                mse = torch.mean((reconstructed - original)**2)

                val_loss += mse.item()

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss/len(train_ds):.6f} | "
            f"val_loss={val_loss/len(val_ds):.6f}"
        )

    return kernel
```

[[autocorrect/main.ipynb]]
```
if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # weight by real (non-pad) tokens so the epoch average matches the loss definition
            mask = labels != pad_id
            n = mask.sum().item()
            total_loss += loss.item() * n
            total_tokens += n
            correct += ((outputs.argmax(-1) == labels) & mask).sum().item()
```
# Image

### Augmentations

```
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.25),
        ratio=(0.3, 3.3),
        value=0
    ),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.7, 1.0)
    ),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Losses

#### Focal
```
def focal_loss_multiclass(logits, targets, gamma=2.0):  
	ce = F.cross_entropy(logits, targets, reduction="none")  
	return ((1 - torch.exp(-ce)).pow(gamma) * ce).mean()
```

#### Dice

```
def _dice_loss(logits: torch.Tensor, targets: torch.Tensor,
               num_classes: int = 4, smooth: float = 1.0) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    oh    = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * oh).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()
```

#### Structural Similarity Index Measure (classification)
```
from torchmetrics.image import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
```

### Classification

```
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = channels // reduction
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand=2, use_se=True):
        super().__init__()

        hidden = in_ch * expand

        self.use_res = (stride == 1 and in_ch == out_ch)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            SEBlock(hidden) if use_se else nn.Identity(),

            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_res else out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(
            MBConv(32, 32, stride=1, expand=2),
            MBConv(32, 32, stride=1, expand=2),

            MBConv(32, 64, stride=2, expand=2),
            MBConv(64, 64, stride=1, expand=2),

            MBConv(64, 128, stride=2, expand=2),
            MBConv(128, 128, stride=1, expand=2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
```


```
nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),

    nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),

    nn.MaxPool2d(2),

    nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
    nn.Conv2d(32, 64, kernel_size=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.MaxPool2d(2),

    nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=32, bias=False),
    nn.Conv2d(64, 128, kernel_size=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
)
```
### Segmentation

-YOLO-less object detection [[obj_detection/main.ipynb]]
#### YOLO
[[ghost_object_detection/main.ipynb]]

#### UNet

```
class _DConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
 
 
class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = _DConv(3,    64)
        self.enc2 = _DConv(64,  128)
        # Bottleneck
        self.bot  = _DConv(128, 256)
        # Decoder
        self.dec2 = _DConv(256 + 128, 128)
        self.dec1 = _DConv(128 +  64,  64)
        # Pooling & head
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(64, 4, kernel_size=1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                                       # [B,  64, 30, 30]
        e2 = self.enc2(self.pool(e1))                           # [B, 128, 15, 15]
        b  = self.bot(self.pool(e2))                            # [B, 256,  7,  7]
 
        d2 = F.interpolate(b,  size=e2.shape[2:],
                           mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))             # [B, 128, 15, 15]
 
        d1 = F.interpolate(d2, size=e1.shape[2:],
                           mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))             # [B,  64, 30, 30]
 
        return self.head(d1)                                    # [B,   4, 30, 30]

```


# Text

### Data

#### Preprocess

```
def preprocess(df, word2idx=None, label2idx=None, max_len=500, tokenize_func=None):
    texts = df["text"].values
    labels = df["category"].values if "category" in df.columns else None #Check if there is a label, if not, fill in "None".

    # Label encoding.
    if label2idx is None and labels is not None:
        unique_labels = set(labels)
        label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    if labels is not None:
        labels = [label2idx[label] for label in labels]

    # Tokenization.
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]

    if word2idx is None:
        # Build a vocabulary list.
        all_tokens = [token for text in tokenized_texts for token in text]
        vocab = Counter(all_tokens)
        print(len(all_tokens))
        vocab_size = 25000  #The total will not exceed 25,000 words.
        vocab = vocab.most_common(vocab_size - 2)
        word2idx = {word: idx + 2 for idx, (word, _) in enumerate(vocab)}
        word2idx["<unk>"] = 0
        word2idx["<pad>"] = 1

    # Numericalization
    def encode_text(text):
        return [word2idx.get(word, word2idx["<unk>"]) for word in text]

    encoded_texts = [encode_text(text) for text in tokenized_texts]

    # Fill or truncate.
    padded_texts = [
        (
            text[:max_len]
            if len(text) > max_len
            else text + [word2idx["<pad>"]] * (max_len - len(text))
        )
        for text in encoded_texts
    ]

    return padded_texts, labels, word2idx, label2idx
```
[[apoai/news_classification/nlp_classification.ipynb]]
#### Collators
```
from transformers import DataCollatorForSeq2Seq
from transformers import DataCollatorWithPadding

collator = DataCollatorForSeq2Seq(tokenizer, model)
collator = DataCollatorWithPadding(tokenizer)
```
```
#LSTM:
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence 

def collate_fn(batch):
    x, y_a, y_c = zip(*batch)
    lens = torch.tensor([len(one_x) for one_x in x])
    x_pad = pad_sequence(x, batch_first=True)
    return x_pad, torch.stack(y_a), torch.stack(y_c), lens
    
#Trasformers:
from torch.nn.utils.rnn import pad_sequence  
  
def collate_fn(batch):  
	images, token_ids = zip(*batch)  
	image_batch = torch.stack(images)  
	  
	token_id_batch = pad_sequence(token_ids, batch_first=True, padding_value=0)  
  
	attention_mask_batch = (token_id_batch != 0).long()  
  
	return image_batch, token_id_batch, attention_mask_batch
```

#### Embeddings feature engineering

[[semantic_changes.ipynb]]

```
def get_features(word):
    i0 = w2i_1900[word]
    i9 = w2i_1990[word]
    diff = W1900[i0] - W1990[i9]
    prod = W1900[i0] * W1990[i9]

    sim = np.dot(W1900[i0], W1990[i9])
    mse = np.mean(np.pow(diff, 2))
    mean_diff = np.mean(diff)
    abs_dist = np.mean(np.abs(diff))
    return sim, mse, mean_diff, abs_dist, np.std(diff), np.max(abs_dist), np.mean(prod), np.std(prod)
```

#### TfIdf

```
TfidfVectorizer(analyzer='char', ngram_range=(2,5)) #character patterns
TfidfVectorizer(analyzer='word' ngram_range=(1, 2), max_features=20000, sublinear_tf=True, stop_words='english')

# since tf-idf output is sparse
TruncatedSVD(128)
```
### Models

#### LSTM

Full code: [[pose_classification/main.ipynb]]
```
# Define a bi-directional LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional = True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits
```

```
class LSTMAutocorrect(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, lens):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (_, _) = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        x = self.fc(x)
        return x
```

#### GRU

[[autocorrect/main.ipynb]]
```
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embed_size,  hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.bigru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, vocab_size)

    def forward(self, x, lens=None, hidden=None):
        x = self.embedding(x)
        if lens is None:
            out, hidden = self.bigru(x, hidden)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
            out, hidden = self.bigru(packed, hidden)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = self.fc(out)
        return out

```

#### LLMs
- custom sampling [[hungary/2026/llm/main.ipynb]]
- Whole GPT2 architecture + beam search [[3_2/llm/main.ipynb]]
- Qwen/Qwen3-Embedding-0.6B

```
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

inputs = processor(images=img, text=options, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        indexes = outputs['logits_per_image'].squeeze(0).argsort(descending = True)[:5]
```

#### Decoder
```
def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
```
#### Encoder
```
def mean_pool(self, last_hidden_states, attention_mask):
    # Expand mask to match hidden state dimensions [batch, seq_len, hidden_dim]
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    
    # Zero out padding token vectors, sum real tokens
    sum_embeddings = torch.sum(last_hidden_states * mask, dim=1)
    
    # Count real tokens per sequence, clamp avoids division by zero
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    
    return sum_embeddings / sum_mask  
```
### Finetune
[[finetune_hf.ipynb]]
[[finetune_torch.ipynb]]
Contrastive: [[roai_perspective/contrastive_learning.py]]

#### Encoding and batching
```
def encode_batch(model, tokenizer, texts, device, batch_size=32):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
        last_hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float().to(device)
        embs = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME).to(device)
bert.eval()

query_embs = encode_batch(bert, tokenizer, train['py_source'].tolist(), device)
cand_embs = encode_batch(bert, tokenizer, train['cpp_source'].tolist(), device)
```

#### Early Stopping
```

trainer = Trainer(model, args, train_dataset=train_tokenized, eval_dataset=test_tokenized,
                   callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
                   data_collator=data_collator)
                   
```

#### LoRA
```
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,   
    r=32,                               # rank (lower = fewer parameters)
    lora_alpha=32,                     # scaling factor
    target_modules=["q", "v"],         # which modules to adapt (Q and V in self-attention)
    lora_dropout=0.1,                  # dropout for LoRA layers
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```
#### Collation

```
correct = {(row['item1'], row['item2']): row['result'] for row in train_ds}

def collate(batch):
    a  = [x['item1']  for x in batch]
    b  = [x['item2']  for x in batch]
    c  = [x['result'] for x in batch]
    n  = len(batch)
    triples = []
    for i in range(n):
        j, k = random.randrange(n), random.randrange(n)
        triples += [(a[i], b[i], c[i]),      # positive
                    (a[i], b[j], c[i]),      # wrong second ingredient
                    (a[i], b[i], c[k])]      # wrong result
    texts  = [f'Combining {x} and {y} creates {z}' for x, y, z in triples]
    labels = torch.tensor([correct.get((x, y)) == z for x, y, z in triples]).long()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=48, return_tensors='pt')
    enc['labels'] = labels
    return enc
```
#### Triplet loss
[[sequence_ordering/main.ipynb]]
#### Contrastive loss
``` 
def loss_fn(image_emb, text_emb, logit_scale):  
	image_emb = F.normalize(image_emb, dim=-1)  
	text_emb = F.normalize(text_emb, dim=-1)  
	  
	logits = image_emb @ text_emb.T * logit_scale.exp()  
	  
	targets = torch.arange(image_emb.size(0), device=image_emb.device)  
	  
	loss_i2t = F.cross_entropy(logits, targets)  
	loss_t2i = F.cross_entropy(logits.T, targets)  
	  
	return (loss_i2t + loss_t2i) / 2
```

# ML

### Unsupervised
- Adjusted Random Index ARI

- Awesome solution: [[ml_gold_standard/main.ipynb]]
### PCA + GaussianMixture
```
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

pca = PCA(0.9)
x_pca = pca.fit_transform(features)
x_inv = pca.inverse_transform(x_pca)
errs = np.linalg.norm(features-x_inv, axis=1)
gmm = GaussianMixture(n_components=2, random_state=0)
labels = gmm.fit_predict(errs.reshape(-1, 1))


for i in [0,1]:
    plt.hist(errs[labels==i], density=True, bins=20)
```
# Time series
[[pendulum.ipynb]]

```
nn.Sequential(
    nn.Conv1d(2, 32, kernel_size=7, padding=3, stride=2, bias=False),
    nn.BatchNorm1d(32),
    nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2, bias=False),
    nn.BatchNorm1d(64),
    nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
    nn.BatchNorm1d(128),

    nn.AdaptiveAvgPool1d(1),
    nn.Flatten()
)
```
# Audio
### Data

```
librosa.display.Audio(sample_path)

y, sample_rate = librosa.load(sample_path)

librosa.display.waveshow(y=y, sr=sample_rate)


D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
DB = librosa.amplitude_to_db(D, ref = np.max)
librosa.display.specshow(DB, sr=sample_rate, hop_length=512,
                         x_axis='time', y_axis='log')



S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64) #(y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=SR,
						x_axis="time", y_axis="mel", cmap="magma")
S_DB = librosa.amplitude_to_db(S, ref=np.max)



db = librosa.power_to_db(mel, ref=np.max)
librosa.display.specshow(db, sr=SR, x_axis='time', y_axis='hz')
plt.colorbar()


Padding:
x_train = [np.pad(entry, pad_width=((0,0), (0, 53-entry.shape[1])), mode='constant') for entry in x_train]
```
### Models
#### CRNN
```
class AudioCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)              # (B, C, T', F')
        x = x.mean(dim=-1)          # collapse frequency → (B, C, T')
        x = x.permute(0, 2, 1)      # (B, T', C)
        _, h = self.gru(x)
        return self.fc(h[-1])
```


# Unique
### Adversarial attacks / Steering
- [[emotions.ipynb]]
- Steering images [[ru-hogspell-baseline-solution.ipynb]]
- Tricking AI detector [[ai_text_detection_bypass/solution.ipynb]]
### Anomaly detection
- [[anomally_detection/main.ipynb]]
### Explainability
- [[credit/main.ipynb]]
### Unlearning
[[unlearning/main.ipynb]]

### Self-supervised
- [[self_supervised.ipynb]]
### VAE Implemented
- [[problem4.ipynb]]
### Clip segmentation
[[07_segmentation/solution.ipynb]]

### 2 models for noisy data
[[noisy_dt/main.ipynb]]
### RL
- [[problem2.ipynb]]
- [[rl_implementation/main.ipynb]]
- [[stochastic_rift_baseline.ipynb]]

### Completely separate but interesting
https://aicc-official.org/solutions/round-4/extreme-condensation
https://github.com/AI-Community-Contest/solutions/blob/main/round-3/drawn-apart.ipynb
https://github.com/AI-Community-Contest/solutions/blob/main/round-5/watermark-removal.ipynb

# Math
### SVD compression
```
tries = 500000
best_error = np.inf
best_ids = []
n = 420

for i in range(tries):
    points = x[np.random.choice(n, 4)]
    mean = np.mean(points, axis=0)
    basis = points - mean

    U, S, V = np.linalg.svd(basis, full_matrices=False)
    basis = V[:3]
    x_norm = x - mean
    proj = (x_norm @ basis.T) @ basis
    dists = np.linalg.norm(x_norm - proj, axis=1)
    ids = np.argpartition(dists, 20)[:20]
    err = np.sum(dists[ids]**2)
    if err < best_error:
        print(f"Found better with loss {err:.2f}")
        best_error = err
        best_ids = ids

best_ids
```
### Reconstructing the features from a linear model
[[ds_reconstruction/starter_kit.ipynb]]
```
n = len(params["scaler_mean"])
s = np.zeros((n,n))
np.fill_diagonal(s, 1.0/np.array(params["scaler_scale"]))
w = np.array(params['coef'])

w_norm = w.T @ s
w_norm = w_norm.reshape(1, -1)
b_norm = np.array(params['intercept'] - w_norm @ np.array(params["scaler_mean"]))

lamda = (y - (w_norm @ x.T + b_norm)) / np.dot(w_norm, w_norm.T)
real_x = x + lamda.T @ w_norm

errs = pipe.predict(real_x)-y
```