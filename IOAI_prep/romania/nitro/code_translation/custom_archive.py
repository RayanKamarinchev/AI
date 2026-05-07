import argparse
import csv

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 256


def load_test_csv(path):
    queries = []
    candidates = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == "query":
                queries.append({"id": row["datapointID"], "source": row["source"]})
            elif row["type"] == "candidate":
                candidates.append({"id": row["datapointID"], "source": row["source"]})
    return queries, candidates


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


def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference for code retrieval"
    )
    parser.add_argument("--test-csv", default="test.csv", help="Path to test.csv")
    parser.add_argument("--output", default="sample_submission.csv", help="Output path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional projection checkpoint from baseline/train.py",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME).to(device)
    bert.eval()

    py_proj = None
    c_proj = None
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        from baseline.train import ProjectionHead

        py_proj = ProjectionHead(ckpt["embed_dim"], ckpt["proj_dim"]).to(device)
        c_proj = ProjectionHead(ckpt["embed_dim"], ckpt["proj_dim"]).to(device)
        py_proj.load_state_dict(ckpt["py_proj"])
        c_proj.load_state_dict(ckpt["c_proj"])
        py_proj.eval()
        c_proj.eval()

    print(f"Loading test data from {args.test_csv}...")
    queries, candidates = load_test_csv(args.test_csv)
    print(f"Queries: {len(queries)}, Candidates: {len(candidates)}")

    query_texts = [q["source"] for q in queries]
    query_ids = [q["id"] for q in queries]
    cand_texts = [c["source"] for c in candidates]
    cand_ids = [c["id"] for c in candidates]

    print("Encoding queries (Python)...")
    query_embs = encode_batch(bert, tokenizer, query_texts, device, args.batch_size)

    print("Encoding candidates (C++)...")
    cand_embs = encode_batch(bert, tokenizer, cand_texts, device, args.batch_size)

    if py_proj and c_proj:
        print("Applying projection heads...")
        with torch.no_grad():
            query_embs = py_proj(query_embs.to(device)).cpu()
            cand_embs = c_proj(cand_embs.to(device)).cpu()

    query_embs = F.normalize(query_embs, p=2, dim=-1)
    cand_embs = F.normalize(cand_embs, p=2, dim=-1)

    print("Ranking candidates...")
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["subtaskID", "datapointID", "answer"])
        for i, qid in enumerate(tqdm(query_ids, desc="Scoring")):
            sims = torch.matmul(cand_embs, query_embs[i])
            ranked_indices = torch.argsort(sims, descending=True).tolist()
            ranked_ids = [cand_ids[j] for j in ranked_indices]
            writer.writerow([1, qid, ";".join(ranked_ids)])

    print(f"Submission saved to {args.output}")


if __name__ == "__main__":
    main()
