import glob, numpy as np, torch
import coremltools as ct
from transformers import AutoModel, AutoTokenizer

B, S = 8, 256
texts = []
for f in sorted(glob.glob('/Users/jason/gitProjects/search_test/src/*.rs'))[:8]:
    texts.append(open(f).read()[:2000])

tok = AutoTokenizer.from_pretrained("nomic-ai/CodeRankEmbed", trust_remote_code=True)
enc = tok(texts, return_tensors="pt", padding="max_length", max_length=S, truncation=True)
ids32 = enc["input_ids"].to(torch.int32)
mask32 = enc["attention_mask"].to(torch.int32)

model = AutoModel.from_pretrained("nomic-ai/CodeRankEmbed", trust_remote_code=True, safe_serialization=True).eval()
with torch.no_grad():
    h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])[0]
ref = torch.nn.functional.normalize(h[:, 0, :], dim=-1).numpy()

m = ct.models.MLModel(f"coderank_b{B}_s{S}.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
out = m.predict({"input_ids": ids32.numpy(), "attention_mask": mask32.numpy()})["embedding"]

cos = (ref * out).sum(axis=1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(out, axis=1))
print("per-chunk cosine vs fp32 torch reference:")
print(np.array2string(cos, precision=5))
print(f"min {cos.min():.5f}  mean {cos.mean():.5f}")
