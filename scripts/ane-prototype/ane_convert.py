import sys, torch, numpy as np
import coremltools as ct
from transformers import AutoModel

B, S = int(sys.argv[1]), int(sys.argv[2])

class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask):
        h = self.m(input_ids=input_ids, attention_mask=attention_mask)[0]
        cls = h[:, 0, :]
        return torch.nn.functional.normalize(cls, dim=-1)

model = AutoModel.from_pretrained("nomic-ai/CodeRankEmbed", trust_remote_code=True, safe_serialization=True)
model.eval()
w = Wrapper(model).eval()

ids = torch.randint(0, 30000, (B, S), dtype=torch.int32)
mask = torch.ones((B, S), dtype=torch.int32)
with torch.no_grad():
    w(ids, mask)  # warm rope caches
    traced = torch.jit.trace(w, (ids, mask), check_trace=False)

mlm = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(B, S), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(B, S), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="embedding")],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS14,
    convert_to="mlprogram",
)
path = f"coderank_b{B}_s{S}.mlpackage"
mlm.save(path)
print("saved", path)
