import copy
import sys, torch, numpy as np
import coremltools as ct
from transformers import AutoModel

B, S = int(sys.argv[1]), int(sys.argv[2])
if B < 1 or S < 2:
    raise SystemExit("batch must be positive and sequence length at least 2")
torch.manual_seed(0)
torch.set_num_threads(4)

class SeparateValueAttention(torch.nn.Module):
    """Avoid the packed QKV value view in short, batched ANE graphs."""
    def __init__(self, attention):
        super().__init__()
        if attention.num_heads_kv != attention.num_heads or attention.rotary_head_dim:
            raise ValueError("separate-value export requires CodeRankEmbed's standard attention layout")
        self.a = attention

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        a = self.a
        batch, seq, _ = hidden_states.shape
        qkv = a.Wqkv(hidden_states).reshape(batch, seq, 3, a.num_heads, a.head_dim)
        qkv = a.rotary_emb(qkv, seqlen_offset=0)
        query = qkv[:, :, 0].permute(0, 2, 1, 3)
        key = qkv[:, :, 1].permute(0, 2, 1, 3)
        # On the M3 CPU_AND_NE path, the packed value view corrupts rows
        # 2/3/6/7 at batch=8, seq=64. An equivalent independent projection
        # avoids that path; Q/K retain the original rotary implementation.
        offset = 2 * a.embed_dim
        value = torch.nn.functional.linear(
            hidden_states, a.Wqkv.weight[offset:],
            None if a.Wqkv.bias is None else a.Wqkv.bias[offset:],
        ).reshape(batch, seq, a.num_heads, a.head_dim).permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-1, -2)) / a.norm_factor
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = a.drop(torch.softmax(scores, dim=-1))
        context = torch.matmul(probabilities, value)
        return a.out_proj(context.permute(0, 2, 1, 3).reshape(batch, seq, a.embed_dim))

class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask):
        # The generic HF forward multiplies (1 - mask) by -FLT_MAX. During
        # FP16 conversion that constant becomes -inf, making unmasked entries
        # 0 * -inf = NaN. Some CoreML shapes optimize it away, others produce
        # invalid embeddings. Keep the additive mask finite in FP16 instead.
        h = self.m.embeddings(input_ids, token_type_ids=torch.zeros_like(input_ids))
        h = self.m.emb_drop(self.m.emb_ln(h))
        bias = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        h = self.m.encoder(h, attention_mask=bias)
        cls = h[:, 0, :]
        return torch.nn.functional.normalize(cls, dim=-1)

model = AutoModel.from_pretrained("nomic-ai/CodeRankEmbed", trust_remote_code=True, safe_serialization=True)
model.eval()
# Keep the reference model untouched so export rewrites are checked against
# the actual FP32 model, including every encoder block.
export_model = model
if B > 1 and S < 128:
    export_model = copy.deepcopy(model)
    for layer in export_model.encoder.layers:
        layer.attn = SeparateValueAttention(layer.attn)
w = Wrapper(export_model).eval()

ids = torch.randint(0, 30000, (B, S), dtype=torch.int32)
mask = torch.ones((B, S), dtype=torch.int32)
with torch.no_grad():
    w(ids, mask)  # warm rope caches
    traced = torch.jit.trace(w, (ids, mask), check_trace=False)
    # Check both the finite-mask wrapper and trace against the untouched FP32
    # model on full, mixed-length and partial batches before conversion.
    for real_rows, mixed_lengths in [(B, False), (B, True), (1, True)]:
        probe_ids = ids.clone()
        probe_mask = torch.zeros_like(mask)
        for row in range(real_rows):
            length = min(S, 4 + row * 7) if mixed_lengths else S
            probe_mask[row, :length] = 1
        probe_ids = probe_ids * probe_mask
        reference = torch.nn.functional.normalize(
            model(input_ids=probe_ids, attention_mask=probe_mask)[0][:real_rows, 0, :], dim=-1
        )
        actual = w(probe_ids, probe_mask)[:real_rows]
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(traced(probe_ids, probe_mask)[:real_rows], actual, rtol=1e-4, atol=1e-5)

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
