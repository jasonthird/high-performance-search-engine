import time, sys, numpy as np
import coremltools as ct

B, S = 8, 256
path = f"coderank_b{B}_s{S}.mlpackage"
ids = np.random.randint(0, 30000, (B, S)).astype(np.int32)
mask = np.ones((B, S), dtype=np.int32)

for name, cu in [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
                 ("ALL", ct.ComputeUnit.ALL),
                 ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU)]:
    m = ct.models.MLModel(path, compute_units=cu)
    inp = {"input_ids": ids, "attention_mask": mask}
    for _ in range(3): m.predict(inp)  # warmup + compile
    t0 = time.time(); n = 15
    for _ in range(n): m.predict(inp)
    dt = (time.time() - t0) / n
    print(f"{name:<12} {dt*1000:7.1f} ms/batch  {B*S/dt:9.0f} tok/s  {B/dt:6.1f} chunks/s")
