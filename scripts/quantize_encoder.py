"""Produce GGUF Q8_0 or Q4_0 matrix weights; preserve vectors and metadata."""
import argparse
from pathlib import Path
import gguf
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--type', choices=['Q8_0', 'Q4_0'], default='Q8_0')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; choose a new artifact path')
    reader = gguf.GGUFReader(args.source)
    if any(t.tensor_type not in {gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32} for t in reader.tensors):
        raise SystemExit('Input must contain only original F16/F32 tensors, not already quantized weights')
    writer = gguf.GGUFWriter(args.output, reader.fields['general.architecture'].contents())
    for name, field in reader.fields.items():
        if name.startswith('GGUF.') or name in {'general.architecture', 'general.file_type', 'general.quantization_version'}:
            continue
        writer.add_key_value(name, field.contents(), field.types[0],
                             field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    kind = getattr(gguf.GGMLQuantizationType, args.type)
    writer.add_file_type(getattr(gguf.LlamaFileType, 'MOSTLY_' + args.type))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    changed = 0
    for tensor in reader.tensors:
        if tensor.data.ndim == 2 and tensor.data.shape[-1] % 32 == 0 and tensor.name.endswith('.weight'):
            data = gguf.quants.quantize(tensor.data.astype(np.float32), kind)
            writer.add_tensor(tensor.name, data, raw_dtype=kind)
            changed += 1
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    # Check structural preservation; successful inference must be tested separately.
    check = gguf.GGUFReader(args.output)
    assert [(t.name, t.shape.tolist()) for t in reader.tensors] == [(t.name, t.shape.tolist()) for t in check.tensors]
    print(f'{changed} matrices converted to {args.type}; {args.output.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
