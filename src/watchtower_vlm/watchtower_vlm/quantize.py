"""Quantize the moondream2 GGUF text model for the llama.cpp backend.

Only the text model is quantized. The vision projector (mmproj) stays f16 and is
still resolved from the repo by the backend.
"""

import argparse
import ctypes
from pathlib import Path

from watchtower_vlm.llamacpp_backend import DEFAULT_GGUF_FILE, resolve_file
from watchtower_vlm.vlm_engine import DEFAULT_MODEL_ID, DEFAULT_REVISION

QUANT_TYPES = ('Q4_0', 'Q4_K_M', 'Q5_K_M', 'Q8_0')


def quantize(source: str, destination: str, quant_type: str, threads: int = 0):
    import llama_cpp

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, f'LLAMA_FTYPE_MOSTLY_{quant_type}')
    params.nthread = threads
    status = llama_cpp.llama_model_quantize(
        source.encode(), destination.encode(), ctypes.byref(params))
    if status != 0:
        raise RuntimeError(f'llama_model_quantize failed ({status}) for {quant_type}')


def output_path(out_dir: Path, source: str, quant_type: str) -> Path:
    stem = Path(source).stem
    for suffix in ('-f16', '-f32', '_f16', '_f32'):
        stem = stem[:-len(suffix)] if stem.endswith(suffix) else stem
    return out_dir / f'{stem}-{quant_type}.gguf'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type', nargs='+', choices=QUANT_TYPES, default=['Q4_K_M'],
                        dest='types')
    parser.add_argument('--out-dir', type=Path, default=Path('models'))
    parser.add_argument('--model-path', default=DEFAULT_MODEL_ID)
    parser.add_argument('--revision', default=DEFAULT_REVISION)
    parser.add_argument('--gguf-file', default=DEFAULT_GGUF_FILE,
                        help='Source f16 GGUF: local path or glob in the repo')
    parser.add_argument('--threads', type=int, default=0, help='0 = all cores')
    parser.add_argument('--force', action='store_true', help='Overwrite existing outputs')
    args = parser.parse_args(argv)

    source = resolve_file(args.model_path, args.gguf_file, args.revision)
    source_mib = Path(source).stat().st_size / 2**20
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'source: {source} ({source_mib:.0f} MiB)')

    for quant_type in args.types:
        destination = output_path(args.out_dir, source, quant_type)
        if destination.exists() and not args.force:
            print(f'{quant_type}: exists, skipping ({destination})')
            continue
        quantize(source, str(destination), quant_type, args.threads)
        mib = destination.stat().st_size / 2**20
        print(f'{quant_type}: {destination} ({mib:.0f} MiB, {mib / source_mib:.0%} of source)')

    print('\nUse with: ros2 launch watchtower_vlm watchtower.launch.py backend:=llamacpp '
          f'gguf_file:={args.out_dir.resolve()}/<file>.gguf')


if __name__ == '__main__':
    main()
