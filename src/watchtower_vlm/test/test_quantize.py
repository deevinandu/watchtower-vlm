import ctypes
from pathlib import Path
import sys
import types

import pytest

from watchtower_vlm.quantize import main, output_path, quantize


class Params(ctypes.Structure):
    _fields_ = [('ftype', ctypes.c_int), ('nthread', ctypes.c_int)]


@pytest.fixture
def fake_llama_cpp(monkeypatch):
    calls = []
    module = types.ModuleType('llama_cpp')
    module.LLAMA_FTYPE_MOSTLY_Q4_K_M = 15
    module.LLAMA_FTYPE_MOSTLY_Q8_0 = 7
    module.llama_model_quantize_default_params = Params

    def quantize_impl(src, dst, params_ref):
        params = ctypes.cast(params_ref, ctypes.POINTER(Params)).contents
        calls.append((src.decode(), dst.decode(), params.ftype, params.nthread))
        Path(dst.decode()).write_bytes(b'q' * 10)
        return module.status

    module.status = 0
    module.llama_model_quantize = quantize_impl
    monkeypatch.setitem(sys.modules, 'llama_cpp', module)
    return module, calls


def test_output_name_strips_precision_suffix(tmp_path):
    assert output_path(tmp_path, '/c/moondream2-text-model-f16.gguf', 'Q4_K_M') == (
        tmp_path / 'moondream2-text-model-Q4_K_M.gguf')


def test_quantize_passes_ftype_and_threads(fake_llama_cpp, tmp_path):
    _, calls = fake_llama_cpp
    quantize('in.gguf', str(tmp_path / 'out.gguf'), 'Q8_0', threads=4)
    assert calls == [('in.gguf', str(tmp_path / 'out.gguf'), 7, 4)]


def test_quantize_raises_on_failure(fake_llama_cpp, tmp_path):
    fake_llama_cpp[0].status = 1
    with pytest.raises(RuntimeError, match='Q4_K_M'):
        quantize('in.gguf', str(tmp_path / 'out.gguf'), 'Q4_K_M')


def test_main_skips_existing_outputs(fake_llama_cpp, tmp_path, capsys):
    _, calls = fake_llama_cpp
    source = tmp_path / 'moondream2-text-model-f16.gguf'
    source.write_bytes(b's' * 40)
    out = tmp_path / 'models'
    argv = ['--gguf-file', str(source), '--out-dir', str(out), '--type', 'Q4_K_M', 'Q8_0']
    main(argv)
    main(argv)
    assert len(calls) == 2
    assert capsys.readouterr().out.count('exists, skipping') == 2
