import sys
import types

from PIL import Image
import pytest

from watchtower_vlm.vlm_engine import VLMEngine


@pytest.fixture
def fake_llama_cpp(monkeypatch):
    """Install a stand-in ``llama_cpp`` package that records how it was called."""
    calls = {}

    class Handler:
        def __init__(self, **kwargs):
            calls['handler'] = kwargs

    class Llama:
        def __init__(self, **kwargs):
            calls['llama'] = kwargs

        def create_chat_completion(self, messages, **kwargs):
            calls['messages'], calls['completion'] = messages, kwargs
            return {'choices': [{'message': {'content': ' Yes'}}]}

    root = types.ModuleType('llama_cpp')
    root.Llama = Llama
    root.llama_supports_gpu_offload = lambda: True
    chat = types.ModuleType('llama_cpp.llama_chat_format')
    chat.MoondreamChatHandler = Handler
    monkeypatch.setitem(sys.modules, 'llama_cpp', root)
    monkeypatch.setitem(sys.modules, 'llama_cpp.llama_chat_format', chat)

    hub = types.ModuleType('huggingface_hub')
    hub.list_repo_files = lambda repo, revision=None: [
        'config.json', 'moondream2-text-model-f16.gguf', 'moondream2-mmproj-f16.gguf']
    hub.hf_hub_download = lambda repo, name, revision=None: f'/cache/{repo}@{revision}/{name}'
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    return calls


def test_hub_globs_and_query_round_trip(fake_llama_cpp):
    engine = VLMEngine(model_path='vikhyatk/moondream2', backend='llamacpp')
    engine.load()
    assert engine.resolved_device == 'cuda (llama.cpp)'
    assert fake_llama_cpp['llama']['model_path'] == (
        '/cache/vikhyatk/moondream2@2025-01-09/moondream2-text-model-f16.gguf')
    assert fake_llama_cpp['handler']['clip_model_path'].endswith('mmproj-f16.gguf')

    assert engine.detect(Image.new('RGB', (8, 8)), 'Spill?') is True
    content = fake_llama_cpp['messages'][0]['content']
    assert content[0]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert content[1]['text'].startswith('Spill?')
    assert fake_llama_cpp['completion']['temperature'] == 0.0


def test_local_files_and_cpu_only(fake_llama_cpp, tmp_path):
    gguf, mmproj = tmp_path / 'text.gguf', tmp_path / 'mmproj.gguf'
    gguf.touch()
    mmproj.touch()
    engine = VLMEngine(backend='llamacpp', gguf_file=str(gguf), mmproj_file=str(mmproj),
                       n_gpu_layers=0)
    engine.load()
    assert fake_llama_cpp['llama']['model_path'] == str(gguf)
    assert fake_llama_cpp['handler']['clip_model_path'] == str(mmproj)
    assert engine.resolved_device == 'cpu (llama.cpp)'


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        VLMEngine(backend='onnx')


def test_ambiguous_glob_is_rejected(fake_llama_cpp):
    engine = VLMEngine(backend='llamacpp', gguf_file='*.gguf')
    with pytest.raises(ValueError, match='exactly one'):
        engine.load()
