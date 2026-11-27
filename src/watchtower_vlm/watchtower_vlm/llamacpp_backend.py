"""moondream2 GGUF inference via llama.cpp (llama-cpp-python), for CPU / edge targets."""

import base64
import fnmatch
import io
import os

DEFAULT_GGUF_FILE = '*text-model*'
DEFAULT_MMPROJ_FILE = '*mmproj*'


class LlamaCppMoondream:
    """Adapter exposing the ``query(image, prompt) -> {'answer': str}`` interface.

    ``gguf_file`` / ``mmproj_file`` are local paths, or filename globs resolved inside the
    Hugging Face repo ``repo_id`` at ``revision`` (downloaded and cached on first use).
    """

    def __init__(self, repo_id: str, gguf_file: str = DEFAULT_GGUF_FILE,
                 mmproj_file: str = DEFAULT_MMPROJ_FILE, revision=None,
                 n_gpu_layers: int = -1, n_ctx: int = 2048, max_tokens: int = 16):
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MoondreamChatHandler

        handler = MoondreamChatHandler(
            clip_model_path=resolve_file(repo_id, mmproj_file, revision), verbose=False)
        self._llm = Llama(model_path=resolve_file(repo_id, gguf_file, revision),
                          chat_handler=handler, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                          verbose=False)
        self._max_tokens = max_tokens

    def query(self, image, prompt: str) -> dict:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        uri = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()
        response = self._llm.create_chat_completion(
            messages=[{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': uri}},
                {'type': 'text', 'text': prompt},
            ]}],
            max_tokens=self._max_tokens, temperature=0.0)
        return {'answer': response['choices'][0]['message']['content'] or ''}


def resolve_file(repo_id: str, pattern: str, revision=None) -> str:
    """Return a local path: ``pattern`` itself if it is a file, else the one repo file it globs."""
    if os.path.isfile(pattern):
        return pattern
    from huggingface_hub import hf_hub_download, list_repo_files

    matches = fnmatch.filter(list_repo_files(repo_id, revision=revision), pattern)
    if len(matches) != 1:
        raise ValueError(f'{pattern!r} matched {matches or "nothing"} in {repo_id}@{revision}; '
                         'it must match exactly one file')
    return hf_hub_download(repo_id, matches[0], revision=revision)


def gpu_offload_supported() -> bool:
    import llama_cpp
    return bool(llama_cpp.llama_supports_gpu_offload())
