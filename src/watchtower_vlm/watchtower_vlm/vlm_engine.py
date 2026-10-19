"""ROS-independent wrapper around a local Vision-Language Model (moondream2)."""

from dataclasses import dataclass
import re
import threading
from typing import Optional

DEFAULT_MODEL_ID = 'vikhyatk/moondream2'
DEFAULT_REVISION = '2025-01-09'
YES_NO_SUFFIX = " Answer with only 'yes' or 'no'."

_YES = re.compile(r'\b(yes|yeah|yep|true|affirmative)\b')
_NO = re.compile(r'\b(no|nope|false|negative|none)\b')


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of one VLM query: ``confidence`` is in [0, 1] that an anomaly exists."""

    anomaly: bool
    confidence: float
    raw_answer: str


def parse_answer(text: str) -> Optional[bool]:
    """Map free-form VLM output to True (yes), False (no) or None (ambiguous)."""
    answer = text.strip().lower()
    if answer.startswith('yes'):
        return True
    if answer.startswith('no'):
        return False
    has_yes, has_no = bool(_YES.search(answer)), bool(_NO.search(answer))
    if has_yes != has_no:
        return has_yes
    return None


class VLMEngine:
    """Loads a VLM once and answers yes/no anomaly questions about PIL images."""

    def __init__(self, model_path: str = DEFAULT_MODEL_ID,
                 revision: Optional[str] = DEFAULT_REVISION,
                 device: str = 'auto', ambiguous_confidence: float = 0.5,
                 model=None):
        self.model_path = model_path
        self.revision = revision or None
        self.device = device
        self.ambiguous_confidence = ambiguous_confidence
        self._model = model
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Load weights (slow, may download). Safe to call more than once."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM

        device = self.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device.startswith('cuda') else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, revision=self.revision, trust_remote_code=True,
            torch_dtype=dtype, device_map={'': device})
        model.eval()
        self._model = model

    def query(self, image, prompt: str) -> str:
        """Return the model's raw text answer for ``prompt`` about ``image``."""
        if self._model is None:
            raise RuntimeError('VLMEngine.load() must be called before inference')
        with self._lock:
            if hasattr(self._model, 'query'):
                return str(self._model.query(image, prompt)['answer'])
            # Older moondream2 revisions expose encode_image/answer_question.
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, revision=self.revision)
            encoded = self._model.encode_image(image)
            return str(self._model.answer_question(encoded, prompt, tokenizer))

    def evaluate(self, image, prompt: str) -> DetectionResult:
        """Ask a yes/no question; ambiguous answers get ``ambiguous_confidence``."""
        raw = self.query(image, prompt.rstrip() + YES_NO_SUFFIX)
        verdict = parse_answer(raw)
        if verdict is None:
            confidence = self.ambiguous_confidence
        else:
            confidence = 1.0 if verdict else 0.0
        return DetectionResult(anomaly=confidence >= 0.5, confidence=confidence, raw_answer=raw)

    def detect(self, image, prompt: str) -> bool:
        """Return True if the VLM reports an anomaly in ``image``."""
        return self.evaluate(image, prompt).anomaly
