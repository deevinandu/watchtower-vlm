from PIL import Image
import pytest

from watchtower_vlm.vlm_engine import parse_answer, VLMEngine, YES_NO_SUFFIX


class FakeModel:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def query(self, image, prompt):
        self.prompts.append(prompt)
        return {'answer': self.answer}


@pytest.mark.parametrize('text,expected', [
    ('Yes', True), ('yes, liquid is on the floor.', True), ('No.', False),
    ('  NO', False), ('There is no spill.', False), ('I think yes', True),
    ('A robot arm on a table.', None), ('maybe yes, maybe no', None),
])
def test_parse_answer(text, expected):
    assert parse_answer(text) is expected


def test_detect_uses_model_answer_and_yes_no_suffix():
    model = FakeModel('Yes')
    engine = VLMEngine(model=model)
    image = Image.new('RGB', (8, 8))
    assert engine.detect(image, 'Is liquid spilling?') is True
    assert model.prompts == ['Is liquid spilling?' + YES_NO_SUFFIX]


def test_ambiguous_answer_uses_configured_confidence():
    engine = VLMEngine(model=FakeModel('a cat'), ambiguous_confidence=0.3)
    result = engine.evaluate(Image.new('RGB', (8, 8)), 'Spill?')
    assert result.confidence == 0.3
    assert result.anomaly is False


def test_query_before_load_raises():
    with pytest.raises(RuntimeError):
        VLMEngine().query(Image.new('RGB', (8, 8)), 'x')
