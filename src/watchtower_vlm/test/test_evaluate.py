import io
import math

from PIL import Image
import pytest

from watchtower_vlm.evaluate import load_dataset, print_report, run, summarise, write_csv
from watchtower_vlm.vlm_engine import VLMEngine

BLUE, GREY = (0, 0, 255), (128, 128, 128)


class ColourModel:
    """Says 'yes' to blue images, 'no' to grey, and rambles on anything else."""

    def query(self, image, prompt):
        pixel = image.getpixel((0, 0))
        answer = {BLUE: 'Yes.', GREY: 'No.'}.get(pixel, 'A picture of a floor.')
        return {'answer': answer}


def make_dataset(root, anomaly, normal):
    for folder, colours in (('anomaly', anomaly), ('normal', normal)):
        (root / folder).mkdir(parents=True)
        for i, colour in enumerate(colours):
            Image.new('RGB', (8, 8), colour).save(root / folder / f'{i}.png')
    return root


def test_folder_dataset_and_metrics(tmp_path):
    # 2 anomalies (one missed), 3 normals (one false alarm, one ambiguous -> 0.5 -> anomaly)
    root = make_dataset(tmp_path, [BLUE, GREY], [GREY, BLUE, (1, 2, 3)])
    samples = load_dataset(root)
    assert sum(s.label for s in samples) == 2 and len(samples) == 5

    predictions = run(VLMEngine(model=ColourModel()), samples, ['Spill?'])
    m = summarise(predictions)
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (1, 2, 1, 1)
    assert m['precision'] == pytest.approx(1 / 3)
    assert m['recall'] == pytest.approx(0.5)
    assert m['fpr'] == pytest.approx(2 / 3)
    assert m['ambiguous'] == 1


def test_csv_dataset_resolves_relative_paths(tmp_path):
    Image.new('RGB', (8, 8), BLUE).save(tmp_path / 'a.png')
    Image.new('RGB', (8, 8), GREY).save(tmp_path / 'b.png')
    (tmp_path / 'labels.csv').write_text('path,label\na.png,anomaly\nb.png,0\n')
    samples = load_dataset(tmp_path / 'labels.csv')
    assert [(s.path.name, s.label) for s in samples] == [('a.png', True), ('b.png', False)]


def test_bad_inputs_raise(tmp_path):
    with pytest.raises(ValueError):
        load_dataset(tmp_path)  # no anomaly/ normal/ folders
    (tmp_path / 'labels.csv').write_text('path,label\na.png,maybe\n')
    with pytest.raises(ValueError):
        load_dataset(tmp_path / 'labels.csv')


def test_report_and_csv_cover_every_prompt(tmp_path):
    root = make_dataset(tmp_path / 'data', [BLUE], [GREY])
    prompts = ['Spill?', 'Is anything on the floor that should not be there?' * 2]
    predictions = run(VLMEngine(model=ColourModel()), load_dataset(root), prompts)
    out = io.StringIO()
    print_report(predictions, prompts, out=out)
    assert 'Spill?' in out.getvalue() and '...' in out.getvalue()
    write_csv(predictions, tmp_path / 'out.csv')
    assert 'ANY' in out.getvalue()
    # 2 images x (2 prompts + 1 ANY row) + header
    assert len((tmp_path / 'out.csv').read_text().strip().splitlines()) == 1 + 6


def test_empty_class_gives_nan_not_crash():
    assert math.isnan(summarise([])['precision'])


class EncodingModel(ColourModel):
    def __init__(self):
        self.encodes = 0

    def encode_image(self, image):
        self.encodes += 1
        return image


def test_any_prompt_ors_results_and_encodes_once(tmp_path):
    from watchtower_vlm.evaluate import ANY_PROMPT

    root = make_dataset(tmp_path, [BLUE], [GREY])
    model = EncodingModel()
    predictions = run(VLMEngine(model=model), load_dataset(root), ['a?', 'b?', 'c?'])
    assert model.encodes == 2  # once per image, not per prompt
    any_rows = [p for p in predictions if p.prompt == ANY_PROMPT]
    assert [p.predicted for p in any_rows] == [True, False]
