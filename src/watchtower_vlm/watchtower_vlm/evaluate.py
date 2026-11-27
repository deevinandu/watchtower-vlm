"""Evaluate prompts against a labelled image set: confusion matrix, precision/recall, latency.

Dataset layouts:
  * a folder containing ``anomaly/`` and ``normal/`` subfolders of images, or
  * a CSV file with ``path,label`` rows (label: 1/0, true/false, anomaly/normal);
    relative paths are resolved against the CSV's folder.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time

from PIL import Image

from watchtower_vlm.benchmark import percentile
from watchtower_vlm.vlm_engine import (
    DEFAULT_MODEL_ID, DEFAULT_PROMPT, DEFAULT_REVISION, VLMEngine)

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ANY_PROMPT = 'ANY (OR of all prompts, as the node decides)'
TRUE_LABELS = {'1', 'true', 'yes', 'anomaly'}
FALSE_LABELS = {'0', 'false', 'no', 'normal'}


@dataclass
class Sample:
    path: Path
    label: bool


@dataclass
class Prediction:
    prompt: str
    sample: Sample
    answer: str
    confidence: float
    predicted: bool
    latency_ms: float


def load_dataset(source: Path) -> list:
    if source.is_dir():
        samples = []
        for folder, label in (('anomaly', True), ('normal', False)):
            directory = source / folder
            if not directory.is_dir():
                raise ValueError(f'{source} must contain anomaly/ and normal/ subfolders')
            samples += [Sample(p, label) for p in sorted(directory.rglob('*'))
                        if p.suffix.lower() in IMAGE_SUFFIXES]
        return samples

    samples = []
    with open(source, newline='') as handle:
        for row in csv.DictReader(handle):
            label = row['label'].strip().lower()
            if label not in TRUE_LABELS | FALSE_LABELS:
                raise ValueError(f'Unrecognised label {row["label"]!r} for {row["path"]}')
            path = Path(row['path'])
            samples.append(Sample(path if path.is_absolute() else source.parent / path,
                                  label in TRUE_LABELS))
    return samples


def run(engine: VLMEngine, samples: list, prompts: list) -> list:
    """Evaluate every prompt on every image; with >1 prompt also add ANY_PROMPT rows."""
    predictions = []
    for sample in samples:
        image = Image.open(sample.path).convert('RGB')
        started = time.perf_counter()
        results = engine.evaluate_many(image, prompts)
        total_ms = (time.perf_counter() - started) * 1000
        for prompt, result in zip(prompts, results):
            predictions.append(Prediction(prompt, sample, result.raw_answer.strip(),
                                          result.confidence, result.anomaly,
                                          total_ms / len(prompts)))
        if len(prompts) > 1:
            worst = max(results, key=lambda r: r.confidence)
            predictions.append(Prediction(ANY_PROMPT, sample, worst.raw_answer.strip(),
                                          worst.confidence, worst.anomaly, total_ms))
    return predictions


def summarise(predictions: list) -> dict:
    """Metrics for one prompt's predictions."""
    tp = sum(p.predicted and p.sample.label for p in predictions)
    fp = sum(p.predicted and not p.sample.label for p in predictions)
    fn = sum(not p.predicted and p.sample.label for p in predictions)
    tn = sum(not p.predicted and not p.sample.label for p in predictions)
    latencies = [p.latency_ms for p in predictions]

    def ratio(num, den):
        return num / den if den else float('nan')

    return {
        'n': len(predictions), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
        'fpr': ratio(fp, fp + tn), 'accuracy': ratio(tp + tn, len(predictions)),
        'ambiguous': sum(p.confidence not in (0.0, 1.0) for p in predictions),
        'latency_mean_ms': statistics.mean(latencies) if latencies else float('nan'),
        'latency_p95_ms': percentile(latencies, 95) if latencies else float('nan'),
    }


def print_report(predictions: list, prompts: list, out=sys.stdout):
    header = (f'{"prompt":<48} {"n":>4} {"TP":>4} {"FP":>4} {"FN":>4} {"TN":>4} '
              f'{"prec":>5} {"recall":>6} {"FPR":>5} {"ambig":>5} {"ms":>6}')
    print(header, file=out)
    print('-' * len(header), file=out)
    for prompt in prompts + ([ANY_PROMPT] if len(prompts) > 1 else []):
        m = summarise([p for p in predictions if p.prompt == prompt])
        label = prompt if len(prompt) <= 48 else prompt[:45] + '...'
        print(f'{label:<48} {m["n"]:>4} {m["tp"]:>4} {m["fp"]:>4} {m["fn"]:>4} {m["tn"]:>4} '
              f'{m["precision"]:>5.2f} {m["recall"]:>6.2f} {m["fpr"]:>5.2f} '
              f'{m["ambiguous"]:>5} {m["latency_mean_ms"]:>6.0f}', file=out)


def write_csv(predictions: list, path: Path):
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['prompt', 'path', 'label', 'predicted', 'confidence', 'answer',
                         'latency_ms'])
        for p in predictions:
            writer.writerow([p.prompt, p.sample.path, int(p.sample.label), int(p.predicted),
                             p.confidence, p.answer, f'{p.latency_ms:.1f}'])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', type=Path, help='Folder (anomaly/, normal/) or CSV file')
    parser.add_argument('--prompt', action='append', dest='prompts',
                        help='Prompt to evaluate; repeat to compare several')
    parser.add_argument('--ambiguous-confidence', type=float, default=0.5,
                        help='Confidence for answers that are neither yes nor no')
    parser.add_argument('--csv', type=Path, help='Write per-image predictions here')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_ID)
    parser.add_argument('--revision', default=DEFAULT_REVISION)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)

    prompts = args.prompts or [DEFAULT_PROMPT]
    samples = load_dataset(args.dataset)
    if not samples:
        parser.error(f'No images found in {args.dataset}')
    positives = sum(s.label for s in samples)
    print(f'{len(samples)} images ({positives} anomaly, {len(samples) - positives} normal), '
          f'{len(prompts)} prompt(s)')

    engine = VLMEngine(model_path=args.model_path, revision=args.revision, device=args.device,
                       ambiguous_confidence=args.ambiguous_confidence)
    engine.load()
    predictions = run(engine, samples, prompts)
    print_report(predictions, prompts)
    if args.csv:
        write_csv(predictions, args.csv)
        print(f'Per-image results written to {args.csv}')


if __name__ == '__main__':
    main()
