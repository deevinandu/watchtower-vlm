"""Benchmark the VLM engine outside ROS: load time, latency, and raw answers."""

import argparse
import statistics
import time

from PIL import Image, ImageDraw

from watchtower_vlm.vlm_engine import DEFAULT_MODEL_ID, DEFAULT_REVISION, VLMEngine


def synthetic_image(size=(640, 480)):
    """Grey floor with a blue puddle: a crude stand-in when no image is given."""
    image = Image.new('RGB', size, (150, 150, 150))
    draw = ImageDraw.Draw(image)
    draw.ellipse((200, 280, 460, 400), fill=(40, 90, 200))
    return image


def percentile(values, pct):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', help='Image file (default: synthetic puddle)')
    parser.add_argument('--prompt', default='Is there a liquid spill on the floor?')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--model-path', default=DEFAULT_MODEL_ID)
    parser.add_argument('--revision', default=DEFAULT_REVISION)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)

    image = Image.open(args.image).convert('RGB') if args.image else synthetic_image()
    engine = VLMEngine(model_path=args.model_path, revision=args.revision, device=args.device)

    start = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - start
    print(f'model:    {args.model_path}@{args.revision} on {engine.resolved_device}')
    print(f'load:     {load_s:.1f} s')

    start = time.perf_counter()
    warm = engine.evaluate(image, args.prompt)
    print(f'warm-up:  {(time.perf_counter() - start) * 1000:.0f} ms')

    latencies = []
    for _ in range(args.runs):
        start = time.perf_counter()
        result = engine.evaluate(image, args.prompt)
        latencies.append((time.perf_counter() - start) * 1000)

    print(f'latency:  mean {statistics.mean(latencies):.0f} ms, '
          f'p50 {percentile(latencies, 50):.0f} ms, p95 {percentile(latencies, 95):.0f} ms '
          f'over {args.runs} runs ({1000 / statistics.mean(latencies):.1f} Hz)')
    if engine.resolved_device.startswith('cuda'):
        import torch
        print(f'gpu mem:  {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB peak')
    print(f'prompt:   {args.prompt}')
    print(f'answer:   {warm.raw_answer!r} -> anomaly={result.anomaly} '
          f'(confidence {result.confidence:.2f})')


if __name__ == '__main__':
    main()
