"""Standalone scoring: exact base weights + trained adapter, with no kNN memory."""
import argparse
import json
from pathlib import Path
from knn_distillation.student import StudentScorer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True, help='Exact pinned proxy base checkpoint directory.')
    p.add_argument('--student', type=Path, required=True, help='Selected student directory with reward_config.json.')
    p.add_argument('--input', type=Path, required=True, help='JSONL rows containing prompt and answer.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--batch-size', type=int, default=16)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    s = StudentScorer.load(a.snapshot, a.student, a.batch_size, a.device)
    scores = s.score([r['prompt'] for r in rows], [r['answer'] for r in rows])
    output = [{**row, 'student_raw': float(raw), 'student_reward_z': float((raw - s.calibration['proxy_mean']) / s.calibration['proxy_std'])}
              for row, raw in zip(rows, scores['raw'])]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in output))
    print('Scored', len(output), 'answers:', a.output)


if __name__ == '__main__':
    main()
