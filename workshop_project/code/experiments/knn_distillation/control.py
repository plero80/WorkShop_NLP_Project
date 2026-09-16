"""Inspect progress or request a pause from another terminal/kernel."""
from pathlib import Path
import argparse
from knn_distillation.io import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'pause'])
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or Path(read(args.project / 'knn_distillation_outputs/latest.json')['output'])
    if not out.exists():
        out = args.project / 'knn_distillation_outputs' / out.name
    if args.action == 'pause':
        (out / 'PAUSE').touch()
        print('Pause requested. Wait until status is paused before stopping the pod.')
    else:
        print((out / 'status.json').read_text())
        print('Output:', out)


if __name__ == '__main__':
    main()
