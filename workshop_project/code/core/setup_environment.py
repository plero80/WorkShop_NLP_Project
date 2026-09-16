"""Install missing dependencies in the launcher's Python; never call HF login."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def pip(*args):
    subprocess.run([sys.executable,'-m','pip',*args],check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repair-cuda',action='store_true',help='Install the official torch 2.8.0 CUDA 12.8 wheel.')
    args=parser.parse_args()
    if sys.version_info < (3,10):
        raise SystemExit('Use Python 3.12 on the RunPod image (minimum 3.10).')
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:
        pip('install','packaging>=23')
        from packaging.requirements import Requirement
        from packaging.version import Version
    try:
        torch_version=metadata.version('torch')
    except metadata.PackageNotFoundError:
        torch_version=None
    if args.repair_cuda or torch_version is None or Version(torch_version)<Version('2.8.0'):
        print('Installing PyTorch 2.8.0 with CUDA 12.8 for the target GPU.',flush=True)
        pip('install','--upgrade','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cu128')
    missing=[]
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#'):
            continue
        req=Requirement(line)
        try:
            installed=metadata.version(req.name)
            if installed not in req.specifier:
                missing.append(line)
        except metadata.PackageNotFoundError:
            missing.append(line)
    if missing:
        pip('install',*missing)
    result=subprocess.run([sys.executable,'-c',
        'import torch,transformers,peft,numpy,pandas,scipy,sklearn; '
        'print("Dependencies:",torch.__version__,transformers.__version__,peft.__version__); '
        'print("CUDA available:",torch.cuda.is_available(),"build:",torch.version.cuda)'],check=True)
    print('Dependency check finished. GPU execution is checked by preflight.',flush=True)


if __name__=='__main__':
    main()
