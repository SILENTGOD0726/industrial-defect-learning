"""Download public author weights and verify the official CIFAR-10 archive."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity, extract_archive

ROOT = Path(__file__).resolve().parent
WEIGHT_URL = "https://github.com/VainF/Torch-Pruning/releases/download/v1.1.4/cifar10_resnet56.pth"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DepGraph-learning/1.0"})
    print(f"Downloading {target.name}", flush=True)
    with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as output:
        expected = int(response.headers.get("Content-Length", 0))
        received = 0
        last = time.monotonic()
        while block := response.read(1024 * 1024):
            output.write(block)
            received += len(block)
            if time.monotonic() - last > 10:
                print(f"  {received / 2**20:.1f} MiB / {expected / 2**20:.1f} MiB", flush=True)
                last = time.monotonic()
    if expected and received != expected:
        raise RuntimeError(f"Incomplete download: {received} != {expected}")
    partial.replace(target)
    print(f"Downloaded {received / 2**20:.1f} MiB", flush=True)


def main():
    data_root = ROOT / "data"
    data_root.mkdir(exist_ok=True)
    archive = data_root / CIFAR10.filename
    if not check_integrity(str(archive), CIFAR10.tgz_md5):
        download(CIFAR10.url, archive)
    if not check_integrity(str(archive), CIFAR10.tgz_md5):
        raise RuntimeError("CIFAR-10 archive MD5 mismatch")
    try:
        train = CIFAR10(str(data_root), train=True, download=False)
        test = CIFAR10(str(data_root), train=False, download=False)
    except RuntimeError:
        print("Extracting verified archive", flush=True)
        extract_archive(str(archive), str(data_root))
        train = CIFAR10(str(data_root), train=True, download=False)
        test = CIFAR10(str(data_root), train=False, download=False)
    weights = ROOT / "weights" / "cifar10_resnet56.pth"
    if not weights.exists():
        download(WEIGHT_URL, weights)
    manifest = {
        "dataset_url": CIFAR10.url,
        "dataset_archive_md5_expected": CIFAR10.tgz_md5,
        "dataset_archive_md5_actual": digest(archive, "md5"),
        "dataset_archive_sha256": digest(archive),
        "train_count": len(train), "test_count": len(test),
        "weight_url": WEIGHT_URL, "weight_bytes": weights.stat().st_size,
        "weight_sha256_recorded": digest(weights),
        "weight_hash_note": "Locally recorded hash; no independent publisher SHA256 was provided.",
    }
    (ROOT / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
