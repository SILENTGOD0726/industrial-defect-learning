"""A bounded, reproducible DepGraph learning experiment on real CIFAR-10."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import resnet56

ROOT = Path(__file__).resolve().parent
COMMIT = "e80127d7c6935c319a7dd6719b8a72e3c1fcbff9"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loaders(args):
    normalize = transforms.Normalize((.4914, .4822, .4465), (.2023, .1994, .2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train = datasets.CIFAR10(str(ROOT / "data"), train=True, download=False, transform=train_transform)
    test = datasets.CIFAR10(str(ROOT / "data"), train=False, download=False, transform=test_transform)
    generator = torch.Generator().manual_seed(args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    return (
        DataLoader(train, shuffle=True, drop_last=True, generator=generator, **common),
        DataLoader(test, shuffle=False, **common),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    started = time.monotonic()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        if logits.shape[1] != 10 or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid model outputs")
        correct += (logits.argmax(1) == labels).sum().item()
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        total += labels.numel()
    return dict(accuracy=correct / total, correct=correct, total=total,
                loss=loss_sum / total, evaluation_wall_seconds=time.monotonic() - started)


def size(model, example):
    model.eval()
    macs, params = tp.utils.count_ops_and_params(model, example_inputs=example)
    return dict(params=int(params), macs_per_image=float(macs),
                counter="torch_pruning.utils.count_ops_and_params; MAC-style operation estimate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=ROOT / "run_1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mac-ratio", type=float, default=2.11)
    parser.add_argument("--reload-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.target_mac_ratio <= 1:
        parser.error("epochs must be positive; target MAC ratio must exceed one")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the selected project environment")
    device = torch.device("cuda:0")
    train_loader, test_loader = loaders(args)
    if args.reload_only:
        print("Independent-process check: reload our locally generated model and evaluate.", flush=True)
        model = torch.load(args.out_dir / "pruned_finetuned.pt", map_location=device, weights_only=False)
        probe = torch.load(args.out_dir / "reload_probe.pt", map_location=device, weights_only=True)
        model.eval()
        with torch.no_grad():
            logits = model(probe["images"])
        result = dict(
            independent_process=True,
            identical_logits=bool(torch.equal(logits, probe["logits"])),
            maximum_logit_error=float((logits - probe["logits"]).abs().max().item()),
            evaluation=evaluate(model, test_loader, device),
            checkpoint_sha256=fingerprint(args.out_dir / "pruned_finetuned.pt"),
        )
        save_json(args.out_dir / "reload_check.json", result)
        print(json.dumps(result, indent=2), flush=True)
        if not result["identical_logits"]:
            raise RuntimeError("Reloaded logits differ")
        return
    if (args.out_dir / "final_info.json").exists():
        raise FileExistsError("Completed run exists; choose a new --out_dir, e.g. run_2")
    started = time.monotonic()
    settings = vars(args).copy()
    settings["out_dir"] = str(args.out_dir)
    results = dict(
        scope="Learning pipeline check, not full paper reproduction",
        author_commit=COMMIT, source_tag="v1.6.1",
        method="group_norm / without sparse learning / global / 400 progressive steps",
        train_count=50000, test_count=10000, input_shape=[1, 3, 32, 32],
        precision="FP32", config=settings,
        environment={k: importlib.metadata.version(k) for k in ["torch", "torchvision", "numpy", "torch-pruning"]},
        gpu=torch.cuda.get_device_name(device),
        data_manifest=json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8")),
        script_sha256=fingerprint(Path(__file__)), model_sha256=fingerprint(ROOT / "model.py"),
        history=[], stages={},
    )
    model = resnet56(num_classes=10)
    state = torch.load(ROOT / "weights" / "cifar10_resnet56.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    example = torch.zeros(1, 3, 32, 32, device=device)
    print("Stage 1: evaluate author baseline on all 10,000 test images.", flush=True)
    baseline = {**size(model, example), **evaluate(model, test_loader, device)}
    results["stages"]["baseline"] = baseline
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images[:8].to(device)
    with torch.no_grad():
        sample_predictions = {"baseline": model(sample_images).argmax(1).cpu().tolist()}
    results["sample_labels"] = sample_labels[:8].tolist()
    results["sample_class_names"] = test_loader.dataset.classes
    save_json(args.out_dir / "progress.json", results)
    print(f"Baseline accuracy={baseline['accuracy']:.4%}; params={baseline['params']}; MACs={baseline['macs_per_image']:.0f}", flush=True)
    print("Stage 2: prune coupled groups until target MAC reduction; no sparse training.", flush=True)
    pruner = tp.pruner.GroupNormPruner(
        model, example, importance=tp.importance.GroupMagnitudeImportance(p=2),
        global_pruning=True, iterative_steps=400, pruning_ratio=1.0,
        ignored_layers=[model.fc],
    )
    steps = 0
    current_size = baseline
    while baseline["macs_per_image"] / current_size["macs_per_image"] < args.target_mac_ratio and steps < 400:
        pruner.step()
        steps += 1
        current_size = size(model, example)
        if steps % 25 == 0:
            print(f"  pruning step {steps}: MAC ratio={baseline['macs_per_image']/current_size['macs_per_image']:.3f}", flush=True)
    del pruner
    gc.collect()
    pruned = {**current_size, **evaluate(model, test_loader, device)}
    if not (pruned["params"] < baseline["params"] and pruned["macs_per_image"] < baseline["macs_per_image"]):
        raise RuntimeError("Structural size did not decrease")
    results["pruning_steps"] = steps
    results["mac_reduction_ratio"] = baseline["macs_per_image"] / pruned["macs_per_image"]
    results["stages"]["pruned"] = pruned
    with torch.no_grad():
        sample_predictions["pruned"] = model(sample_images).argmax(1).cpu().tolist()
    save_json(args.out_dir / "progress.json", results)
    print(f"Pruned accuracy={pruned['accuracy']:.4%}; params={pruned['params']}; MAC ratio={results['mac_reduction_ratio']:.3f}", flush=True)
    print(f"Stage 3: finetune for {args.epochs} epoch(s); training set only, no test-set model selection.", flush=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=.1)
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        total = 0
        for index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            total += labels.numel()
            if index % 50 == 0:
                print(f"  epoch {epoch+1}, batch {index+1}/{len(train_loader)}, loss={loss.item():.4f}", flush=True)
        record = dict(epoch=epoch+1, train_loss=loss_sum / total, train_seen=total,
                      learning_rate=optimizer.param_groups[0]["lr"], **evaluate(model, test_loader, device))
        results["history"].append(record)
        save_json(args.out_dir / "progress.json", results)
        print(f"Epoch {epoch+1}: test accuracy={record['accuracy']:.4%}", flush=True)
        scheduler.step()
    results["stages"]["finetuned"] = {**size(model, example), **record}
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:8].to(device)
    with torch.no_grad():
        probe_logits = model(images)
    sample_predictions["finetuned"] = probe_logits.argmax(1).cpu().tolist()
    results["sample_predictions"] = sample_predictions
    torch.save({"images": images.cpu(), "logits": probe_logits.cpu()}, args.out_dir / "reload_probe.pt")
    torch.save(model, args.out_dir / "pruned_finetuned.pt")
    results["checkpoint_sha256"] = fingerprint(args.out_dir / "pruned_finetuned.pt")
    results["checkpoint_bytes"] = (args.out_dir / "pruned_finetuned.pt").stat().st_size
    results["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    results["total_wall_seconds"] = time.monotonic() - started
    results["latency_measured"] = False
    save_json(args.out_dir / "final_info.json", results)
    with (ROOT / "notes.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"\nRun {args.out_dir.name}: baseline={baseline['accuracy']:.4%}, pruned={pruned['accuracy']:.4%}, finetuned={record['accuracy']:.4%}; {args.epochs} epoch(s); MAC ratio={results['mac_reduction_ratio']:.3f}; not full reproduction.\n")
    print("Pipeline complete. Run --reload-only in a new process, then plot.py.", flush=True)


if __name__ == "__main__":
    main()
