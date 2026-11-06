"""第一课：运行 python train.py；重新加载预测：python train.py --predict-only。"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def make_model():
    # [N,1,28,28] -> [N,784] -> [N,256] -> [N,128] -> [N,10]
    # 输出是 logits（原始分数），直接交给 CrossEntropyLoss，不先做 softmax。
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))


def evaluate(model, loader, device):
    model.eval()  # 切换层的运行模式；本课的简单网络没有 Dropout/BatchNorm。
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():  # 评估不记录梯度，避免不必要的开销。
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return {'loss': total_loss / count, 'accuracy': correct / count}


def predict_saved(device):
    checkpoint = torch.load(OUT / 'model.pt', map_location=device, weights_only=True)
    model = make_model().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test = datasets.FashionMNIST(ROOT / 'data', train=False, download=True,
                                 transform=transforms.ToTensor())
    images = torch.stack([test[i][0] for i in range(12)])
    with torch.no_grad():
        probabilities = model(images.to(device)).softmax(1).cpu()
    predictions = probabilities.argmax(1).tolist()
    records = []
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        label = test[i][1]
        pred = predictions[i]
        ax.imshow(images[i, 0], cmap='gray')
        ax.set_title(f'True: {CLASSES[label]}\nPred: {CLASSES[pred]}',
                     color='green' if pred == label else 'red', fontsize=9)
        ax.axis('off')
        records.append({'index': i, 'true': CLASSES[label], 'predicted': CLASSES[pred],
                        'softmax_score': probabilities[i, pred].item()})
    fig.tight_layout()
    fig.savefig(OUT / 'predictions.png', dpi=150)
    plt.close(fig)
    (OUT / 'predictions.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('Reloaded checkpoint; prediction examples:', records[:3], flush=True)
    return model, images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch={torch.__version__}; device={device}', flush=True)
    if args.predict_only:
        predict_saved(device)
        return

    start = time.perf_counter()
    # ToTensor 将灰度像素变成 [0,1] 的 float32，且添加通道维。
    dataset = datasets.FashionMNIST(ROOT / 'data', train=True, download=True,
                                    transform=transforms.ToTensor())
    train, val = random_split(dataset, [54000, 6000],
                              generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val, batch_size=256, num_workers=0)
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    images, labels = next(iter(train_loader))
    print(f'images: shape={list(images.shape)}, dtype={images.dtype}; '
          f'labels: shape={list(labels.shape)}, dtype={labels.dtype}', flush=True)
    print(f'logits shape={list(model(images.to(device)).shape)}', flush=True)
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()             # 1. 清除上一次梯度
            logits = model(images)            # 2. 前向传播
            loss = loss_fn(logits, labels)    # 3. 计算损失
            loss.backward()                   # 4. 计算梯度
            optimizer.step()                  # 5. 更新参数
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
        validation = evaluate(model, val_loader, device)
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / count,
               'train_accuracy': correct / count, 'val_loss': validation['loss'],
               'val_accuracy': validation['accuracy']}
        history.append(row)
        print(json.dumps(row), flush=True)

    # 固定轮数完成后才评估测试集；本课不根据测试集调参。
    test = datasets.FashionMNIST(ROOT / 'data', train=False, download=True,
                                 transform=transforms.ToTensor())
    test_metrics = evaluate(model, DataLoader(test, batch_size=256), device)
    torch.save({'model_state_dict': model.state_dict(), 'classes': CLASSES}, OUT / 'model.pt')
    reloaded, examples = predict_saved(device)
    with torch.no_grad():
        same = torch.equal(model(examples.to(device)), reloaded(examples.to(device)))
    if not same:
        raise RuntimeError('Reloaded predictions differ')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    summary = {'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
               'torch': torch.__version__, 'seed': 42, 'epochs': args.epochs,
               'batch_size': 128, 'learning_rate': 0.001,
               'samples': {'train': len(train), 'validation': len(val), 'test': len(test)},
               'history': history, 'test': test_metrics, 'reload_logits_identical': same,
               'elapsed_seconds_including_download': time.perf_counter() - start,
               'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None}
    (OUT / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, ax in zip(['loss', 'accuracy'], axes):
        ax.plot([r['epoch'] for r in history], [r['train_' + key] for r in history], label='train')
        ax.plot([r['epoch'] for r in history], [r['val_' + key] for r in history], label='validation')
        ax.set(xlabel='Epoch', ylabel=key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'learning_curves.png', dpi=150)
    plt.close(fig)
    print('FINAL', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
