# 第一课：FashionMNIST 完整训练流程

这是入门教学实验，用简单的全连接网络分类服装图片。它不代表原创研究成果。

## 运行

在本仓库根目录的终端执行：

```powershell
python experiments/fashionmnist/train.py
```

默认训练 3 轮。首次运行下载数据，以后直接读取本地文件。重新训练会覆盖 outputs 中的结果。

关闭训练程序后，可以单独加载已保存的参数并预测，无需重新训练：

```powershell
python experiments/fashionmnist/train.py --predict-only
```

本次使用已有环境：Python 3.12、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121。没有更改这些包。

## 按顺序阅读 train.py

1. `datasets.FashionMNIST`：加载图片和标签。`ToTensor` 把图片变成 float32 张量，像素范围为 [0,1]。
2. `random_split`：从官方训练数据划出 54,000 张训练图片和 6,000 张验证图片。官方测试数据另有 10,000 张。不要用测试集选择参数。
3. `DataLoader`：每批取 128 张训练图片，训练顺序打乱；`num_workers=0` 便于在 Windows 上理解和运行。
4. `make_model`：把每张 28×28 的灰度图展平成 784 个数，经过 256、128 个隐藏单元，输出 10 个类别分数。ReLU 引入非线性。这一步先不使用 CNN。
5. `CrossEntropyLoss`：接受原始分数 logits 和整数类别标签；不要事先对 logits 做 softmax。
6. `zero_grad → 前向传播 → loss → backward → step`：分别清梯度、预测、计算损失、求梯度、更新参数。
7. `evaluate`：用 eval 切换运行模式，用 no_grad 关闭梯度记录。此模型没有 Dropout/BatchNorm，但保留正确的通用流程。
8. `torch.save`：保存参数。加载时需要先创建同样结构的模型，再调用 load_state_dict。

## 看懂结果

- `outputs/metrics.json`：每轮训练与验证损失、准确率，最后的测试结果、运行时间和显存记录。
- `outputs/learning_curves.png`：损失和准确率随训练轮次的变化。
- `outputs/predictions.png`：测试集前 12 张图片的真实类别与预测，红字表示预测错误。图片不是随机选出的代表性样本，不应用它替代整体评估。
- `outputs/predictions.json`：图片索引、类别和 softmax 分数。分数不等于经过校准的正确概率。
- `outputs/model.pt`：保存的模型参数。

训练准确率在参数不断更新的过程中统计；验证准确率由该轮结束后的模型计算，因此两者并非完全相同口径。
显存数字是 PyTorch 张量的峰值已分配显存，不是整机或整个进程的显存占用。
运行时间包括下载、训练、评估和部分结果输出；不应当作纯训练速度。
设置随机种子有助复现，但不保证跨软件版本、硬件或所有 GPU 算子的逐位一致。

## 学完自查

1. 为什么图片是 `[128,1,28,28]`，输出是 `[128,10]`？
2. `loss.backward()` 是否已经更新了参数？
3. 为什么每批都需要清梯度？
4. 验证阶段为什么不调用 optimizer.step？
5. 模型预测错误，是不是一定代表代码有 bug？

官方入门参考：https://docs.pytorch.org/tutorials/beginner/basics/intro
