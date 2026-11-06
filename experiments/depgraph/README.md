# DepGraph：结构化剪枝学习实验

先阅读[中文结果](run_1/学习结果.md)和[参数、计算量对比图](run_1/Figure_1.png)。这次以 CIFAR-10 / ResNet-56 验证“预训练模型 → 剪枝 → 微调 → 重载”，没有完成论文全部结果。

作者代码：[Torch-Pruning](https://github.com/VainF/Torch-Pruning)，固定 v1.6.1，提交 `e80127d7c6935c319a7dd6719b8a72e3c1fcbff9`。`model.py` 来自该版本的 `reproduce/engine/models/cifar/resnet_tiny.py`，原文件注释和许可证已保留。本实验入口适配 Python 3.12，并不是作者原始复现脚本。

## Windows / NVIDIA GPU 环境示例

历史实验使用 Python 3.12.5、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、FP32、批量128，运行在 RTX 3080 Laptop GPU 上。下面需要 Git、网络及满足 CUDA 12.1 运行要求的 NVIDIA 驱动；安装前请按自己的硬件选择合适的 PyTorch 版本。

在本文件所在目录打开 PowerShell：

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
& .\.venv\Scripts\python.exe -m pip install numpy matplotlib "git+https://github.com/VainF/Torch-Pruning.git@e80127d7c6935c319a7dd6719b8a72e3c1fcbff9"
& .\.venv\Scripts\python.exe prepare_data.py
& .\.venv\Scripts\python.exe -u experiment.py --out_dir runs/my_first --epochs 1
& .\.venv\Scripts\python.exe -u experiment.py --out_dir runs/my_first --reload-only
& .\.venv\Scripts\python.exe plot.py --out_dir runs/my_first
& .\.venv\Scripts\python.exe summarize.py --out_dir runs/my_first
```

脚本会下载 CIFAR-10 和作者公开的预训练权重；数据归其原权利人所有。批量大小可以用 `--batch-size` 调整，改动会改变实验设置。新运行写入 `runs/`，请避免覆盖提供的 `run_1/`。

## 结果与局限

基线准确率 93.53%，剪枝后 15.26%，微调1轮后 86.74%；三个数字使用相同的 10,000 张测试图。参数量从 855,770 降至 562,878。详细设置和检查保存在 `run_1/`；公开版记录把本机绝对输出路径改成了相对路径。

只有一个随机种子、一次微调、未做稀疏训练；不能据此声称完整复现论文或推断统计显著性。计算量下降不是实测速度。`pruned_finetuned.pt` 是完整 PyTorch 对象，仅从可信来源加载；使用 `--reload-only` 时只读取自己信任的检查点。该公开仓库仅包含复现所需源码、历史结果和模型，不包含数据集或作者原始权重。
