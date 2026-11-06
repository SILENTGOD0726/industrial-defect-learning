# 工业缺陷检测与模型压缩：学习与实验记录

一份面向初学者的学习仓库，汇总工业表面缺陷检测、结构化剪枝、知识蒸馏与边缘部署的阅读笔记，以及在真实数据上完成的基础实验。

## 从这里开始

1. [15 篇论文学习报告 PDF](reports/15-papers/学习报告.pdf)｜[交互学习网页](reports/15-papers/交互学习报告.html)｜[证据与图表数据](reports/15-papers/证据与图表/)
2. [六篇工业表面缺陷论文精读报告](reports/6-papers/六篇精读报告.md)｜[PDF](reports/6-papers/六篇精读报告.pdf)
3. [DepGraph 剪枝实验：说明、代码与结果](experiments/depgraph/README.md)
4. [FashionMNIST 入门实验](experiments/fashionmnist/README.md)
5. [六篇论文组会汇报 PPT v2](presentations/六篇工业缺陷文献_组会汇报_v2.pptx)
6. [英文综述 v0.4](review-draft/review_draft_v04.md)｜[审稿与核验记录](review-draft/reviews/2026-10-15-manuscript-v04-adversarial-review.md)

GitHub 上的 HTML 文件以源码形式预览；下载仓库后在浏览器中打开可使用交互功能。参考文献中的原论文请通过 DOI 或出版社页面获取。

## 一个实际实验：DepGraph

使用作者预训练的 ResNet-56，在 CIFAR-10 的同一 10,000 张测试图像上：

| 阶段 | 准确率 | 参数量 |
|---|---:|---:|
| 原始模型 | 93.53% | 855,770 |
| 刚剪枝 | 15.26% | 562,878 |
| 微调 1 轮 | 86.74% | 562,878 |

参数减少 34.23%，单张图像 MAC 口径计算量减少 52.82%。保存模型后经独立进程重载，测试准确率一致。配置与原始记录见[实验结果](experiments/depgraph/run_1/学习结果.md)及 [JSON](experiments/depgraph/run_1/final_info.json)。

这是一轮学习性流程验证，尚未完成论文全部实验；CIFAR-10 是普通图片分类数据，不能将其结果视为工业缺陷检测结果。没有测量实际推理加速，也没有验证 RK3568 部署。

## 状态和引用

- 英文综述为工作稿，现有评审建议大修，不应视为投稿定稿。六篇汇报是 v2，仍有后续修改空间。
- 论文原文和数据集不随仓库分发。报告中的论文数据均需结合原文数据集、设备与测试条件理解；不同论文的 mAP/FPS 不适宜直接排名。
- 部分报告包含论文原表/公式摘图并标注来源；论文、图片与数据归各自权利人所有，使用时请查看原文许可证。本人整理内容及作者第三方代码的权利范围各自独立，仓库公开可读不代表全部材料采用同一开源许可证。
- DepGraph 所使用的第三方模型代码保留了原始作者注释和[许可文件](experiments/depgraph/AUTHOR_LICENSE.txt)。
