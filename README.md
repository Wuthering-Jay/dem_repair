# dem_repair_project

`dem_repair_project` 是一个面向“基于多尺度开度 + 条件 GAN 的植被密集山区 LiDAR 空洞 DEM 修复”的 Python + PyTorch 工程骨架。

当前阶段只完成工程初始化，不实现复杂算法。目标是先把目录、配置、脚本入口和核心模块占位搭好，便于后续逐步实现监督版 DEM 修复、地形特征构建、GAN 训练与推理流程。

## 项目简介

项目面向以下数据与任务：

- 输入数据：
  - LAS 1.2 激光点云，包含坐标、`intensity`、`return number`、`classification`
  - DEM 栅格，可作为 `gt_dem` 参考基准
  - DSM 栅格，可作为条件输入，但不能当作 DEM 真值
- 目标任务：
  - 在植被密集山区场景下，构建 LiDAR 空洞区域的 DEM 修复流程
  - 先实现监督学习基线，再扩展到条件 GAN
  - 后续引入 slope、curvature、openness、ridge/valley 等结构约束

## 目录说明

- `configs/`：所有数据、模型、训练、推理参数的 YAML 配置
- `.agents/skills/`：面向本仓库的任务技能说明，约束样本构造、特征计算、指标评估和训练排障
- `data/raw/`：原始 LAS、DEM、DSM 数据的 train/val/test 分区
- `data/processed/`：栅格化、特征、样本和数据划分等中间产物
- `datasets/`：PyTorch 数据集和预处理变换
- `models/`：生成器、判别器、基础模块与结构头部占位
- `losses/`：DEM、对抗和结构损失函数占位
- `metrics/`：DEM 与结构评估指标占位
- `utils/`：IO、LAS、栅格、掩膜、地形特征、可视化、配置等通用工具
- `scripts/`：数据构建、检查、可视化与导出脚本入口
- `outputs/`：训练输出、日志、图像和预测结果
- `train.py` / `val.py` / `infer.py`：训练、验证、推理 CLI 入口

## 当前状态

- 已完成：项目骨架初始化
- 未完成：样本构建逻辑、真实模型结构、损失函数实现、训练循环、推理流程

## 后续开发计划

1. 实现 `scripts/build_rasters.py`，完成 LAS 到规则栅格的基础预处理
2. 实现 `utils/terrain_features.py`，支持 slope / curvature / openness 计算
3. 实现 `scripts/build_samples.py` 和 `datasets/dem_repair_dataset.py`
4. 实现监督版生成器与基础 DEM 重建损失
5. 在监督版稳定后，再实现条件 GAN 判别器与对抗训练

## 快速开始

```bash
pip install -r requirements.txt
python train.py --config configs/train/train_supervised.yaml
python val.py --config configs/train/train_supervised.yaml
python infer.py --config configs/infer/infer.yaml
```

当前这些入口只会打印 `not implemented yet`，用于验证工程结构和 CLI 已接通。
