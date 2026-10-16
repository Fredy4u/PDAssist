# PD 脑 FDG-PET 研究

## 已下载的数据

`data/cnpet/` 是 NITRC 的中国正常人 FDG-PET 模板和 116 例正常受试者归一化影像。它是 PET 流程开发的真实影像输入，但没有 PD 组，因此只适合检查 NIfTI 读入、空间配准、掩膜和归一化步骤，不能检验 PD 相关假设或训练 PD 分类器。[NITRC 数据页](https://www.nitrc.org/projects/cnpet/)

## 已准备的实验

运行合成流程，检查 SSM/PCA、按受试者分组交叉验证和 PDRP 类表达计算能否贯通：

```powershell
python scripts/pet_experiment.py
```

该实验以真实正常 PET 模板作为背景，但所有受试者影像和标签均为模拟数据，不能把分数或模拟图案解释为临床 PD 结果。

若拿到获批的 PD 与健康对照影像，将已配准至同一模板空间的影像路径整理为 CSV（一行一个扫描，列名 `subject_id,label,path`，HC=0、PD=1），再运行：

```powershell
python scripts/pet_experiment.py --manifest scans.csv --mask brain_mask.nii.gz --out outputs/real_pet
```

影像与独立确定的掩膜必须具有相同体素网格和 affine。脚本按受试者分组交叉验证，并在每个训练折内拟合降维与模型，避免同一受试者多次扫描跨训练/测试折。
