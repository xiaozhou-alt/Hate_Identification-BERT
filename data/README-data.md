## 说明

ps：此处是对于数据集处理的总体说明

比赛中下载到的原始数据集为：

```c
data.json      # 原始训练集
test1.json     # 初赛测试集
test2.json     # 复赛测试集
demo.txt       # 结果提交示例文档，此处未给出
```

*data.ipynb* 用于数据集的基础统计和划分，其中 *train.json* 为训练集（3200条），*test.json* 为测试集（800条）划分自 *data.json* 

*cut.ipynb* 用于 **HanLP** 库在数据集上的语义划分、情感分析、关键词提取的展示

