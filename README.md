# ml-technical-term
机器学习和深度学习领域的专业术语汇总，包括中英文对照。

**表1 基本理论与概念**

| EN                       | ZH             | 全称/缩写                       | 备注                                                         |
| ------------------------ | -------------- | ------------------------------- | ------------------------------------------------------------ |
| AI                       | 人工智能       | Artificial Intelligence         |                                                              |
| CV                       | 计算机视觉     | Computer Vision                 | 计算机视觉是人工智能领域的一个重要分支。                     |
| AGI                      | 通用人工智能   | Artificial General Intelligence |                                                              |
| zero-shot                | 零样本         |                                 | 零样本学习                                                   |
| few-shot                 | 少样本         |                                 | 少样本学习                                                   |
| SOTA                     | 最佳性能       | state-of-the-art                | 指模型得到最佳效果                                           |
| Benchmark                | 基准           |                                 | 基准模型，一般是同行中认可度比较高和效果好的算法。           |
| Baseline                 | 基线           |                                 | 一般指算法的初始版本。这个名词在很多的竞赛中很常见，一般都是在此基础上进行‘“魔改”，以它为标准，来判断改进的好坏。 |
| LM                       | 语言模型       | Language Model                  |                                                              |
| ML                       | 机器学习       | Mechine Learning                |                                                              |
| Supervised Learning      | 监督学习       |                                 |                                                              |
| Unsupervised Learning    | 无监督学习     |                                 |                                                              |
| Semi-supervised Learning | 半监督学习     |                                 |                                                              |
| RL                       | 强化学习       | Reinforcement Learning          |                                                              |
| DL                       | 深度学习       | Deep Learning                   |                                                              |
| DDP                      | 分布式数据并行 | Distributed Data Parallel       | 一种常见的分布式训练方法                                     |

**表2 NLP 领域术语**

| EN                                        | ZH                       | 全称/缩写                                    | 备注                                                         |
| ----------------------------------------- | ------------------------ | -------------------------------------------- | ------------------------------------------------------------ |
| NLP                                       | 自然语言处理             | Natural Language Processing                  | 自然语言处理是人工智能领域的一个重要分支。                   |
| NLU                                       | 自然语言理解             | Neural Language Understanding                |                                                              |
| NLG                                       | 自然语言生成             | Neural Language Generation                   |                                                              |
| Transformer                                       | Transformer 中文为“变换器”，但不够达雅，日常还是称英文               |                          |                                                              |
| LLM                                       | 大语言模型              | Large Language Model                         | 狭义上也常称为“大模型”                                                            |
| GPT                                       | 生成式预训练Transformer模型               | Generative Pre-trained Transformer                         |                                                              |
| SFT                                       | 监督微调                 | Supervised Fine-Tuning                       |                                                              |
| LoRA                                      | 大型语言模型的低秩自适应 | Low-Rank Adaptation of Large Language Models | 是一种降低模型可训练参数，又尽量不损失模型表现的大模型微调方法。 |
| RLHF                                      | 人类反馈的强化学习       | Reinforcement Learning from Human Feedback   |                                                              |
| PEFT                                      | 参数高效微调             | Parameter-Efficient Fine-Tuning              |                                                              |
| RAG                                       | 检索增强生成             | Retrieval Augmented Generation               |                                                              |
| IE                                        | 信息抽取                 | Information Extraction                       |                                                              |
| NER                                       | 命名实体识别             | Named Entity Recognition                     |                                                              |
| RE                                        | 关系抽取                 | Relationship Extraction                      |                                                              |
| Machine Translation                       | 机器翻译                 |                                              |                                                              |
| Word Segmentation                         | 分词                     |                                              |                                                              |
| POS                                       | 词性标注                 | Part-of-speech Tagging                       |                                                              |
| Pretrained                                       | 预训练                 |                        |                                                              |
| Pre-trained Language Representation Model | 预训练语言表示模型       |                                              |                                                              |
| fine tuning                               | 微调                     |                                              |                                                              |

**表3 算法模型**

| EN                  | ZH                          | 全称/缩写                               | 备注                                             |
| ------------------- | --------------------------- | --------------------------------------- | ------------------------------------------------ |
| MLP                 | 多层感知机                  | Multilayer Perceptron                   |                                                  |
| NN                  | 神经网络                    | Neural Network                          |                                                  |
| Activation Function | 激活函数                    |                                         |                                                  |
| BiRNN               | 双向循环神经网络            | Bidirectional Recurrent Neural Networks |                                                  |
| ReLU                | 线性修正单元                | Rectified Linear Unit                   | 激活函数。                                       |
| ResNet              | 残差网络                    | Residual Network                        |                                                  |
| Cross Validation    | 交叉验证                    | CV                                      | 是在机器学习建立模型和验证模型参数时常用的方法。 |
| Softmax             |                             |                                         | Softmax函数是将一个实数向量转换为一个概率分布。  |
| GRU                 | 门控循环单元                | Gated Recurrent Unit                    |                                                  |
| SVM                 | 支持向量机                  | Support Vector Machine                  |                                                  |
| KNN                 | K-近邻算法                  | K-Nearest Neighbors                     |                                                  |
| LR                  | 线性回归                    | Linear Regression                       |                                                  |
| LR                  | 逻辑回归                    | Logistic Regression                     |                                                  |
| DT                  | 决策树                      | Decision Tree                           |                                                  |
| Random Forest       | 随机森林                    |                                         |                                                  |
| PCA                 | 主成分分析                  | Principal Component Analysis            |                                                  |
| CNN                 | 卷积神经网络                | Convolutional Neural Network            |                                                  |
| RNN                 | 循环神经网络                | Recurrent Neural Network                |                                                  |
| LSTM                | 长短期记忆网络              | Long Short-Term Memory                  |                                                  |
| GPT                 | 生成式预训练Transformer模型 | Generative Pre-trained Transformer      |                                                  |



### 参考

[深度学习的57个专业术语 - 曹明 - 博客园 (cnblogs.com)](https://www.cnblogs.com/think90/p/7080251.html)

[带你了解ICCV、ECCV、CVPR三大国际会议_eccv是什么级别-CSDN博客](https://blog.csdn.net/m0_46988935/article/details/109378535)

[【深度学习常见术语解释（更新中...）】 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/603815281)
