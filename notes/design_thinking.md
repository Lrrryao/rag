---
title: medical_rag系统的选择
tags: rag
---

## 问题描述

最近在搞毕设，学习rag相关板块时，看到了一些可能的技术选择

### 关于模型

可以调用`大模型API`如chatgpt等, 当然也可以去huggingface 找到一些专用于医疗领域的小模型并部署自己的服务器上，当然也可以考虑二者一起

prompt压缩，关键词 lingua，这个是一个很好的优化机制，可以减少调用大模型api的token从而减少成本

关于多模态调用，这里并不需要额外的使用一些什么，因为现在大模型都已经包含了图片的识别；而对于小模型，可能还是有这个需求，那么届时可以让用户自己选择


### 思考
 
 重中之重的是 **基础**， 这里面说的什么 rerank模型，召回等，你都不知道在讲什么，满脑子想的都是去部署小模型还是调用大模型api? 这个不是最关键，最关键的结果！

 我认为工作的当前重点应该放在了解`rag系统的组成`：比如rag系统总共应该包含多少类模型，比如embedding，rerank等等， `如何提高回答的准确率上` 和 `如何利用agent提高性能/效率/准确率` 以及 `如何保证整个系统的并发性`， 
 
 但是话又说回来，这个系统预估将会是一个问答系统，分词/chunk/embedding 应该都是在后台训练，直接和用户交互的只是大模型本身

### 关于市面上的中文rag项目，应该怎么取舍

prompt:我想要写一个rag项目，市面常见的是基于中文的rag项目，然而我的毕设项目要求基于英语，那么如果想要基于这类中文的rag项目基础上改为英语RAG, 会有什么不同?
| 模块          | 中文RAG       | 英文RAG替换建议                    |
| ------------- | ------------- | ---------------------------------- |
| Embedding模型 | `bge-base-zh` | `all-MiniLM-L6-v2` 或 `e5-base-v2` |
| LLM模型       | `ChatGLM3-6B` | `Llama3-8B-Instruct` 或 `GPT-3.5`  |
| 分词器        | `jieba`       | `spaCy`                            |
| 分块器        | 正则分段      | `RecursiveCharacterTextSplitter`   |
| 检索器        | 向量检索      | 向量 + BM25混合检索                |
| 评估集        | C-MTEB        | MTEB / Natural Questions           |
| 评估工具      | Rouge-L       | RAGAS / BERTScore                  |

建议
| 选题方向        | 说明                                     |
| --------------- | ---------------------------------------- |
| **英文医疗RAG** | 用英文PubMed摘要构建知识库，回答医学问题 |

不要试图“改中文项目”，而是重新搭一个英文pipeline，用英文原生工具链（LlamaIndex + sentence-transformers + GPT-4），反而更快更稳。


### 关于输入输出类型

似乎需要对多模态类型的数据提供支持。 但是目前的重点仍然是 可用性+准确性 优先于一切

### 结论！！！！！

首先因为OCR模块本身已经占用了一定的时间，所以我认为RAG模块只需要在保证并发性的前提下，能够保证准确性（大概需要通过某些指标评估），可能再加上一点agent（optional） 

embedding模型和re-rank模型就直接选用开源 方便上手使用的即可，可以

对于rag的实现，我偏向于使用 VectorRAG+GraphRAG 尝试复现 [更强的RAG：向量数据库和知识图谱的合](https://www.cnblogs.com/hohoa/p/18456986)


Graph RAG具备以下局限性：

高质量图数据依赖：高质量的图数据对Graph RAG非常关键，如果处理出高质量的图数据有时很困难，特别是对于无结构的纯文本或标注质量不高的数据。
应用的复杂性：对于一个RAG系统，同时支持非结构化数据和图数据的混合检索，会增加检索系统设计和实现的复杂性。


不同RAG的比较：
![](../img/RAGs.png)


其中 `工业设计`+ `VectorRag` + `基本agent功能` 可以参考[How to Build a Production-Ready RAG AI Agent in Python](https://www.youtube.com/watch?v=AUQJ9eeP-Ls)，但建议先看这一集[课程的内容概述](https://www.bilibili.com/video/BV12hNue3EiE?spm_id_from=333.788.videopod.episodes&vd_source=356b2a99de74fd575902006e483d781c&p=2)

`GraphRAG` + `agent进阶功能` + `监控功能`可以参考[基于LangChain和知识图谱的大模型医疗问答机器人项目](https://www.bilibili.com/video/BV12hNue3EiE/?spm_id_from=333.788.videopod.episodes&vd_source=356b2a99de74fd575902006e483d781c&p=8)，该项目的源码可以在咸鱼上搜索 `LangChain和知识图谱` 即可买到


## 最终设计

多模态的输入识别 直接调用OCR即可，但是OCR可能需要应该一个专用的医学场景的模型和一个通用模型结合，这个我们后面看。但是有一点，**这个rag系统一定是输入为纯Txt文本的**，因为图片输入会先被OCR处理。

最终访问采用VectorRAG, 然后参考我在B站的RAG文件夹里的一些视频的观点，看看能否提高rag问答的分数（这个分数评估是一定要搞的）。然后这里感觉可以在问题的回答里采用agent结构，让其具有memory功能。最后再搞一个报告生成的多agent系统： 第一层agent获取全文上下文信息，第二层agent一个负责报告的文本生成，另一个负责图片的生成或者搜索，具体是靠aigc生成还是图片检索 后面再说（主要用于对陌生/罕见/偏僻的病名 添加症状图片，以方便用户理解）