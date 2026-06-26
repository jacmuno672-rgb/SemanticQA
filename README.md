# 中文语义相似度问答系统

基于 Word2Vec 词向量的语义相似度中文问答系统，使用 PyTorch 实现 CBOW 模型训练，Flask 构建 Web 交互界面。

## 快速开始

### 方式一：直接运行可执行文件（推荐）

**Windows：** 双击 `dist/QA_System.exe`，浏览器会自动打开

**Linux：** 终端执行 `dist/问答系统`，浏览器会自动打开

程序启动后，浏览器访问 `http://127.0.0.1:5000` 即可使用。

### 方式二：源码运行

```bash
pip install -r requirements.txt
python app.py
```

## 项目结构

```
FinalProject/
├── app.py                   # Flask Web 入口
├── build_system.py          # 一键构建（数据→词向量→知识库）
├── download_data.py         # 数据集和训练语料
├── requirements.txt         # 依赖清单
├── core/
│   ├── config.py            # 全局配置
│   ├── preprocess.py        # jieba 分词 + 词汇表
│   ├── word2vec_trainer.py  # PyTorch CBOW 训练
│   ├── vectorizer.py        # IDF 加权句向量
│   ├── similarity.py        # 余弦相似度计算
│   ├── knowledge_base.py    # 知识库构建与持久化
│   └── qa_engine.py         # 问答引擎
├── templates/chat.html      # 对话界面
├── static/style.css         # 界面样式
├── data/                    # 知识库 JSON 和原始语料
├── models/                  # 训练好的词向量和词汇表
└── dist/                    # 可执行文件
```

## 技术栈

- **词向量模型**：PyTorch CBOW，128 维
- **中文分词**：jieba
- **句向量**：IDF 加权词向量平均
- **相似度**：余弦相似度
- **Web 框架**：Flask
- **数据**：182 条 QA 对，6 个类别（科技/体育/娱乐/教育/生活/文化）

## 测试示例

| 测试问题 | 匹配问题 | 相似度 |
|---------|---------|--------|
| 什么是深度学习？ | 什么是深度学习？ | 1.000 |
| 电脑怎么能看懂图片？ | 电脑怎么识别图片？ | 0.919 |
| 踢球的比赛多久一次？ | 足球赛多久办一次？ | 0.730 |
| 怎么选大学专业？ | 大学专业如何选择？ | 0.769 |

---

*自然语言处理期末项目 — 基于词向量的语义相似度问答系统*
