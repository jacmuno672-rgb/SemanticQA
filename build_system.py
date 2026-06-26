import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.preprocess import load_raw_texts, tokenize_all, build_vocab, tokenize
from core.word2vec_trainer import train_word2vec
from core.knowledge_base import KnowledgeBase
from core.vectorizer import Vectorizer
from core.config import RAW_DATA_DIR, MIN_COUNT
from download_data import download_data

import json


def main():
    print("=" * 60)
    print("  中文语义相似度问答系统 — 一键构建")
    print("=" * 60)

    print("\n[1/5] 准备数据...")
    download_data(RAW_DATA_DIR)

    print("\n[2/5] 加载语料和分词...")
    texts = load_raw_texts(RAW_DATA_DIR)
    tokenized = tokenize_all(texts)
    word2idx, _ = build_vocab(tokenized, MIN_COUNT)
    vocab_size = len(word2idx)
    print(f"  词汇表大小: {vocab_size}")

    print("\n[3/5] 训练 Word2Vec...")
    train_word2vec(tokenized, word2idx, vocab_size)

    print("\n[4/5] 构建知识库...")
    qa_path = os.path.join(RAW_DATA_DIR, "qa_pairs.json")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    vectorizer = Vectorizer()
    vectorizer.load()

    questions = [item["question"] for item in qa_pairs]
    vectorizer.compute_idf(questions)

    kb = KnowledgeBase()
    kb.build(qa_pairs, vectorizer)

    print("\n[5/5] 测试检索...")
    from core.qa_engine import QAEngine
    engine = QAEngine()
    if engine.init():
        test_questions = [
            "什么是深度学习？",
            "世界杯多久举办一次？",
            "怎么提高学习成绩？",
            "春节有什么习俗？",
        ]
        for q in test_questions:
            result = engine.ask_single(q)
            print(f"\n  问: {q}")
            print(f"  答: {result['answer'][:60]}...")
            print(f"  匹配: {result['question'][:40]}... (相似度: {result['score']:.3f})")

    print("\n" + "=" * 60)
    print("  构建完成！运行 python app.py 启动对话界面")
    print("=" * 60)


if __name__ == "__main__":
    main()
