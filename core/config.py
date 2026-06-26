import os
import sys

if getattr(sys, 'frozen', False):
    PROJECT_ROOT = sys._MEIPASS
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

KB_PATH = os.path.join(DATA_DIR, "knowledge_base.json")
W2V_MODEL_PATH = os.path.join(MODEL_DIR, "word2vec.pth")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")

EMBED_DIM = 128
WINDOW_SIZE = 3
MIN_COUNT = 2
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.001
TOP_K = 5

CATEGORIES = ["科技", "体育", "娱乐", "教育", "生活", "文化"]
