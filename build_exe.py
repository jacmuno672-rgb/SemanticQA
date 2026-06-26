import os
import sys
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIST_DIR = os.path.join(SCRIPT_DIR, "dist")
if os.path.exists(DIST_DIR):
    shutil.rmtree(DIST_DIR)

BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
if os.path.exists(BUILD_DIR):
    shutil.rmtree(BUILD_DIR)

SPEC_FILE = os.path.join(SCRIPT_DIR, "问答系统.spec")
if os.path.exists(SPEC_FILE):
    os.remove(SPEC_FILE)

cmd = (
    f'pyinstaller --onefile '
    f'--add-data="templates:templates" '
    f'--add-data="static:static" '
    f'--add-data="data:data" '
    f'--add-data="models:models" '
    f'--hidden-import core.config '
    f'--hidden-import core.preprocess '
    f'--hidden-import core.word2vec_trainer '
    f'--hidden-import core.vectorizer '
    f'--hidden-import core.similarity '
    f'--hidden-import core.knowledge_base '
    f'--hidden-import core.qa_engine '
    f'--hidden-import jieba '
    f'--hidden-import jieba.posseg '
    f'--hidden-import jieba.analyse '
    f'--hidden-import numpy '
    f'--hidden-import torch '
    f'--hidden-import sklearn '
    f'--hidden-import sklearn.metrics.pairwise '
    f'--hidden-import flask '
    f'--name "问答系统" '
    f'--noconsole '
    f'app.py'
)

print("=" * 60)
print("  开始打包问答系统...")
print("=" * 60)
print(f"\n执行命令:\n{cmd}\n")

result = os.system(cmd)
if result != 0:
    print(f"\n打包失败，错误码: {result}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  打包完成！")
print("=" * 60)

exe_path = os.path.join(SCRIPT_DIR, "dist", "问答系统")
if sys.platform == "win32":
    exe_path += ".exe"

if os.path.exists(exe_path):
    size_mb = os.path.getsize(exe_path) / (1024 * 1024)
    print(f"\n  可执行文件路径: {exe_path}")
    print(f"  文件大小: {size_mb:.1f} MB")
else:
    print(f"\n  [!] 未找到生成的可执行文件，请检查 dist/ 目录")
