import sys
import os
import webbrowser
import threading

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request, jsonify

from core.qa_engine import QAEngine

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

engine = QAEngine()
system_ready = engine.init()

if not system_ready:
    print("=" * 60)
    print("  警告：系统未初始化！请先运行: python build_system.py")
    print("=" * 60)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"})

    top_k = data.get("top_k", 3)
    results = engine.ask(question, top_k=top_k)
    return jsonify({"results": results})


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("\n  ========================================")
    print("    中文语义相似度问答系统")
    print("    正在启动服务...")
    print("  ========================================\n")
    threading.Timer(1.5, open_browser).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
