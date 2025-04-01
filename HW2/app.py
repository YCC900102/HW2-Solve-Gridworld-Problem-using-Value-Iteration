from flask import Flask, render_template, request, jsonify
import os
import json
from utils import policy_iteration  # 改為使用 policy_iteration

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    grid_size = None
    if request.method == "POST":
        try:
            grid_size = int(request.form["grid_size"])
            if not (5 <= grid_size <= 9):
                grid_size = None
        except ValueError:
            grid_size = None
    return render_template("index.html", grid_size=grid_size)

@app.route("/save_map", methods=["POST"])
def save_map():
    data = request.get_json()
    with open("map_data.json", "w") as f:
        json.dump(data, f)
    return jsonify({"message": "地圖儲存成功！"})

@app.route("/evaluate", methods=["GET"])
def evaluate():
    with open("map_data.json", "r") as f:
        data = json.load(f)

    grid_size = data["size"]
    start = data["start"]
    end = data["end"]
    obstacles = data["obstacles"]

    V, policy = policy_iteration(grid_size, start, end, obstacles)

    value_matrix = [[round(float(v), 2) for v in row] for row in V]
    policy_matrix = policy  # 已經是 symbol 格式

    return jsonify({
        "value_matrix": value_matrix,
        "policy_matrix": policy_matrix
    })

if __name__ == "__main__":
    app.run(debug=True)
