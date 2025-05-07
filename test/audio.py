import os
import json
import librosa
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from sklearn.decomposition import PCA

app = Flask(__name__)

# 定义文件存储路径
UPLOAD_DIR = "uploaded_files"
HASH_CODE_AUDIO_FILE = "hash_codes_a.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ITQ 算法
def itq(matrix, n_iter=50):
    pca = PCA(n_components=min(matrix.shape[0], matrix.shape[1]))  
    matrix_pca = pca.fit_transform(matrix)

    random_state = np.random.RandomState(0)
    R = random_state.randn(matrix_pca.shape[1], matrix_pca.shape[1])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    for _ in range(n_iter):
        Z = matrix_pca @ R
        binary_code = np.sign(Z)
        C = binary_code.T @ matrix_pca
        U, _, Vt = np.linalg.svd(C)
        R = U @ Vt

    Z = matrix_pca @ R
    hash_matrix = (Z > 0).astype(int)
    return hash_matrix

# 提取MFCC特征
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = mfccs.T
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs)
    return mfccs_scaled

# 生成哈希码
def generate_hash_code(file_path):
    mfccs_scaled = extract_mfcc(file_path)
    hash_code = itq(mfccs_scaled)
    return hash_code.flatten()

# 保存音频文件的哈希码到文件
def save_hash_code(file_path, hash_code, hash_file="hash_codes_a.json"):
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            hash_table = json.load(f)
    else:
        hash_table = {}

    # 更改文件名为 video_0.mp4 格式
    filename = os.path.basename(file_path)
    new_filename = filename.replace("audio", "video").replace(".wav", ".mp4").replace(".mp3", ".mp4").replace(".flac", ".mp4")
    
    # 保存新的文件名和哈希码
    hash_table[new_filename] = hash_code.tolist()

    with open(hash_file, 'w') as f:
        json.dump(hash_table, f, indent=4)

# 路由：主页
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("files")
        file_hashes = []

        for file in files:
            if file and file.filename.endswith(('mp3', 'wav', 'flac')):
                # 保存上传的音频文件
                file_location = os.path.join(UPLOAD_DIR, file.filename)
                file.save(file_location)

                # 生成音频哈希码并保存
                hash_code = generate_hash_code(file_location)
                save_hash_code(file_location, hash_code)

                file_hashes.append({
                    'filename': file.filename,
                    'hash_code': hash_code.tolist()
                })

        return render_template("audio.html", file_hashes=file_hashes)

    return render_template("audio.html")

# 提供已上传的音频文件
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
