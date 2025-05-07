import os
import json
import librosa
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

app = Flask(__name__)

# 定义文件存储路径
UPLOAD_DIR = "uploaded_files"
VIDEOS_DIR = "videos"
HASH_CODE_AUDIO_FILE = "hash_codes_a.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

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

# 加载音频文件的哈希码
def load_hash_codes(hash_file="hash_codes_a.json"):
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return json.load(f)
    else:
        return {}

# 计算哈希码的余弦相似度
def calculate_cosine_similarity(query_hash, hash_table):
    similarities = {}
    
    for filename, stored_hash in hash_table.items():
        stored_hash = np.array(stored_hash)
        similarity = cosine_similarity([query_hash], [stored_hash])[0][0]
        similarities[filename] = similarity

    return similarities

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

        # 查找最相似的音频
        hash_table = load_hash_codes(HASH_CODE_AUDIO_FILE)

        # 假设查询的是最后上传的文件
        query_hash = file_hashes[-1]['hash_code']
        similarities = calculate_cosine_similarity(query_hash, hash_table)

        # 找到相似度最高的音频文件
        most_similar_file = max(similarities, key=similarities.get)
        
        # 查找对应的视频文件（假设视频文件名和音频文件名相同）
        video_filename = most_similar_file.replace('.mp3', '.mp4').replace('.wav', '.mp4').replace('.flac', '.mp4')

        # 提供视频的文件名供前端显示
        return render_template("a-search.html", file_hashes=file_hashes, most_similar_video=video_filename)

    return render_template("a-search.html")

# 提供已上传的音频文件
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# 提供视频文件
@app.route('/videos/<filename>')
def video(filename):
    return send_from_directory(VIDEOS_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
