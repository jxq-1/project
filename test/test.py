import os
import json
import torch
import librosa
import clip
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

# 定义文件存储路径
UPLOAD_DIR = "uploaded_files"
VIDEOS_DIR = "videos"
HASH_CODE_IMAGE_FILE = "hash_codes_p.json"
HASH_CODE_AUDIO_FILE = "hash_codes_a.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# 使用CLIP提取图片特征
def extract_image_features(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)

    image_features = image_features.cpu().numpy().flatten()
    return image_features

# 生成图片哈希码
def generate_image_hash_code(image_path):
    image_features = extract_image_features(image_path)
    return image_features

# 提取音频特征（MFCC）
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = mfccs.T
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs)
    return mfccs_scaled

# 生成音频哈希码
def generate_audio_hash_code(file_path):
    mfccs_scaled = extract_audio_features(file_path)
    hash_code = itq(mfccs_scaled)
    return hash_code.flatten()

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

# 保存哈希码到文件
def save_hash_code(file_path, hash_code, hash_file):
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            hash_table = json.load(f)
    else:
        hash_table = {}

    filename = os.path.basename(file_path)
    hash_table[filename] = hash_code.tolist()

    with open(hash_file, 'w') as f:
        json.dump(hash_table, f, indent=4)

# 加载哈希码
def load_hash_codes(hash_file):
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return json.load(f)
    else:
        return {}

# 计算余弦相似度
def calculate_cosine_similarity(query_hash, hash_table):
    similarities = {}
    
    for filename, stored_hash in hash_table.items():
        stored_hash = np.array(stored_hash)
        similarity = cosine_similarity([query_hash], [stored_hash])[0][0]
        similarities[filename] = similarity

    return similarities

# 从最相似帧文件名生成视频文件名
def get_video_filename_from_image(image_filename):
    try:
        parts = image_filename.split('_')
        if len(parts) >= 2 and parts[0] == "group":
            video_filename = f"video_{parts[1]}.mp4"  # 生成 video_x.mp4
            return video_filename
        else:
            raise ValueError(f"Invalid filename format: {image_filename}. Expected 'group_x_frame_0000.png'.")
    except Exception as e:
        print(f"Error: {e}")
        return None

# 主页路由
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("files")
        file_hashes = []
        file_type = None

        for file in files:
            if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                # 处理图片
                file_location = os.path.join(UPLOAD_DIR, file.filename)
                file.save(file_location)
                hash_code = generate_image_hash_code(file_location)
                save_hash_code(file_location, hash_code, HASH_CODE_IMAGE_FILE)
                file_hashes.append({'filename': file.filename, 'hash_code': hash_code.tolist()})
                file_type = 'image'

            elif file and file.filename.lower().endswith(('mp3', 'wav', 'flac')):
                # 处理音频
                file_location = os.path.join(UPLOAD_DIR, file.filename)
                file.save(file_location)
                hash_code = generate_audio_hash_code(file_location)
                save_hash_code(file_location, hash_code, HASH_CODE_AUDIO_FILE)
                file_hashes.append({'filename': file.filename, 'hash_code': hash_code.tolist()})
                file_type = 'audio'

        if file_type == 'image':
            # 图片检索
            image_hash_table = load_hash_codes(HASH_CODE_IMAGE_FILE)
            query_hash = file_hashes[-1]['hash_code']
            similarities = calculate_cosine_similarity(query_hash, image_hash_table)
            most_similar_image = max(similarities, key=similarities.get)
            video_filename = get_video_filename_from_image(most_similar_image)

        elif file_type == 'audio':
            # 音频检索
            audio_hash_table = load_hash_codes(HASH_CODE_AUDIO_FILE)
            query_hash = file_hashes[-1]['hash_code']
            similarities = calculate_cosine_similarity(query_hash, audio_hash_table)
            most_similar_audio = max(similarities, key=similarities.get)
            video_filename = most_similar_audio.replace('.mp3', '.mp4').replace('.wav', '.mp4').replace('.flac', '.mp4')

        return render_template("index.html", most_similar_video=video_filename, most_similar_video_name=video_filename)

    return render_template("index.html")

# 提供视频文件
@app.route('/videos/<filename>')
def video(filename):
    return send_from_directory(VIDEOS_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
