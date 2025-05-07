import os
import json
import torch
import clip
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 定义文件存储路径
UPLOAD_DIR = "uploaded_files"
VIDEOS_DIR = "videos"
HASH_CODE_IMAGE_FILE = "hash_codes_p.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# 使用CLIP提取图片特征
def extract_image_features(image_path):
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    # 加载并预处理图像
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # 获取图像特征
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    image_features = image_features.cpu().numpy().flatten()  # 转为numpy数组
    return image_features

# 生成图片哈希码
def generate_image_hash_code(image_path):
    image_features = extract_image_features(image_path)
    return image_features

# 保存图片文件的哈希码到文件
def save_image_hash_code(image_path, hash_code, hash_file=HASH_CODE_IMAGE_FILE):
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            hash_table = json.load(f)
    else:
        hash_table = {}

    filename = os.path.basename(image_path)
    hash_table[filename] = hash_code.tolist()  # 保存为列表

    with open(hash_file, 'w') as f:
        json.dump(hash_table, f, indent=4)

# 加载图片文件的哈希码
def load_image_hash_codes(hash_file=HASH_CODE_IMAGE_FILE):
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return json.load(f)
    else:
        return {}

# 计算哈希码的余弦相似度（图片）
def calculate_image_cosine_similarity(query_hash, hash_table):
    similarities = {}
    
    for filename, stored_hash in hash_table.items():
        stored_hash = np.array(stored_hash)  # 确保是一个numpy数组
        similarity = cosine_similarity([query_hash], [stored_hash])[0][0]
        similarities[filename] = similarity

    return similarities

# 从最相似帧文件名生成视频文件名
def get_video_filename_from_image(image_filename):
    # 假设图片文件名的格式为 group_x_frame_0000.png
    try:
        # 提取 'group_x' 部分
        parts = image_filename.split('_')
        if len(parts) >= 2 and parts[0] == "group":
            # 生成对应的视频文件名
            video_filename = f"video_{parts[1]}.mp4"  # video_x.mp4
            return video_filename
        else:
            raise ValueError(f"Invalid filename format: {image_filename}. Expected 'group_x_frame_0000.png'.")
    except Exception as e:
        print(f"Error: {e}")
        return None

# 路由：主页
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("files")
        file_hashes = []

        for file in files:
            if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                # 保存上传的图片文件
                file_location = os.path.join(UPLOAD_DIR, file.filename)
                file.save(file_location)

                # 生成图片哈希码并保存
                hash_code = generate_image_hash_code(file_location)
                save_image_hash_code(file_location, hash_code)

                file_hashes.append({
                    'filename': file.filename,
                    'hash_code': hash_code.tolist()
                })

        # 查找最相似的图片
        image_hash_table = load_image_hash_codes(HASH_CODE_IMAGE_FILE)

        # 假设查询的是最后上传的图片文件
        query_hash = file_hashes[-1]['hash_code']
        similarities = calculate_image_cosine_similarity(query_hash, image_hash_table)

        # 找到相似度最高的图片文件
        most_similar_image = max(similarities, key=similarities.get)

        # 根据最相似图片的文件名生成视频文件名
        video_filename = get_video_filename_from_image(most_similar_image)

        return render_template("p-search.html", most_similar_video=video_filename, most_similar_video_name=video_filename)

    return render_template("p-search.html")

# 提供已上传的图片文件
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# 提供视频文件
@app.route('/videos/<filename>')
def video(filename):
    return send_from_directory(VIDEOS_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
