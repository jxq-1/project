import cv2
import os

def extract_frames(video_path, output_dir, target_fps=1):
    """
    从视频中提取帧并保存到指定目录。
    
    参数:
        video_path (str): 输入视频文件路径。
        output_dir (str): 输出帧图像的存储目录。
        target_fps (int): 每秒提取的目标帧数（默认为1帧/秒）。
    """
    # 检查视频文件路径是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件 {video_path} 不存在！")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    # 获取视频原始帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print("错误: 无法获取视频帧率，可能视频文件已损坏。")
        return
    print(f"视频原始帧率: {original_fps} fps")

    # 计算帧间隔（原始帧率和目标帧率的比值）
    frame_interval = int(original_fps / target_fps)

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("视频读取完成或遇到错误。")
            break

        # 根据帧间隔保存帧
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.png")
            success = cv2.imwrite(output_path, frame)
            if success:
                print(f"保存帧: {output_path}")
                saved_frame_count += 1
            else:
                print(f"错误: 无法保存帧到 {output_path}")

        frame_count += 1

    cap.release()
    print(f"提取完成！总共提取了 {saved_frame_count} 帧，保存到目录: {output_dir}")


# 主程序
if __name__ == "__main__":
    video_path = r"C:\Users\DELL\Desktop\Test\repos\DATA\shared\video\files\5AaELleGHcI\5AaELleGHcI_5_190.mp4"  # 替换为您的视频路径
    output_dir = r"C:\Users\DELL\Desktop\Test\output_frames"  # 替换为您的输出帧存储目录
    target_fps = 1  # 每秒提取的帧数，改为 1 帧/秒

    extract_frames(video_path, output_dir, target_fps)
