import cv2
import os
import glob

def video_to_frames(video_path, output_root=None):
    """
    将视频转换为同名文件夹下的图片序列。
    """
    # 1. 获取视频文件名（不带后缀）
    filename = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(filename)[0]
    
    # 2. 确定输出路径
    # 如果没有指定输出根目录，默认在视频所在目录下创建文件夹
    if output_root is None:
        output_dir = os.path.join(os.path.dirname(video_path), video_name_no_ext)
    else:
        output_dir = os.path.join(output_root, video_name_no_ext)
        
    # 创建文件夹（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建文件夹: {output_dir}")
    else:
        print(f"文件夹已存在，图片将保存至: {output_dir}")

    # 3. 读取视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"开始处理: {filename} (共 {total_frames} 帧)...")

    while True:
        ret, frame = cap.read()
        
        # 如果读取不到帧（视频结束），则退出循环
        if not ret:
            break
        
        # 4. 保存图片
        # 使用 frame_0001.jpg 这种格式，方便排序
        img_name = f"frame_{frame_count:05d}.jpg"
        save_path = os.path.join(output_dir, img_name)
        
        cv2.imwrite(save_path, frame)
        
        frame_count += 1
        
        # 可选：打印进度 (每100帧打印一次)
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧", end='\r')

    cap.release()
    print(f"\n完成！已保存 {frame_count} 张图片到 {output_dir}\n")

# --- 主程序 ---
if __name__ == "__main__":
    # 设置包含 mp4 的文件夹路径 ( '.' 代表当前文件夹)
    source_folder = '/home/sunqi/code/projects/SDS-Bridge/2D_experiments/results/bridge_gen/a_DSLR_photo_of_a_cat_lr75.000_seed0_scale40.0_nsteps500/' 
    
    # 查找所有 mp4 文件
    mp4_files = glob.glob(os.path.join(source_folder, "*.mp4"))
    
    if not mp4_files:
        print("当前目录下未找到 .mp4 文件。")
    else:
        for mp4 in mp4_files:
            video_to_frames(mp4)