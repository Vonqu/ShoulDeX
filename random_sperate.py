import os
import random
import shutil

def split_csv_files(source_dir, dest_dir_a, dest_dir_b, ratio=0.8):
    # 创建目标文件夹
    os.makedirs(dest_dir_a, exist_ok=True)
    os.makedirs(dest_dir_b, exist_ok=True)

    # 获取所有 CSV 文件
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    total_files = len(csv_files)
    print(f"总共找到 {total_files} 个 CSV 文件。")

    # 随机打乱并划分
    random.shuffle(csv_files)
    split_index = int(ratio * total_files)
    files_a = csv_files[:split_index]
    files_b = csv_files[split_index:]

    # 拷贝文件到目标文件夹
    for file in files_a:
        shutil.copy(os.path.join(source_dir, file), os.path.join(dest_dir_a, file))
    for file in files_b:
        shutil.copy(os.path.join(source_dir, file), os.path.join(dest_dir_b, file))

    print(f"已将 {len(files_a)} 个文件放入 '{dest_dir_a}'，{len(files_b)} 个文件放入 '{dest_dir_b}'。")

# 使用示例
if __name__ == "__main__":
    source_folder = "F:\coding\ShoulDeX\ShoulDex_v1.0\ShoulDeX\data\dft\qnc\mixed"
    target_folder_a = "F:\coding\ShoulDeX\ShoulDex_v1.0\ShoulDeX\data\dft\qnc\mixed\\train"
    target_folder_b = "F:\coding\ShoulDeX\ShoulDex_v1.0\ShoulDeX\data\dft\qnc\mixed\\test"
    split_ratio = 0.8  # 80% 到 A，20% 到 B

    split_csv_files(source_folder, target_folder_a, target_folder_b, split_ratio)
