import os
import sys


def add_prefix_to_csv_files(folder_path, prefix):
    if not os.path.isdir(folder_path):
        print(f"指定的文件夹不存在: {folder_path}")
        return
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv') and not filename.startswith(prefix):
            old_path = os.path.join(folder_path, filename)
            new_filename = prefix + filename
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_filename}")


def main():
    # 在这里手动设置你的文件夹路径和前缀
    participant_id = 'qnc'
    motion_type = 'compensatory'
    folder_path = f"F:\coding\ShoulDeX\ShoulDex_v1.0\ShoulDeX\data\dft_final\{participant_id}\{motion_type}"  # 修改为你的目标文件夹路径
    prefix = "P1_"  # 修改为你想要的前缀
    
    print(f"目标文件夹: {folder_path}")
    print(f"添加前缀: {prefix}")
    
    # 确认是否继续
    confirm = input("确认要执行重命名操作吗？(y/n): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return
    
    add_prefix_to_csv_files(folder_path, prefix)
    print("重命名完成！")


if __name__ == "__main__":
    main()
