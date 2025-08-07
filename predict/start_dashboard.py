#!/usr/bin/env python3
"""
角度预测系统启动脚本
直接启动核心功能，不进行库检查
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

def start_websocket_server():
    """启动WebSocket服务器"""
    print("[INFO] 启动WebSocket服务器...")
    
    # 启动WebSocket服务器进程
    server_process = subprocess.Popen([
        sys.executable, "websocket_server.py"
    ], cwd=Path(__file__).parent)
    
    return server_process

def open_frontend():
    """打开前端界面"""
    print("[INFO] 打开前端界面...")
    
    # 前端文件路径
    # frontend_path = Path(__file__).parent / "angle_prediction_dashboard.html"
    frontend_path = Path(__file__).parent / "angle_prediction_dashboard_v2.html"
    
    if frontend_path.exists():
        # 使用默认浏览器打开前端
        webbrowser.open(f'file://{frontend_path.absolute()}')
        print(f"[SUCCESS] 前端界面已打开: {frontend_path}")
    else:
        print(f"[ERROR] 前端文件不存在: {frontend_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("[INFO] 角度预测系统启动")
    print("=" * 60)
    
    try:
        # 启动WebSocket服务器
        server_process = start_websocket_server()
        
        # 等待服务器启动
        print("[INFO] 等待服务器启动...")
        time.sleep(3)
        
        # 打开前端界面
        open_frontend()
        
        print("\n[SUCCESS] 系统启动完成！")
        print("[INFO] 前端界面已在浏览器中打开")
        print("[INFO] WebSocket服务器运行在 ws://localhost:8765")
        print("[INFO] 按 Ctrl+C 停止系统")
        
        # 保持程序运行
        try:
            while True:
                time.sleep(1)
                # 检查服务器进程是否还在运行
                if server_process.poll() is not None:
                    print("[ERROR] WebSocket服务器意外退出")
                    break
        except KeyboardInterrupt:
            print("\n[INFO] 正在停止服务...")
            
            # 停止服务器进程
            if server_process:
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                    print("[SUCCESS] WebSocket服务器已停止")
                except subprocess.TimeoutExpired:
                    server_process.kill()
                    print("[WARNING] 强制停止WebSocket服务器")
            
            print("[INFO] 程序已退出")
            
    except Exception as e:
        print(f"[ERROR] 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 