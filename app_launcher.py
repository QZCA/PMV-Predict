import sys,os
import subprocess
import ctypes
import encodings

# 环境判断
if getattr(sys, 'frozen', False):
    # 在 EXE 环境中执行
    temp_path = sys._MEIPASS

    python_dll_path = os.path.join(temp_path, 'python311.dll')

    # 将临时路径添加到系统 PATH 中
    os.environ['PATH'] = temp_path + ";" + os.environ['PATH']
    os.environ['PYTHONHOME'] = temp_path  # 设置 Python 运行环境
    os.environ['PYTHONPATH'] = temp_path  # 设置 Python 库路径
    app_path = sys._MEIPASS  # 获取 EXE 解压后的临时路径
    # 强制将标准库路径添加到 sys.path
    sys.path.append(os.path.join(app_path, 'lib-dynload'))
    sys.path.append(os.path.join(app_path, 'base_library.zip'))
    
    # 检查是否成功加载 DLL
    ctypes.CDLL(python_dll_path)
    print(f"Successfully loaded Python DLL from {python_dll_path}")
    
    app_path = sys._MEIPASS  # 获取 EXE 解压后的临时路径
    app_script = os.path.join(app_path, 'PWV_APP.py')
    print(f"当前 Path: {app_path}")
    streamlit_path = os.path.join(app_path, 'streamlit')  # 临时路径中的 streamlit
    print(f"Streamlit 临时路径：{streamlit_path}")
    
    # 手动添加 streamlit 到 sys.path
    if streamlit_path not in sys.path:
        sys.path.append(streamlit_path)

else:
    # 在开发环境中执行
    app_path = os.path.dirname(os.path.abspath(__file__))
    app_script = os.path.join(app_path, 'PWV_APP.py')

# 使用 Popen 启动 Streamlit 应用，避免阻塞主进程
subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_script])
