import os
import subprocess
import sys

# 获取当前目录
app_path = os.path.dirname(os.path.abspath(__file__))

# 绝对路径
xgboost_dll_path = r"C:\Users\qzca\AppData\Roaming\Python\Python311\site-packages\xgboost\lib\xgboost.dll"
model_path = os.path.join(app_path, 'xgboost_regressor_model.bin')
xgboost_version_path = r"C:\Users\qzca\AppData\Roaming\Python\Python311\site-packages\xgboost\VERSION"
python_dll_path = r"C:\Users\qzca\AppData\Local\Programs\Python\Python311\python311.dll"  # Python DLL 路径

# PyInstaller 打包命令
pyinstaller_cmd = [
    "pyinstaller", "--onefile",
    "--collect-all", "streamlit",  # 收集 Streamlit 库的所有资源
    "--collect-all", "importlib.metadata",  # 收集相关依赖
    "--collect-all", "encodings",
    "--hidden-import", "encodings",  # 隐藏导入 encodings
    "--hidden-import", "streamlit",  # 隐藏导入 Streamlit
    f'--add-binary={xgboost_dll_path};./xgboost/lib',  # 添加 xgboost.dll 文件
    f'--add-binary={python_dll_path};./',  # 添加 python311.dll 到根目录
    f'--add-data={model_path};.',  # 添加训练好的模型
    f'--add-data={xgboost_version_path};./xgboost',  # 添加 xgboost 的 VERSION 文件
    f'--add-data={os.path.join(app_path, "PWV_APP.py")}:.',  # 添加 PWV_APP.py
    f'--add-data={os.path.join(app_path, "app_launcher.py")}:.',  # 添加 app_launcher.py
    'app_launcher.py'  # 目标启动脚本
]

# 打包应用
subprocess.run(pyinstaller_cmd, check=True)

