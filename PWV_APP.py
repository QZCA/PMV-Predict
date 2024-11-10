print("进入 PWV_APP.py 逻辑")

import os,sys
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
print('streamlit模块导入完成')
os.environ["STREAMLIT_LOG_LEVEL"] = "error"  # 只显示错误信息


# 环境判断
if getattr(sys, 'frozen', False):
    # 在 EXE 环境中执行时，不再启动 Streamlit
    app_path = sys._MEIPASS  # 获取 EXE 解压后的临时路径
    print(f"Current Path: {app_path}")
else:
    # 在开发环境中执行
    app_path = os.path.dirname(os.path.abspath(__file__))

# 模型和 DLL 文件路径
model_path = os.path.join(app_path, 'xgboost_regressor_model.bin')
dll_path = os.path.join(app_path, 'xgboost', 'lib', 'xgboost.dll')
from xgboost import XGBRegressor
# 加载模型等其他应用逻辑
@st.cache_resource
def load_model(model_path):
    model = XGBRegressor()
    model.load_model(model_path)
    print('XGboost模型加载成功！')
    return model

model = load_model(model_path)

##正文

# 定义预测函数
def predict(features):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return prediction[0]

# 设置应用程序的标题
st.title("PMV 预测程序")

# 创建选项卡来分隔四个子界面
tab1, tab2, tab3, tab4 , tab5 = st.tabs(["手动输入预测", "滑块预测","图形显示预测","图形显示预测(三维)","文件上传批量预测"])

# 特征重要性数据
feature_importances = {
    "SET": 0.800585,
    "Air temperature": 0.158153,
    "Met": 0.029631,
    "Air velocity": 0.006931,
    "Clo": 0.003446,
    "Relative humidity": 0.001255,
}

# 特征顺序根据特征重要性排列
ordered_features = ["SET", "Air temperature", "Met", "Air velocity", "Clo", "Relative humidity"]

# 第一个子界面：手动输入预测
with tab1:
    st.header("手动输入预测")
    
    # 按照重要性顺序排列输入框
    set_val = st.number_input("SET", 6.55, 61.53, value=25.77, key="set_input")
    air_temp = st.number_input("Air temperature", 0.60, 48.80, value=24.35, key="temp_input")
    met_val = st.number_input("Met", 0.60, 6.80, value=1.21, key="met_input")
    air_velocity = st.number_input("Air velocity (m/s)", 0.00, 56.17, value=0.15, key="velocity_input")
    clo_val = st.number_input("Clo", 0.03, 2.87, value=0.70, key="clo_input")
    rel_humidity = st.number_input("Relative humidity (%)", 0.50, 100.00, value=46.66, key="humidity_input")
    
    if st.button("预测 PMV"):
        features = [set_val, air_temp, met_val, air_velocity, clo_val, rel_humidity]
        prediction = predict(features)
        st.markdown(f"预测的 PMV 值为: <span style='color: skyblue; font-size: 20px; font-weight: bold'>{prediction:.2f}</span>", unsafe_allow_html=True)


    # 特征重要性图
    st.write(f"=="*43+"\n")
    st.subheader("各特征对 PMV 的影响")
    st.write(f"数值越大代表该特征对PMV的影响越大，反之越小")
    
    # 绘制柱状图
    fig, ax = plt.subplots()
    sns.barplot(x=list(feature_importances.values()), y=list(feature_importances.keys()),
                ax=ax, palette="Blues_d", hue=list(feature_importances.keys()))
    for i, v in enumerate(feature_importances.values()):
        ax.text(v, i, f"{v:.4f}", color='k', va='center')
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance for PMV Prediction")
    st.pyplot(fig)

# 第二个子界面：滑块预测
with tab2:
    st.header("滑块预测")
    
    # 按照重要性顺序排列滑块
    set_val = st.slider("SET", 6.55, 61.53, value=25.77, key="set_slider")
    air_temp = st.slider("Air temperature", 0.60, 48.80, value=24.35, key="temp_slider")
    met_val = st.slider("Met", 0.60, 6.80, value=1.21, key="met_slider")
    air_velocity = st.slider("Air velocity (m/s)", 0.00, 56.17, value=0.15, key="velocity_slider")
    clo_val = st.slider("Clo", 0.03, 2.87, value=0.70, key="clo_slider")
    rel_humidity = st.slider("Relative humidity (%)", 0.50, 100.00, value=46.66, key="humidity_slider")

    features = [set_val, air_temp, met_val, air_velocity, clo_val, rel_humidity]
    prediction = predict(features)
    st.markdown(f"预测的 PMV 值为: <span style='color: skyblue; font-size: 20px; font-weight: bold'>{prediction:.2f}</span>", unsafe_allow_html=True)


# 第三个子界面：图形显示预测
with tab3:
    st.header("图形显示预测")
    
    # 选择菜单，选择一个变量
    selected_variable = st.selectbox("选择一个变量", ordered_features)
    
    # 输入框手动输入其他变量的值
    st.subheader("手动输入其他变量的值")

    set_val = st.number_input("SET", 6.55, 61.53, value=25.77, key="set_manual")
    air_temp = st.number_input("Air temperature", 0.60, 48.80, value=24.35, key="temp_manual")
    met_val = st.number_input("Met", 0.60, 6.80, value=1.21, key="met_manual")
    air_velocity = st.number_input("Air velocity (m/s)", 0.00, 56.17, value=0.15, key="velocity_manual")
    clo_val = st.number_input("Clo", 0.03, 2.87, value=0.70, key="clo_manual")
    rel_humidity = st.number_input("Relative humidity (%)", 0.50, 100.00, value=46.66, key="humidity_manual")

    # 获取选择的变量的范围
    variable_range = {
        "SET": (6.55, 61.53),
        "Air temperature": (0.60, 48.80),
        "Met": (0.60, 6.80),
        "Air velocity": (0.00, 56.17),
        "Clo": (0.03, 2.87),
        "Relative humidity": (0.50, 100.00),
    }

    # 生成选定变量的取值范围
    var_min, var_max = variable_range[selected_variable]
    values = np.linspace(var_min, var_max, 100)
    
    # 设置其他变量的固定值
    fixed_values = {
        "SET": set_val,
        "Air temperature": air_temp,
        "Met": met_val,
        "Air velocity": air_velocity,
        "Clo": clo_val,
        "Relative humidity": rel_humidity,
    }

    # 计算每个取值下的 PMV
    pmv_values = []
    for value in values:
        # 更新选择变量的值
        fixed_values[selected_variable] = value
        
        # 使用更新后的变量值进行预测
        features = [fixed_values[feat] for feat in ordered_features]
        pmv = predict(features)
        pmv_values.append(pmv)

    # 绘制散点图
    fig, ax = plt.subplots(figsize=(7,3.5),dpi=128)
    sns.scatterplot(x=values, y=pmv_values, marker="o", edgecolor='black', alpha=0.4, ax=ax)
    ax.set_xlabel(selected_variable)
    ax.set_ylabel("Predicted PMV")
    ax.set_title(f"Diff {selected_variable} to PMV  ")
    
    # 在图的左上角显示其他5个变量的取值
    ax.text(0.05, 0.95, "\n".join([f"{k}: {v}" for k, v in fixed_values.items() if k != selected_variable]),
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', color='black')

    # 隐藏其他边界线，只保留x轴和y轴的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    st.pyplot(fig)



with tab4:
    st.header("两个变量与PMV的三维关系图")
    
    # 选择两个变量
    selected_variable_1 = st.selectbox("选择第一个变量", ordered_features)
    selected_variable_2 = st.selectbox("选择第二个变量", ordered_features)
    
    # 输入框手动输入其他四个变量的值
    st.subheader("手动输入其他变量的值")
    set_val = st.number_input("SET", 6.55, 61.53, value=25.77, key="set_manual_2")
    air_temp = st.number_input("Air temperature", 0.60, 48.80, value=24.35, key="temp_manual_2")
    met_val = st.number_input("Met", 0.60, 6.80, value=1.21, key="met_manual_2")
    air_velocity = st.number_input("Air velocity (m/s)", 0.00, 56.17, value=0.15, key="velocity_manual_2")
    clo_val = st.number_input("Clo", 0.03, 2.87, value=0.70, key="clo_manual_2")
    rel_humidity = st.number_input("Relative humidity (%)", 0.50, 100.00, value=46.66, key="humidity_manual_2")

    # 获取选择的变量的范围
    variable_range = {
        "SET": (6.55, 61.53),
        "Air temperature": (0.60, 48.80),
        "Met": (0.60, 6.80),
        "Air velocity": (0.00, 56.17),
        "Clo": (0.03, 2.87),
        "Relative humidity": (0.50, 100.00),
    }

    # 生成选定变量的取值范围
    var_min_1, var_max_1 = variable_range[selected_variable_1]
    var_min_2, var_max_2 = variable_range[selected_variable_2]
    
    # 创建网格用于绘图
    values_1 = np.linspace(var_min_1, var_max_1, 20)
    values_2 = np.linspace(var_min_2, var_max_2, 20)
    X, Y = np.meshgrid(values_1, values_2)
    
    # 设置其他变量的固定值
    fixed_values = {
        "SET": set_val,
        "Air temperature": air_temp,
        "Met": met_val,
        "Air velocity": air_velocity,
        "Clo": clo_val,
        "Relative humidity": rel_humidity,
    }

    # 计算每个网格点的 PMV
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fixed_values[selected_variable_1] = X[i, j]
            fixed_values[selected_variable_2] = Y[i, j]
            
            # 使用更新后的变量值进行预测
            features = [fixed_values[feat] for feat in ordered_features]
            Z[i, j] = predict(features)

    # 绘制三维散点图
    fig = plt.figure(figsize=(9, 7),dpi=128)  # 调整图表尺寸
    ax = fig.add_subplot(111, projection='3d')

    # 根据PMV值来为每个点着色
    sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', marker='o', alpha=0.7)
    
    # 设置轴标签
    ax.set_xlabel(selected_variable_1)
    ax.set_ylabel(selected_variable_2)
    ax.set_zlabel("Predicted PMV")
    ax.set_title(f"{selected_variable_1} with {selected_variable_2} to PMV ")
    
    # 显示颜色条
    fig.colorbar(sc, ax=ax, label='PMV')

    # 显示其他四个固定变量的取值
    # 只显示未被选择的变量的取值
    remaining_vars = [var for var in ordered_features if var not in [selected_variable_1, selected_variable_2]]
    text = "\n".join([f"{var}: {fixed_values[var]}" for var in remaining_vars])
    
    ax.text2D(0.05, 0.95, text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

    st.pyplot(fig)


with tab5:
    st.header("文件上传批量预测")
    uploaded_file = st.file_uploader("上传一个 Excel 或 CSV 文件", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # 检查是否包含所有需要的列
        required_columns = ["SET", "Clo", "Met", "Air temperature", "Relative humidity (%)", "Air velocity (m/s)"]
        if all(column in df.columns for column in required_columns):
            X = df[required_columns]
            predictions = model.predict(X)
            df['Predicted PMV'] = predictions
            df.to_csv('pred_result.csv', index=False)
            st.write("预测结果已保存到 pred_result.csv")
            st.write(df.head())
        else:
            st.write("上传的文件缺少必要的列")

    # 案例数据表格
    st.write(f"=="*43+"\n")
    st.subheader("案例数据格式参考")
    st.write(f"上传文件必须包含这6列变量，且表头名称要和如下示例一模一样：")
    example_data = {
        "SET": [22.20, 24.45, 23.74, 24.49, 23.90],
        "Clo": [0.57, 0.57, 0.57, 0.57, 0.57],
        "Met": [1.0, 1.1, 1.1, 1.0, 1.0],
        "Air temperature": [24.3, 25.7, 24.6, 26.4, 25.0],
        "Relative humidity (%)": [36.8, 33.1, 34.9, 31.7, 33.3],
        "Air velocity (m/s)": [0.27, 0.09, 0.06, 0.13, 0.07],
    }
    example_df = pd.DataFrame(example_data)
    st.write(example_df)

