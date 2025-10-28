import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# 页面标题
st.title("Granite Prediction App")

# 加载训练数据
@st.cache_resource
def load_data():
    # 读取数据，并确保第一行作为列名
    data = pd.read_csv('https://raw.githubusercontent.com/fenggeHan/granite_classification_prediction_app/main/1240shiyan.csv', header=0)
    return data

data = load_data()

# 提取特征和标签
features = data.iloc[:1240, :-1]  # 特征: 所有列，去掉最后一列标签列
labels = data.iloc[:1240, -1]  # 标签: 最后一列

# 每次用户访问时重新训练模型
@st.cache_resource
def train_model():
    # 数据划分
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

    # 创建并训练随机森林模型
    model = RandomForestClassifier(n_estimators=110, max_features=11, random_state=42, max_depth=10)
    model.fit(x_train, y_train)

    # 输出训练准确度
    y_train_pred = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    y_test_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return model, train_accuracy, test_accuracy

model, train_accuracy, test_accuracy = train_model()

# 显示模型训练准确度
st.write(f"训练准确度: {train_accuracy:.4f}")
st.write(f"测试准确度: {test_accuracy:.4f}")

# 用户上传数据
uploaded_file = st.file_uploader("上传符合模板的数据CSV文件", type="csv")

# 显示模板下载链接
st.markdown("如果你没有数据模板，请下载 [Data Template-granite.csv](https://github.com/fenggeHan/granite_classification_prediction_app/blob/main/Data%20Template-granite.csv)")

if uploaded_file is not None:
    # 读取上传的CSV文件
    user_data = pd.read_csv(uploaded_file, header=0)

    # 确保用户数据列数和训练数据一致
    if user_data.shape[1] == features.shape[1]:
        # 用户上传的数据用于预测
        X_user = user_data  # 17个特征

        # 进行预测
        predictions = model.predict(X_user)

        # 显示预测结果
        st.write("预测结果:")
        st.write(predictions)

        # 将预测结果显示为表格
        result_df = user_data.copy()
        result_df['Prediction'] = predictions
        st.write(result_df)
    else:
        st.error(f"上传的CSV文件特征列数应为{features.shape[1]}列，请检查数据格式。")

else:
    st.info("请上传一个符合模板的CSV文件进行预测。")

