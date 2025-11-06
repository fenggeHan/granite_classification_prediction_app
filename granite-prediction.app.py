import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import io  # 用于在内存中处理Excel文件

# 页面标题
st.title("Granite Prediction App")

# 显示模板下载链接
st.markdown("如果你没有数据模板，请点击下面的按钮下载模板：")

# 直接提供模板文件下载链接
template_url = 'https://raw.githubusercontent.com/fenggeHan/granite_classification_prediction_app/refs/heads/main/Data%20Template-granite.csv'

# 生成模板下载按钮
st.download_button(
    label="点击下载模板",
    data=pd.read_csv(template_url).to_csv(index=False),  # 直接读取并转换为CSV格式
    file_name="Data_Template-granite.csv",
    mime="text/csv"
)

# 用户上传数据
uploaded_file = st.file_uploader("上传符合模板的数据CSV文件", type="csv")

# 加载训练数据并预处理
@st.cache_resource
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/fenggeHan/granite_classification_prediction_app/refs/heads/main/1240shiyan%20-%20o.csv')
    
    # 数据预处理：对数变换（避免负值）
    data_transformed = data.copy()
    data_transformed.iloc[:, :-1] = np.log1p(data_transformed.iloc[:, :-1])  # 对数变换，避免零或负值
    
    # 计算期望（均值）和方差
    mean_values = data_transformed.iloc[:, :-1].mean()
    std_values = data_transformed.iloc[:, :-1].std()

    # 返回预处理后的数据以及期望和方差
    return data_transformed, mean_values, std_values

# 训练并返回模型
@st.cache_resource
def train_model(data):
    features = data.iloc[:, :-1]  # 特征: 所有列，去掉最后一列标签列
    labels = data.iloc[:, -1]  # 标签: 最后一列

    # 数据划分
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

    # 创建并训练随机森林模型
    model = RandomForestClassifier(n_estimators=110, max_features=11, random_state=42, max_depth=10)
    model.fit(x_train, y_train)
    
    # 评估训练和测试准确度
    y_train_pred = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    y_test_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    return model, train_accuracy, test_accuracy

# 加载训练数据并训练模型
data, mean_values, std_values = load_data()
model, train_accuracy, test_accuracy = train_model(data)

# 显示训练和测试准确度
st.write(f"训练准确度: {train_accuracy:.4f}")
st.write(f"测试准确度: {test_accuracy:.4f}")

# 如果上传了文件
if uploaded_file is not None:
    # 读取上传的CSV文件，并确保正确读取数据（第一行是列名）
    user_data = pd.read_csv(uploaded_file, header=0)  # 强制读取第一行作为列名

    # 检查上传的数据列数是否匹配（25列特征）
    if user_data.shape[1] == 25:
        # 对上传的数据进行预处理：对数变换
        user_data_transformed = user_data.copy()
        user_data_transformed.iloc[:, :-1] = np.log1p(user_data_transformed.iloc[:, :-1])  # 对数变换，避免零或负值
        
        # 确保数据是数值型并且没有缺失值
        user_data_transformed.iloc[:, :-1] = user_data_transformed.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
        
        # 删除缺失值
        user_data_transformed.dropna(inplace=True)

        # 标准化处理：使用训练集的期望和方差对上传的数据进行标准化
        scaler = StandardScaler()
        scaler.mean_ = mean_values.values  # 使用训练数据的均值（转换为numpy数组）
        scaler.scale_ = std_values.values  # 使用训练数据的标准差（转换为numpy数组）

        # 确保标准化时数据的维度一致
        user_data_features = user_data_transformed.iloc[:, :-1].values  # 提取特征部分
        user_data_transformed.iloc[:, :-1] = scaler.transform(user_data_features)

        # 进行预测
        predictions = model.predict(user_data_transformed)

        # 打印预测结果（网页上显示）
        st.write("预测结果:")
        st.write(predictions)

        # 将预测结果存放到一个 DataFrame 中，并插入到第26列
        result_df = user_data.copy()  # 保留用户上传的数据
        result_df['Prediction'] = predictions  # 将预测结果直接插入为新的列（第26列）

        # 显示合并后的数据
        st.write("合并后的数据预览：")
        st.write(result_df)  # 显示合并后的数据

        # 将预测结果保存为Excel文件并提供下载链接
        excel_file = io.BytesIO()  # 在内存中创建一个字节流对象
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name="Predictions")  # 将结果写入Excel文件
        excel_file.seek(0)  # 重置文件指针到开头

        # 生成下载按钮
        st.download_button(
            label="点击下载包含预测结果的Excel文件",
            data=excel_file,
            file_name="predictions_with_results.xlsx",  # 用户下载的文件名
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    else:
        st.error(f"上传的CSV文件特征列数应为25列，请检查数据格式。")
else:
    st.info("请上传一个符合模板的CSV文件进行预测。")
