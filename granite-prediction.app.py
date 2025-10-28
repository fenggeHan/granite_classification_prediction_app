import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

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

# 加载训练数据
@st.cache_resource
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/fenggeHan/granite_classification_prediction_app/refs/heads/main/1240shiyan.csv')
    return data

data = load_data()

# 提取特征和标签
features = data.iloc[:, :-1]  # 特征: 所有列，去掉最后一列标签列
labels = data.iloc[:, -1]  # 标签: 最后一列

# 每次用户访问时重新训练模型
@st.cache_resource
def train_model():
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

# 调用训练函数
model, train_accuracy, test_accuracy = train_model()

# 显示训练和测试准确度
st.write(f"训练准确度: {train_accuracy:.4f}")
st.write(f"测试准确度: {test_accuracy:.4f}")

# 如果上传了文件
if uploaded_file is not None:
    # 读取上传的CSV文件，并确保正确读取数据（第一行是列名）
    user_data = pd.read_csv(uploaded_file, header=0)  # 强制读取第一行作为列名

    # 检查上传的数据列数是否匹配
    if user_data.shape[1] == features.shape[1]:
        # 进行预测
        predictions = model.predict(user_data)

        # 打印预测结果（网页上显示）
        st.write("预测结果:")
        st.write(predictions)

        # 将预测结果存放到一个 DataFrame 中，并插入到第26列
        result_df = user_data.copy()  # 保留用户上传的数据
        result_df.insert(25, 'Prediction', predictions)  # 在第26列插入预测结果

        # 显示合并后的数据
        st.write("合并后的数据预览：")
        st.write(result_df.head())  # 显示前几行合并后的数据

        # 下载结果（直接内存，不写文件）
        output_csv = result_df.to_csv(index=False).encode("utf-8")  # 使用 utf-8 编码
        st.download_button(
            label="Download prediction results",
            data=output_csv,  # 直接内存下载
            file_name="prediction_results_with_data.csv",  # 文件名
            mime="text/csv"  # 文件类型
        )
    else:
        st.error(f"上传的CSV文件特征列数应为{features.shape[1]}列，请检查数据格式。")
else:
    st.info("请上传一个符合模板的CSV文件进行预测。")
