import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io  # 用于在内存中处理Excel文件

# 页面标题
st.title("Application Program for Predicting Granite Genesis Types")
st.title("花岗岩成因类型预测应用程序")
st.write("This model uses apatite trace elements to predict the genesis types of granite, and the results are available for users to download!")
st.write("该模型使用磷灰石微量元素预测花岗岩的成因类型，结果可供用户下载！")
st.write("Developer: Dr. Fengge Han; School of Sciences, East China University of Technology; School of Earth and Planetary Sciences, East China University of Technology, Nanchang, China")
st.write("开发人员：韩凤歌（博士）；东华理工大学理学院;东华理工大学地球与行星科学学院，南昌，中国")
st.write("Email: hanfengge@ecut.edu.cn")
st.markdown('---')
    # return data

# 加载训练数据
@st.cache_resource
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/fenggeHan/granite_classification_prediction_app/refs/heads/main/1240shiyan%20-%20o.csv')
    return data

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
data = load_data()
model, train_accuracy, test_accuracy = train_model(data)

# 显示训练和测试准确度
st.write("The model is being trained on the provided dataset. Please wait...")
st.write("模型训练中，请稍等……")
st.write(f"Training accuracy（训练精确度）: {train_accuracy:.4f}")
st.write(f"Testing accuracy（测试精确度）: {test_accuracy:.4f}")
st.success("Model training completed（模型训练已完成）")  # 模型训练完成提示
st.markdown('---')

# 步骤 1：显示模板下载链接
st.subheader("Step 1: Download Data Template (if needed)")
st.subheader("第一步：请下载数据模板（如果需要）")
st.markdown("If you do not have a data template, please click the button below to download the template:")
st.markdown("如果您没有数据模板，请单击下面的按钮下载模板：")

# 直接提供模板文件下载链接
template_url = 'https://raw.githubusercontent.com/fenggeHan/granite_classification_prediction_app/refs/heads/main/Data%20Template-granite.csv'

# 生成模板下载按钮
st.download_button(
    label="Click to download the template（单击此处下载模板）",
    data=pd.read_csv(template_url).to_csv(index=False),  # 直接读取并转换为CSV格式
    file_name="Data_Template-granite.csv",
    mime="text/csv"
)
st.success("Template download completed!（模板下载完成！）")  # 模版下载完成提示
st.markdown('---')

# 步骤 2：用户上传数据
st.subheader("Step 2: Please upload Your Data for Prediction")
st.subheader("第二步：请您上传您的数据用于预测")
uploaded_file = st.file_uploader("Please upload a CSV file that matches the download template（请上传与下载模板匹配的CSV文件）", type="csv")

# 如果上传了文件
if uploaded_file is not None:
    # 读取上传的CSV文件，并确保正确读取数据（第一行是列名）
    user_data = pd.read_csv(uploaded_file, header=0)  # 强制读取第一行作为列名

    # 检查上传的数据列数是否匹配（25列特征）
    if user_data.shape[1] == 25:
        # 进行预测
        predictions = model.predict(user_data)

        # 打印预测结果（网页上显示）
        st.write("Prediction Results（预测结果）:")
        st.write(predictions)

        # 将预测结果存放到一个 DataFrame 中，并插入到第26列
        result_df = user_data.copy()  # 保留用户上传的数据
        result_df['Prediction'] = predictions  # 将预测结果直接插入为新的列（第26列）

        # 显示合并后的数据
        st.write("Preview of the merged data（显示合并后数据的预测结果）:")
        st.write(result_df)  # 显示合并后的数据

        # 将预测结果保存为Excel文件并提供下载链接
        excel_file = io.BytesIO()  # 在内存中创建一个字节流对象
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name="Predictions")  # 将结果写入Excel文件
        excel_file.seek(0)  # 重置文件指针到开头

        # 生成下载按钮
        st.download_button(
            label="Click to download the prediction results in Excel file（点击下载Excel文件中的预测结果）",
                      data=excel_file,
            file_name="predictions_with_results.xlsx",  # 用户下载的文件名
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    else:
        st.error(f"The uploaded CSV file should contain 25 feature columns. Please check the data format.")
        st.error(f"上传的CSV文件应包含25个特征列。请检查数据格式。")
        
else:
    st.warning("Please check your data and upload a CSV file that matches the template for prediction.\n 请检查您的数据，并上传一个与预测模板匹配的CSV文件。")
st.markdown('---')
    # return data

st.subheader("Citation（引用）")
st.write("* Han, F., Leng, C., Chen, J., Zou, S. & Wang, D. (2025). Machine lerarning method for discriminating granite genetic types based on trace element composition of apatite. Acta Petrologica Sinica, 41 (02), 737-750. (in Chinese with English abstract). doi: 10. 18654/1000-0569/")
st.write("* 韩凤歌, 冷成彪, 陈加杰, 邹少浩,王大钊. 2025. 基于磷灰石微量元素组成的机器学习方法判别花岗岩成因类型. 岩石学报, 41(02): 737-750. doi: 10. 18654/1000-0569/")
























