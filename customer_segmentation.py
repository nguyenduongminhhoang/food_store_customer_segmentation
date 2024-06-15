import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns



# ---------------------------
# GUI
st.title("Data Science Project")
st.image("Project 1 - Customer Segmentation.png")


# GUI
menu = ["Giới thiệu", "Thống kê chung", "Thông tin khách hàng"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Giới thiệu':    
    st.subheader("Giới thiệu")
    st.write("Trang web hỗ trợ các chủ cửa hàng: ")
    st.markdown("- Tiện lợi theo dõi các giao dịch của cửa hàng")
    st.markdown("- Thông kê và đưa ra một góc nhìn tổng quát về giao dịch")
    st.markdown("- Phân cụm các nhóm khách hàng")
    st.markdown("---")

elif choice == "Thống kê chung":
    st.subheader("Thống kê chung")
    df = None
    uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
    if uploaded_file_1 is not None:
        df = pd.read_csv(uploaded_file_1)
        df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
        # global lines
        st.dataframe(df)
        st.code(df)
        tab1, tab2 = st.tabs(["Doanh thu", "Khách hàng"])
        with tab1:
            st.header("Doanh thu")
            # Code
            df_spending = pd.DataFrame(df.groupby(by='Member_number')['order_value'].sum().sort_values(ascending=False)).reset_index()
            top_3 = df_spending.head(3)
            st.subheader('Top 3 những khách hàng tiêu tiền nhiều nhất: ')
            st.markdown("- Mã Member: {} - Chi tiêu: {}".format(top_3.Member_number.iloc[0], top_3.order_value.iloc[0]))
            st.markdown("- Mã Member: {} - Chi tiêu: {}".format(top_3.Member_number.iloc[1], top_3.order_value.iloc[1]))
            st.markdown("- Mã Member: {} - Chi tiêu: {}".format(top_3.Member_number.iloc[2], top_3.order_value.iloc[2]))
            st.markdown('---')
            st.subheader('Theo tổng thể: ')
            st.markdown('Trung bình một người chi: {}'.format(round(df_spending['order_value'].mean(),2)))
            st.markdown('Tổng thu nhập: {}'.format(round(df.order_value.sum(),2)))
            st.markdown('---')
            st.subheader('Trong tháng gần đây nhất (12/2015): ')
            st.markdown('Trung bình một người chi: {}'.format(round(df_spending['order_value'].mean(),2)))
            st.markdown('Tổng thu nhập: {}'.format(round(df.order_value.sum(),2)))
            st.markdown('---')
            st.subheader("Biểu đồ thể hiện doanh thu:")
            # Code
            df['year_month'] = df['Date'].dt.to_period('M')
            # Group by year_month and sum the order values
            monthly_sums = df.groupby('year_month')['order_value'].sum().reset_index()
            monthly_sums['year_month'] = monthly_sums['year_month'].astype(str)
            fig, ax = plt.subplots(figsize=(12, 6))
            # Plotting using Seaborn
            sns.barplot(x='year_month', y='order_value', data=monthly_sums, palette='Blues_d', ax=ax)
            # Customize the plot
            ax.set_title('Tổng doanh thu mỗi tháng')
            ax.set_xlabel('Năm-Tháng')
            ax.set_ylabel('Doanh thu')
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
            # Display the plot using st.pyplot()
            st.pyplot(fig)

        with tab2:
            st.header("Khách hàng - Tần suất")
            #Code
            df_freq = pd.DataFrame(df['Member_number'].value_counts().reset_index())
            df_freq.columns = ['Member_number', 'Count']
            st.subheader('Top 4 những khách hàng mua nhiều nhất: ')
            st.markdown("- Mã Member: {} - Số lần: {}".format(df_freq.Member_number.iloc[0], df_freq.Count.iloc[0]))
            st.markdown("- Mã Member: {} - Số lần: {}".format(df_freq.Member_number.iloc[1], df_freq.Count.iloc[1]))
            st.markdown("- Mã Member: {} - Số lần: {}".format(df_freq.Member_number.iloc[2], df_freq.Count.iloc[2]))
            st.subheader('Tổng cộng có: {} members'.format(df_freq.shape[0]))
            st.subheader('Trung bình một người mua: {} lần'.format(round(df_freq['Count'].mean())))
            st.markdown("---")
            st.header("Khách hàng - Ngừng thực hiện giao dịch")
            st.subheader("Biểu đồ thể hiện thời điểm ngừng giao dịch của khách hàng:")
            # Code
            df_inactive = pd.DataFrame(df['Member_number'].value_counts().reset_index())
            df_inactive.columns = ['Member_number', 'Count']
            df_inactive.sort_values(by = 'Member_number', inplace=True)
            df_inactive.reset_index(drop=True, inplace=True)
            inactive_since = []
            for i in df_inactive['Member_number']:
                df_ = df[df['Member_number']==i]
                inactive_since.append(df_['Date'].max())
            df_inactive['Inactive Since'] = inactive_since
            df_inactive['year_month'] = df_inactive['Inactive Since'].dt.to_period('M')
            # Group by year_month and sum the order values
            date_count = df_inactive.groupby('year_month')['Count'].count().reset_index()
            date_count['year_month'] = date_count['year_month'].astype(str)
            fig, ax = plt.subplots(figsize=(12, 6))
            # Plotting using Seaborn
            sns.barplot(x='year_month', y='Count', data=date_count, palette='Blues_d', ax=ax)
            # Customize the plot
            ax.set_title('Thời điểm mà khách hàng ngừng hoạt động, cùng với số lượng')
            ax.set_xlabel('Năm-Tháng')
            ax.set_ylabel('Số lượng khách hàng ngừng hoạt động')
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
            # Display the plot using st.pyplot()
            st.pyplot(fig)


elif choice == 'Thông tin khách hàng':
    # st.subheader("Upload your data")
    st.markdown("##### Please upload your file and enter your memberID. We will then extract the information.")
    # flag = False
    df = None
    uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
    if uploaded_file_1 is not None:
        df = pd.read_csv(uploaded_file_1)
        # global lines
        st.dataframe(df)
        # st.code(df)
        # ---------------------------
        # Data pre-processing
        df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
        df['order_date'] = df['Date']
        ## RFM Analysis
        max_date = df['Date'].max().date()
        Recency = lambda x: (max_date - x.max().date()).days
        Frequency = lambda x: len(x.unique())
        Monetary = lambda x: round(sum(x),2)
        df_RFM = df.groupby('Member_number').agg({'Date': Recency, 'order_date': Frequency, 'order_value': Monetary})
        df_RFM.rename(columns={'Date': 'Recency', 'order_date': 'Frequency', 'order_value': 'Monetary'}, inplace=True)
        ## Create labels for Recency, Frequency, Monetary
        r_label = range(4,0,-1)
        f_label = range(1,5)
        m_label = range(1,5)
        ## Asign labels to 4 percentile group
        r_groups = pd.qcut(df_RFM["Recency"].rank(method="first"), q=4, labels = r_label)
        f_groups = pd.qcut(df_RFM["Frequency"].rank(method="first"), q=4, labels= f_label)
        m_groups = pd.qcut(df_RFM["Monetary"].rank(method="first"), q=4, labels= f_label)
        df_RFM = df_RFM.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)
        ## Concat RFM quartile values to create RFM segments
        def join_rfm(x):
            return str(int(x["R"])) + str(int(x["F"])) + str(int(x["M"]))
        df_RFM["RFM_segments"] = df_RFM.apply(join_rfm, axis=1)
        df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)

        # Build model
        ## Define rule-based
        def rfm_level(df):
            # Check for special 'STARS' and 'NEW' conditions first
            if df['RFM_Score'] == 12:
                return 'VIP'
            elif df['R'] == 4 and df['F'] == 1 and df['M'] == 1:
                return 'NEW COMER'
            elif df['R'] == 4 and df['M'] == 4:
                return 'NEW COMER BIG SPENDER'
            elif df['M'] == 4 and df['F'] >= 3:
                return 'LOYAL BIG SPENDER'
            elif df['M'] == 4:
                return 'BIG SPENDER'
            elif df['F'] == 4:
                return 'LOYAL'
            elif df['R'] == 4:
                return 'ACTIVE'
            elif df['R'] == 1:
                return 'LOST'
            elif df['M'] == 1:
                return 'LIGHT'
            else:
                return 'REGULARS'
        ## Create a new column RFM_Level
        df_RFM['RFM_Level'] = df_RFM.apply(rfm_level, axis=1)
        ## Reset index
        df_RFM = df_RFM.reset_index()
        ## Function tp filter based on ID
        def filter_dataframe_by_id(df, id_value):
            # Lọc DataFrame theo ID
            filtered_df = df[df['Member_number'] == id_value]
            return filtered_df
        ## Function to extract info
        def find_segment_by_id(df, id_value):
            # Lọc DataFrame theo ID
            result = df[df["Member_number"] == id_value]
            if not result.empty:
                # Lấy giá trị segment đầu tiên (giả sử ID là duy nhất)
                segment = result.iloc[0]['RFM_Level']
                return segment
            else:
                return None
        # ---------------------------

    memberID = st.text_input(label="Input memberID: ")
    if memberID:
        segment= find_segment_by_id(df_RFM, float(memberID))
        filter_df_rfm = filter_dataframe_by_id(df_RFM, float(memberID))
        filter_df = filter_dataframe_by_id(df, float(memberID))
        if segment:
            st.code(f'ID {memberID} is {segment} member. RFM score is {filter_df_rfm["RFM_Score"].iloc[0]}.\nThe user last went shopping {filter_df_rfm["Recency"].iloc[0]} days ago, purchased at the store {filter_df_rfm["Frequency"].iloc[0]} times with the total spending ${filter_df_rfm["Monetary"].iloc[0]}')
            st.code(filter_df_rfm)
            st.code(filter_df)
            st.bar_chart(data=filter_df, x="productName", y="items", color=None, width=None, height=None, use_container_width=True)
            st.bar_chart(data=filter_df, x="Date", y="items", color=None, width=None, height=None, use_container_width=True)


    

