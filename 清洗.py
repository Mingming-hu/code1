import pandas as pd

def clean_and_count_data(df, id_col='chatID', time_col='日期'):
    """
    数据清洗函数：
    1. 保留每个ID首次出现时的数据
    2. 统计每个ID在当前时间段的数据条数
    
    参数:
    df: 原始DataFrame
    id_col: ID列名 (默认'id')
    time_col: 时间列名 (默认'timestamp')
    
    返回:
    清洗后的DataFrame，包含首次出现数据和计数
    """
    # 确保时间列是datetime类型
    df[time_col] = pd.to_datetime(df[time_col])
    
    # 按ID和时间排序
    df_sorted = df.sort_values(by=[id_col, time_col])
    
    # 找出每个ID的首次出现记录
    first_occurrences = df_sorted.drop_duplicates(subset=[id_col], keep='first')
    
    # 计算每个ID在每个时间段的数据条数
    count_df = df.groupby([id_col, pd.Grouper(key=time_col, freq='D')]).size().reset_index(name='当前聊天轮数')
    
    # 合并首次出现记录和计数数据
    result = pd.merge(first_occurrences, 
                     count_df, 
                     on=[id_col, time_col], 
                     how='left')
    
    return result

# 示例数据
df = pd.read_excel("C:\\Users\\jiezhang76\\Downloads\\12月脱敏-郑润泽清洗(1).xlsx")
# 使用清洗函数
cleaned_data = clean_and_count_data(df)

print("清洗后的数据:")
cleaned_data.to_csv("C:\\Users\\jiezhang76\\Downloads\\初步清洗.csv")
print(cleaned_data)