import pandas as pd
from imblearn.over_sampling import RandomOverSampler


def load_and_filter_data(path, min_len=20, max_len=128):
    """
    读取数据并筛选评论长度在 [min_len, max_len] 范围内的样本。
    """
    print("正在加载数据...")
    df = pd.read_csv(path, names=['label', 'topic', 'text'])

    print("去除缺失值...")
    df = df.dropna(subset=['label', 'text'])

    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)

    print(f"筛选长度在 {min_len}~{max_len} 字符之间的文本...")
    df = df[(df['text'].str.len() >= min_len) & (df['text'].str.len() <= max_len)]

    print(f"筛选后数据量：{len(df)} 条")
    return df


def balance_and_limit(df, limit=10000):
    """
    使用过采样进行类别平衡，并限制总数据量为 limit。
    """
    print("正在进行类别平衡...")
    ros = RandomOverSampler(random_state=42)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    topics = df['topic'].tolist()

    texts_bal, labels_bal = ros.fit_resample(pd.DataFrame(texts), pd.Series(labels))
    topics_bal = pd.Series(topics).iloc[ros.sample_indices_].tolist()

    df_balanced = pd.DataFrame({
        'label': labels_bal,
        'topic': topics_bal,
        'text': texts_bal[0]
    })

    if len(df_balanced) > limit:
        print(f"数据量过大，正在随机采样至 {limit} 条...")
        df_balanced = df_balanced.sample(n=limit, random_state=42).reset_index(drop=True)

    print("类别分布：")
    print(df_balanced['label'].value_counts())
    return df_balanced


def save_balanced_data(df, output_path="/root/datamining2/balanced_train_data.csv"):
    """
    保存最终筛选并平衡后的数据到CSV。
    """
    df.to_csv(output_path, index=False, header=False)
    print(f"已保存平衡后的训练数据至：{output_path}")


def main():
    input_path = "/root/datamining2/train_part_1.csv"
    output_path = "/root/datamining2/balanced_train_data.csv"

    df = load_and_filter_data(input_path)
    df_balanced = balance_and_limit(df, limit=10000)
    save_balanced_data(df_balanced, output_path)


if __name__ == "__main__":
    main()
