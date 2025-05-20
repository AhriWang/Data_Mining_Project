class Config:
    """
    模型配置类，包含所有可配置参数
    """
    # 模型参数
    model_name = "/root/datamining2/qwen2.5-0.5B/"  # 改为使用BERT模型，更兼容旧版本PyTorch
    max_seq_length = 128  # 最大序列长度，超过此长度的文本将被截断
    num_classes = 2  # 分类类别数量，二分类为2
    
    # 训练参数
    batch_size = 64  # 批次大小，可根据GPU内存调整
    learning_rate = 2e-5  # 学习率
    num_epochs = 10  # 训练轮数
    unfreeze_layers = 4
    
    # 路径配置
    train_path = "/root/datamining2/selected_data.py"  # 训练集路径
    dev_path = "/root/datamining2/dev.csv"  # 验证集路径
    test_path = "/root/datamining2/test.csv"  # 测试集路径
    model_save_path = "/root/datamining2/models"  # 模型保存路径