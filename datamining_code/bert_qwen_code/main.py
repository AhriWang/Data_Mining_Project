import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier
import torch.nn as nn
import os
from torch.cuda.amp import autocast, GradScaler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_hf_mirrors():
    """
    设置Hugging Face镜像，加速模型下载
    """
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = './hf_cache'


# 设置镜像
set_hf_mirrors()


def evaluate(model, eval_loader, device):
    """
    评估模型性能
    """
    print("开始评估模型...")
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)
    accuracy = correct_predictions.double() / total_predictions
    print(f"评估完成：平均损失 = {avg_loss:.4f}, 准确率 = {accuracy:.4f}")
    return avg_loss, accuracy


def train(train_texts, train_labels, val_texts=None, val_labels=None):
    """
    启用混合精度训练（自动使用 bf16 或 fp16）
    """
    print("开始训练模型...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 自动选择混合精度类型
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"使用混合精度类型: {'bf16' if amp_dtype == torch.bfloat16 else 'fp16'}")

    local_model_path = "/root/datamining2/qwen2.5-0.5B/"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SentimentClassifier(local_model_path, config.num_classes)
    model.to(device)

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    if val_texts is not None and val_labels is not None:
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    total_steps = len(train_loader) * config.num_epochs
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    scaler = GradScaler(enabled=(amp_dtype == torch.float16))  # bf16 不需要 scaler
    best_accuracy = 0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        print(f"开始训练 epoch {epoch + 1}/{config.num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast(dtype=amp_dtype):  # 关键：自动混合精度上下文
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"批次 {batch_idx}/{len(train_loader)} 训练损失: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{config.num_epochs}')
        print(f'Average training loss: {avg_train_loss:.4f}')

        scheduler.step()

        if val_texts is not None and val_labels is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                model.save_pretrained(config.model_save_path)
                tokenizer.save_pretrained(config.model_save_path)
                print(f"保存新的最佳模型，准确率: {val_accuracy:.4f}")

    return model


def predict(text, model_path=None):
    """
    使用训练好的模型进行预测
    """
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SentimentClassifier(config.model_name, config.num_classes)
    if model_path:
        model.load_model(model_path)
    model.to(device)
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config.max_seq_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predictions = torch.max(outputs, dim=1)

    return predictions.item()


if __name__ == "__main__":
    set_hf_mirrors()

    config = Config()
    data_loader = DataLoaderClass(config)

    print("加载训练集...")
    train_texts, train_labels = data_loader.load_csv("/root/datamining2/balanced_train_data.csv")
    print("加载验证集...")
    val_texts, val_labels = data_loader.load_csv("/root/datamining2/dev.csv")
    print("加载测试集...")
    test_texts, test_labels = data_loader.load_csv("/root/datamining2/test.csv")

    print(f"训练集: {len(train_texts)} 样本")
    print(f"验证集: {len(val_texts)} 样本")
    print(f"测试集: {len(test_texts)} 样本")

    print("开始训练模型...")
    model = train(train_texts, train_labels, val_texts, val_labels)

    example_text = "这个产品质量非常好，我很满意！"
    prediction = predict(example_text, config.model_save_path)
    sentiment = "正面" if prediction == 1 else "负面"
    print(f"示例文本: '{example_text}'")
    print(f"情感预测: {sentiment} (类别 {prediction})")
