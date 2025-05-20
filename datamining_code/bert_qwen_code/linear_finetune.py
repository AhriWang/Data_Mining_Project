import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import SentimentClassifier
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
import os
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def evaluate(model, eval_loader, device, epoch=None, save_metrics_path=None):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            _, preds = torch.max(outputs, dim=1)

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)
    accuracy = correct_predictions.double() / total_predictions
    f1 = f1_score(all_labels, all_preds, average='weighted')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n评估指标：")
    print(f"准确率（Accuracy）: {accuracy:.4f}")
    print(f"F1-score（加权）: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("混淆矩阵：")
    print(cm)
    print("分类报告：")
    print(classification_report(all_labels, all_preds))

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f"Confusion Matrix - Epoch {epoch+1}" if epoch is not None else "Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # 保存指标
    if save_metrics_path is not None:
        row = {
            "epoch": epoch + 1 if epoch is not None else -1,
            "val_loss": avg_loss,
            "accuracy": accuracy.item(),
            "f1_score": f1,
            "auc_roc": auc
        }

        if not os.path.exists(save_metrics_path):
            df = pd.DataFrame(columns=row.keys())
        else:
            df = pd.read_csv(save_metrics_path)

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(save_metrics_path, index=False)

    return avg_loss, accuracy


def linear_finetune(previous_model_path, save_path, train_texts, train_labels, val_texts, val_labels):
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(previous_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SentimentClassifier(previous_model_path, num_classes=config.num_classes)
    model.load_model(previous_model_path)
    model.to(device)

    # 只训练分类头（冻结 Qwen 主干）
    for param in model.qwen.parameters():
        param.requires_grad = False

    # 优化器只更新 requires_grad = True 的部分
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate)

    # 构建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    best_accuracy = 0
    metrics_log_path = os.path.join(save_path, "metrics_log.csv")

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, device, epoch, metrics_log_path)
        print(f"Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            model.save_model(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"保存当前最佳模型到 {save_path}")


if __name__ == "__main__":
    config = Config()
    data_loader = DataLoaderClass(config)

    train_texts, train_labels = data_loader.load_csv("/root/datamining2/balanced_train_data.csv")
    val_texts, val_labels = data_loader.load_csv("/root/datamining2/dev.csv")

    previous_model_path = "/root/datamining2/models"   # 已有模型
    save_path = "/root/datamining2/models3"            # 保存线性微调后的模型

    linear_finetune(previous_model_path, save_path, train_texts, train_labels, val_texts, val_labels)
