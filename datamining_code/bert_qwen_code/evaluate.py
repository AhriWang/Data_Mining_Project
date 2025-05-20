from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate(model, eval_loader, device):
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
            all_probs.extend(probs[:, 1].detach().cpu().numpy())  # 正类概率
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
        auc = 0.0  # 仅一种类时报错，设为 0

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n评估指标：")
    print(f"准确率（Accuracy）: {accuracy:.4f}")
    print(f"F1-score（加权）: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("混淆矩阵：")
    print(cm)
    print("分类报告：")
    print(classification_report(all_labels, all_preds))

    # 混淆矩阵图
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return avg_loss, accuracy
