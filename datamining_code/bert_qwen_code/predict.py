import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch.nn.functional as F


# 你的模型结构：Qwen 主干 + Dropout + Linear 分类头
class SentimentClassifier(nn.Module):
    def __init__(self, model_path, num_labels=2):
        super(SentimentClassifier, self).__init__()
        self.qwen = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.qwen.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        cls_output = last_hidden_state[:, 0, :]  # 取第一个 token (CLS)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


# 配置参数
class Config:
    model_ckpt_path = "/root/datamining2/models/pytorch_model.bin"  # 保存的模型权重
    qwen_model_path = "/root/datamining2/qwen2.5-0.5B"              # 用于加载结构
    max_len = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"


def predict_sentiment(model, tokenizer, texts, config):
    model.eval()
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=config.max_len,
        return_tensors="pt"
    )

    input_ids = encoded['input_ids'].to(config.device)
    attention_mask = encoded['attention_mask'].to(config.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    return preds.cpu().tolist(), probs.cpu().tolist()


def main():
    config = Config()

    # 加载 tokenizer 和模型结构
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_path, trust_remote_code=True)
    model = SentimentClassifier(config.qwen_model_path).to(config.device)

    # 加载权重
    state_dict = torch.load(config.model_ckpt_path, map_location=config.device)
    model.load_state_dict(state_dict)

    # 示例评论（英文）
    test_comments = [
        "This product is amazing! I love it so much.",
        "Worst purchase I've ever made. Completely disappointed.",
        "The quality is okay, not great but not terrible.",
        "Excellent service and very fast delivery!",
        "Good product for whoever wants to waste money.",
        "Worst choice i ever made...",
        "It broke after two days. Waste of money."
    ]

    predictions, probs = predict_sentiment(model, tokenizer, test_comments, config)

    for i, (text, label, prob) in enumerate(zip(test_comments, predictions, probs)):
        sentiment = "Positive" if label == 1 else "Negative"
        confidence = round(max(prob) * 100, 2)
        print(f"\nExample {i+1}:")
        print(f"Text: {text}")
        print(f"Prediction: {sentiment} ({confidence}%)")


if __name__ == "__main__":
    main()
