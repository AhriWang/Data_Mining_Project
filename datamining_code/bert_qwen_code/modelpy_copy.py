import torch.nn as nn
from transformers import AutoModel
import torch


class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(SentimentClassifier, self).__init__()
        # 加载 Qwen2.5-0.5B 模型
        self.qwen = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.qwen.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # 使用 Qwen2.5-0.5B 进行前向传播
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 获取 [CLS] 的输出（对于 Qwen2.5-0.5B，通常会返回最后一层的池化输出）
        pooled_output = outputs.last_hidden_state[:, 0]  # 取[CLS]标记的输出
        output = self.dropout(pooled_output)
        return self.classifier(output)

    def save_model(self, path):
        # 保存模型的状态字典
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        # 加载模型的状态字典
        self.load_state_dict(torch.load(path))
