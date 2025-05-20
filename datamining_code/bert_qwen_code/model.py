import torch
import torch.nn as nn
import os
from transformers import AutoModel, PreTrainedModel, AutoConfig
import json


class SentimentClassifier(PreTrainedModel):
    def __init__(self, model_name, num_classes=2):
        config = AutoConfig.from_pretrained(model_name)
        super(SentimentClassifier, self).__init__(config)

        # 加载 Qwen2.5-0.5B 模型
        self.qwen = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.qwen.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # 使用 Qwen2.5-0.5B 进行前向传播
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0]  # 取[CLS]标记的输出
        output = self.dropout(pooled_output)
        return self.classifier(output)

    def save_pretrained(self, save_directory):
        """
        保存模型权重和配置文件
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 保存模型的权重
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # 保存模型配置
        config_path = os.path.join(save_directory, 'config.json')
        self.config.save_pretrained(save_directory)

        print(f"模型已保存到 {save_directory}")

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, num_classes=2, **kwargs):
        """
        从预训练模型加载权重和配置
        """
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = super(SentimentClassifier, cls).from_pretrained(model_name_or_path, config=config, *args, num_classes=2,**kwargs)
        return model

    def save_model(self, path):
        """
        保存模型的状态字典
        :param path: 保存路径
        """
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = os.path.join(path, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")

        # 保存模型的配置文件（可选）
        config_path = os.path.join(path, "config.json")
        config = {
            'model_name': self.qwen.config._name_or_path,
            'num_classes': self.classifier.out_features,
            'hidden_size': getattr(self.qwen.config, 'hidden_size', None),
            'vocab_size': getattr(self.qwen.config, 'vocab_size', None),
            'max_position_embeddings': getattr(self.qwen.config, 'max_position_embeddings', None),
            'type_vocab_size': getattr(self.qwen.config, 'type_vocab_size', None),  # 安全访问
            'num_attention_heads': getattr(self.qwen.config, 'num_attention_heads', None),
            'num_hidden_layers': getattr(self.qwen.config, 'num_hidden_layers', None)
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        print(f"模型配置已保存至 {config_path}")

    def load_model(self, path):
        """
        加载模型的状态字典
        :param path: 加载路径
        """
        model_path = os.path.join(path, "pytorch_model.bin")
        self.load_state_dict(torch.load(model_path))
        print(f"模型已从 {model_path} 加载")

        # 加载模型的配置文件（可选）
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"模型配置已加载自 {config_path}")

        return config  # 返回加载的配置文件
