from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
