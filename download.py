from transformers import AutoTokenizer, AutoModel

model_name = "nlpaueb/legal-bert-base-uncased"
save_path = r"/Users/balamuralibr/legalbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)