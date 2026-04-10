from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "zeyneppinarsoy/amazon-sentiment-bert-zp"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def predict(text):
    #giriş verisini hazırla
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # 
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    #olasılıkları hesapla (Softmax)
    probs = torch.softmax(logits, dim=1).flatten()
    
    # en yüksek olasılığı ve hangi sınıfa ait olduğunu bul
    conf, class_idx = torch.max(probs, dim=0)
    
    label = "Positive" if class_idx.item() == 1 else "Negative"
    confidence_percent = conf.item() * 100
    
    # sonucu formatla
    return f"{label} (confidence: %{confidence_percent:.2f})"