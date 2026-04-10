import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd

# Load data
train_df = pd.read_csv("data/train_small.csv", encoding="utf-8-sig")
test_df = pd.read_csv("data/test_small.csv", encoding="utf-8-sig")

# title + content birleştir (NaN güvenli)
train_df["text"] = train_df["title"].fillna("") + " " + train_df["content"].fillna("")
test_df["text"] = test_df["title"].fillna("") + " " + test_df["content"].fillna("")

# sadece gerekli kolonları tut
train_df = train_df[["text", "label"]]
test_df = test_df[["text", "label"]]


# Tokenizer // burada BERT'in üstüne sentiment_classification katmanı ekliyoruz.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)


# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

#Batch’teki tüm cümleler aynı uzunlukta olmalı bu yüzden padding ile düzenleme yapıyoruz.
#"max_length=128"e sabitleyerek her cümlenin 128 token olmasını sağladık.
#truncation ile eğer cümle çok uzunsa ve belirlediğimiz max tokenin(128) dışına çıkıyorsa cümlenin sonundan kesecek.

train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

train_ds = train_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])

train_ds.set_format("torch")
test_ds.set_format("torch")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# W = Weight Decay > Overfittingi azaltır. 
# 2e-5 = altın oran.  lr büyürse, model bozulur. küçülürse, model öğrenemez
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train
#Gradient = yönü söyler, ağırlığı x değiştirmek, loss nasıl etkilenir?


# range 3: tüm dataseti baştan sona 3 kez dolaş
for epoch in range(3):
    model.train()
    total_loss = 0
    
#tqdm, Kaç batch kaldı, eğitim ne durumda görmek için.
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()} #batchi gpuya taşır
        
#forwardpass
#input → logits; logits + labels → loss
#Logits = modelin ham karar puanları
        outputs = model(**batch)
        loss = outputs.loss
        
#backprop (fine-tune)
        loss.backward()         #gradient hesapla
        optimizer.step()        #ağırlıkları güncelle
        optimizer.zero_grad()   #eski gradienti temizle.sıfırlamazsan, eski- yeni batch karışır. model yanlış öğrenir
                                
        total_loss += loss.item()
        
#loss = modelin ne kadar yanlış öğrendiğini gösteren değer
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
preds, labels = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        pred = torch.argmax(outputs.logits, dim=1) #karar anı, argmax karşılaştırma yapıyor
        preds.extend(pred.cpu().numpy())
        labels.extend(batch["labels"].cpu().numpy())

print("Accuracy:", accuracy_score(labels, preds))

# Save model
model.save_pretrained("model")
tokenizer.save_pretrained("model")
