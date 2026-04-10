from src.data_loader import load_amazon_dataset

dataset = load_amazon_dataset(sample=True)

print(dataset["train"][0])
print("Train size:", len(dataset["train"]))
print("Test size:", len(dataset["test"]))

#small dataseti kaydet
dataset["train"].to_csv("data/train_small.csv")
dataset["test"].to_csv("data/test_small.csv")