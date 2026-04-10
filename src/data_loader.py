from datasets import load_dataset

def load_amazon_dataset(sample=False, train_size=5000, test_size=1000):
    dataset = load_dataset("amazon_polarity")

    if sample:
        train = dataset["train"].shuffle(seed=42).select(range(train_size))
        test = dataset["test"].shuffle(seed=42).select(range(test_size))
        return {"train": train, "test": test}

    return dataset
