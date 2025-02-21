import pandas as pd  

# Load the LIAR dataset
liar_data = pd.read_csv("Liar_Dataset.csv", sep=",")  
print("LIAR Dataset Loaded Successfully!")  

# Load FakeNewsNet (example path)
fake_news = pd.read_csv("politifact_fake.csv")  
print("politifact_fake Dataset Loaded Successfully!")

