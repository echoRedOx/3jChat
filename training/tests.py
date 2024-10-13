from datasets import load_dataset
import pandas as pd

json_data = load_dataset("training_data/john-openai/turns.json")
print(json_data)