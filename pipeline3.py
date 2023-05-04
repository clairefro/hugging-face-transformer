from transformers import pipeline 

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for HuggingFace course my whole life.")

print(res)