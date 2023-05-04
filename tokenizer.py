from transformers import pipeline 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for HuggingFace course my whole life.")

print(res)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("I've been waiting for HuggingFace course my whole life.")

print(res)
# is the exact same as the first time

# now we will look at tokenizer 

sequence = "Using a Transformer network is simple"
res = tokenizer(sequence)
print(res)
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)

# {'input_ids': [101, 2478, 1037, 10938, 2121, 2897, 2003, 3722, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
# ['using', 'a', 'transform', '##er', 'network', 'is', 'simple']
# [2478, 1037, 10938, 2121, 2897, 2003, 3722]
# using a transformer network is simple


# ids 101, 102 are beginning and end of sentence