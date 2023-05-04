from transformers import pipeline 

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

res = classifier(
    "This course is about coding Python",
    candidate_labels=["education", "politics", "business"]
)

print(res)