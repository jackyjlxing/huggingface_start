from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification

classifier1 = pipeline("sentiment-analysis")

res = classifier1("I've been waiting for a huggingFace course.  ")

print(f"sentiment analysis result with default model : --- {res}")

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "In this course, we will teach you how to ", 
    max_length=30,
    num_return_sequences=2,
    )

print(f"generator result with default model : --- {res}")

classifier2=pipeline("zero-shot-classification")

res = classifier2(
    "This is a course about Python list comprehension", 
    candidate_labels=["education", "politics", "business"],
)

print(f"zero shot result with default model : --- {res}")



model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sequence = "Using a Transformer network is simple"
res = tokenizer(sequence)
print(f"----tokenized sequence : {res}")
tokens = tokenizer.tokenize(sequence)
print(f"----tokens : {tokens}")
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"----ids : {ids}")
decoded_string = tokenizer.decode(ids)
print(f"----decoded string : {decoded_string}")
