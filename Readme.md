# 🤖 GPT-2 Text Generator
> A project by Aniket Singh  
> Generative AI Intern at **CodeCraft**

---

## 📌 Project Overview

This project showcases my first task as a **Generative AI Intern** at CodeCraft. I fine-tuned **OpenAI’s GPT-2 language model** using a custom text dataset that includes structured, relatable content such as:
- 📝 Long and short stories
- 💌 Thank-you messages
- 💬 Daily AI conversation prompts
- ❓ General knowledge Q&A
- 👨‍💻 Coding questions with answers
- 📚 Letters to friends and family

The model was trained using Hugging Face's **Transformers** and **Datasets** libraries inside a Google Colab environment.

---

## 🛠️ Tools & Libraries Used

| Tool/Service         | Purpose                              |
|----------------------|--------------------------------------|
| 🤗 Hugging Face Transformers | Fine-tuning GPT-2 model           |
| 🤗 Datasets           | Custom dataset management            |
| Google Colab         | Model training & GPU usage           |
| Python               | Programming language                 |
| Streamlit            | Web-based text generation UI         |
| GitHub               | Project repository hosting           |

---
[Open In Colab](https://colab.research.google.com/drive/1SCr8leJzUS-HpRv9uqttF49nSp3_80ox?usp=sharing)

## 📂 Dataset Details

The dataset is a structured `.txt` file (~2MB), manually created and cleaned. It contains over 100+ formatted entries like:

```text
=== Long Story ===
Once upon a time in a remote village in India, there lived a boy named Aarav...

=== Thank You Message ===
Dear Friend,  
Thank you for always being there for me...

=== Coding Question ===
Q: Write a Python function to reverse a list.  
A:
```python
def reverse_list(lst):
    return lst[::-1]
yaml
Copy
Edit

---

## 💡 Prompt Structure

To get precise and formatted outputs, I designed a **prompt-tagging scheme**:
- `=== Short Story ===`
- `=== Thank You Message ===`
- `=== General Knowledge Question ===`
- `=== Coding Question ===`
- `=== Daily AI Prompt ===`

This ensures the model learns context better and generates structured content accordingly.

---

## 📍 Training Procedure (Google Colab)

I followed these steps in Google Colab:

1. **Install dependencies**
```python
!pip install transformers datasets accelerate
Upload and prepare dataset

python
Copy
Edit
from datasets import Dataset
from transformers import AutoTokenizer

# Load custom dataset
with open("gpt2_structured_dataset.txt", "r", encoding="utf-8") as f:
    data = f.read().split("=== ")
records = [{"text": "=== " + entry.strip()} for entry in data if entry.strip()]
dataset = Dataset.from_list(records)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("gpt2")
def tokenize(batch): return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
tokenized_dataset = dataset.map(tokenize, batched=True)
Fine-tune GPT-2

python
Copy
Edit
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-checkpoints",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset
)

trainer.train()
Save the model

python
Copy
Edit
model.save_pretrained("./gpt2-custom")
tokenizer.save_pretrained("./gpt2-custom")
🧪 Testing the Model
To test generated results:

python
Copy
Edit
from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-custom", tokenizer="./gpt2-custom")
print(generator("=== Short Story ===\n", max_length=100, do_sample=True)[0]['generated_text'])
🌐 Streamlit App (Web UI)
Run this app to interact with your model using dropdown prompts.

bash
Copy
Edit
streamlit run app.py
Sample UI:

python
Copy
Edit
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

st.title("🧠 GPT-2 Custom Text Generator")

option = st.selectbox("Choose a prompt type", ["Short Story", "Thank You Message", "Coding Question"])
prompt = f"=== {option} ===\\n"

model = AutoModelForCausalLM.from_pretrained("gpt2-custom")
tokenizer = AutoTokenizer.from_pretrained("gpt2-custom")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

if st.button("Generate Text"):
    output = generator(prompt, max_length=120, do_sample=True, top_k=40, temperature=0.9)[0]['generated_text']
    st.write("### Output:")
    st.write(output)
🧠 Learnings & Skills Gained
As part of this project, I learned:

How to structure data for LLM training

How to fine-tune GPT-2 with Hugging Face

Prompt design for structured output

Tokenization and model evaluation

Building a UI using Streamlit

Deploying projects using GitHub


🙋‍♂️ Author
Aniket Singh
Generative AI Intern at CodeCraft
🔗www.linkedin.com/in/aniket-singh7as | 🔗 https://github.com/TheAniketSingh-07 



---

Would you like me to:
- Add this directly to your GitHub `README.md` file?
- Bundle this into your zipped repo?
- Help you write a LinkedIn post version of this?

Just say the word!
