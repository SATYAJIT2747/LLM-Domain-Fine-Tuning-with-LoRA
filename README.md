# LLM Domain Fine-Tuning with LoRA
### Code Revision + Training & Inference Guide

This notebook demonstrates **domain-specific causal language-model training using LoRA (PEFT)**.

The goal here is not to revise the theory.  
The goal is to understand:

- What each code block does
- Why it is required
- What parameters we change
- What happens during training
- How the trained adapter is loaded for inference
- Which parts need modification for a real dataset

---

# 1. Overall Pipeline

The complete workflow is:

```text
Install Libraries
       ↓
Import Libraries
       ↓
Check GPU
       ↓
Load Tokenizer
       ↓
Prepare Dataset
       ↓
Tokenize Dataset
       ↓
Create Causal-LM Data Collator
       ↓
Load Base LLM
       ↓
Configure LoRA
       ↓
Attach LoRA to Model
       ↓
Check Trainable Parameters
       ↓
Define TrainingArguments
       ↓
Create Trainer
       ↓
trainer.train()
       ↓
Save LoRA Adapter
       ↓
--------------------
       ↓
Load Base Model
       ↓
Load LoRA Adapter
       ↓
Load Tokenizer
       ↓
Generate Response
```

---

# 2. Installation

```python
!pip install -U transformers datasets peft accelerate
```

### Why?

We need four major libraries:

```text
transformers → model + tokenizer + Trainer
datasets     → dataset handling
peft         → LoRA
accelerate   → device/GPU management
```

### Remember

For LoRA:

```python
transformers
peft
datasets
accelerate
```

are the main libraries.

---

# 3. Imports

```python
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)
```

### Important classes

| Class | Purpose |
|---|---|
| `AutoTokenizer` | Text ↔ token IDs |
| `AutoModelForCausalLM` | Load causal LLM |
| `TrainingArguments` | Configure training |
| `Trainer` | Training loop abstraction |
| `DataCollatorForLanguageModeling` | Prepare batches/labels |
| `LoraConfig` | Define LoRA |
| `get_peft_model()` | Attach LoRA to model |

---

# 4. Check GPU

```python
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

### Why?

Before loading a 1.1B model, check whether CUDA is available.

```python
torch.cuda.is_available()
```

returns:

```text
True
```

if PyTorch can access the GPU.

---

# 5. Select Base Model

```python
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
```

This is the **base pretrained model**.

We don't train the entire model.

Instead:

```text
Base Model
    +
LoRA Adapter
    ↓
Fine-tuned Model
```

---

# 6. Load Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Tokenizer converts:

```text
Text
 ↓
Tokens
 ↓
Token IDs
```

For example:

```text
"Hello world"
       ↓
[token1, token2]
       ↓
[...., ....]
```

---

## Padding Token

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

Some causal LMs don't have a dedicated padding token.

So we reuse EOS:

```text
PAD = EOS
```

This prevents padding-related errors during batching.

### Remember

If you see:

```python
tokenizer.pad_token is None
```

you may need:

```python
tokenizer.pad_token = tokenizer.eos_token
```

---

# 7. Create Dataset

The notebook uses a small example dataset:

```python
texts = [
    "Machine learning is a field of artificial intelligence.",
    "Deep learning uses neural networks to learn representations.",
    "Transformers use attention mechanisms to process sequences.",
    "LoRA is a parameter efficient fine tuning technique.",
    "Large language models can be fine tuned using LoRA."
]

dataset = Dataset.from_dict({
    "text": texts
})
```

The important thing is that the dataset has a text column:

```text
dataset
   ↓
"text"
   ↓
training text
```

For a real project, this section is replaced with your actual domain dataset.

---

# 8. Tokenization

```python
MAX_LENGTH = 256

def tokenize_function(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )
```

### What happens?

For every example:

```text
example["text"]
       ↓
tokenizer()
       ↓
input_ids
attention_mask
```

---

## Apply Tokenization

```python
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
```

### Important

`map()` applies the tokenizer to the dataset.

```text
Raw Dataset
    ↓
dataset.map()
    ↓
Tokenized Dataset
```

### `batched=True`

Instead of calling the function one example at a time, batches are passed to it.

### `remove_columns=["text"]`

The original raw text column is removed after tokenization.

---

# 9. Data Collator

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

This is important for **causal language modeling**.

```python
mlm=False
```

means:

> Don't perform masked language modeling.

Instead, the model learns **next-token prediction**.

Conceptually:

```text
Input:

I love machine

Target:

love machine learning
```

The model learns to predict the next token.

---

# 10. Load Base Model

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### `AutoModelForCausalLM`

Used because TinyLlama is a causal language model.

### `torch_dtype=torch.float16`

Loads model weights in FP16.

This reduces GPU memory usage compared with FP32.

### `device_map="auto"`

Lets Transformers automatically place the model on available hardware.

For example:

```text
GPU available
    ↓
Model → GPU
```

---

# 11. Disable Cache During Training

```python
model.config.use_cache = False
```

Caching is useful during generation, but it is normally disabled during training.

So:

```text
Training → use_cache=False
Inference → cache can be enabled
```

### Easy interview question

**Why disable `use_cache` during training?**

Because KV caching is designed mainly to speed up autoregressive generation and is unnecessary during standard training.

---

# 12. LoRA Configuration

This is the most important code section.

```python
lora_config = LoraConfig(

    task_type=TaskType.CAUSAL_LM,

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    bias="none",

    target_modules=[
        "q_proj",
        "v_proj"
    ]
)
```

---

## `task_type`

```python
task_type=TaskType.CAUSAL_LM
```

Tells PEFT that we're fine-tuning a causal language model.

---

## `r`

```python
r=8
```

LoRA rank.

Remember the tradeoff:

```text
Higher r
   ↓
More parameters
More capacity
More memory

Lower r
   ↓
Fewer parameters
Less memory
Lower capacity
```

---

## `lora_alpha`

```python
lora_alpha=16
```

Controls the scaling of the LoRA update.

Here:

```text
r = 8
alpha = 16
```

So the effective scaling factor is commonly represented as:

```text
alpha / r = 16 / 8 = 2
```

---

## `lora_dropout`

```python
lora_dropout=0.05
```

Dropout applied to the LoRA path.

Used to reduce overfitting.

---

## `bias`

```python
bias="none"
```

Means bias parameters aren't trained as part of this LoRA configuration.

---

# 13. Target Modules

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

This tells LoRA **where to insert the trainable adapter layers**.

Here:

```text
Attention
   ├── q_proj ← LoRA
   ├── k_proj
   ├── v_proj ← LoRA
   └── ...
```

This is an important thing to remember when adapting LoRA code to another architecture.

Different models can use different projection names.

---

# 14. Attach LoRA

```python
model = get_peft_model(
    model,
    lora_config
)
```

This transforms:

```text
Original Model
     ↓
get_peft_model()
     ↓
LoRA-enabled Model
```

The important point:

```text
Original model parameters
        ↓
      Frozen

LoRA parameters
        ↓
      Trainable
```

So instead of updating all 1.1B parameters, only a small number of parameters are trained.

---

# 15. Check Trainable Parameters

```python
model.print_trainable_parameters()
```

This is a very useful debugging step.

You should see something like:

```text
trainable params: ...
all params: ...
trainable%: ...
```

The important thing is:

```text
trainable %
      ↓
very small
```

If suddenly most of the model is trainable, something is wrong with your PEFT configuration.

---

# 16. Training Arguments

```python
training_args = TrainingArguments(

    output_dir="./tinyllama-lora",

    num_train_epochs=5,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    fp16=True,

    logging_steps=20,

    save_total_limit=1,

    report_to="none",

    save_strategy="epoch"
)
```

This controls the training process.

---

# 17. `output_dir`

```python
output_dir="./tinyllama-lora"
```

Where checkpoints and training outputs are stored.

---

# 18. `num_train_epochs`

```python
num_train_epochs=5
```

Number of complete passes through the dataset.

```text
1 epoch = entire dataset once
```

So:

```text
5 epochs
=
dataset processed 5 times
```

---

# 19. Batch Size

```python
per_device_train_batch_size=1
```

One example is processed per GPU at a time.

Small batch size is useful when GPU memory is limited.

---

# 20. Gradient Accumulation

```python
gradient_accumulation_steps=8
```

Instead of updating the model after every single example:

```text
Batch 1 → gradients
Batch 2 → gradients
Batch 3 → gradients
...
Batch 8 → update weights
```

Effective batch size:

```text
1 × 8 = 8
```

So:

```text
effective batch size
=
per_device_batch_size
×
gradient_accumulation_steps
```

For multiple GPUs, the number of devices also matters.

---

# 21. Learning Rate

```python
learning_rate=2e-4
```

Learning rate controls how strongly LoRA parameters are updated.

For LoRA, learning rates can generally be higher than full-model fine-tuning because only the adapter parameters are being trained.

---

# 22. FP16

```python
fp16=True
```

Uses half-precision training.

Benefits:

```text
FP16
 ↓
less memory
faster computation on supported GPUs
```

But FP16 requires compatible hardware/software.

---

# 23. Logging

```python
logging_steps=20
```

Training information is logged every 20 steps.

Useful for monitoring:

```text
loss
training progress
```

---

# 24. Checkpoint Saving

```python
save_strategy="epoch"
```

Save a checkpoint after every epoch.

```python
save_total_limit=1
```

Keep only one checkpoint to avoid filling disk space.

---

# 25. Create Trainer

```python
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=data_collator
)
```

`Trainer` connects everything:

```text
Model
 +
TrainingArguments
 +
Dataset
 +
DataCollator
 ↓
Trainer
```

It handles the training loop for you.

---

# 26. Start Training

```python
trainer.train()
```

This is where the actual optimization starts.

Conceptually:

```text
Dataset
   ↓
Tokenizer output
   ↓
Batch
   ↓
Model
   ↓
Loss
   ↓
Backpropagation
   ↓
LoRA parameters updated
```

The base model remains frozen.

---

# 27. Save LoRA Adapter

```python
adapter_path = "./tinyllama-lora-final"

model.save_pretrained(adapter_path)

tokenizer.save_pretrained(adapter_path)
```

Important:

This does **not** save another complete 1.1B model.

Instead:

```text
Base TinyLlama
     +
LoRA Adapter
```

The adapter contains the learned LoRA weights.

This makes the trained output much smaller than saving an entire fine-tuned model.

---

# 28. INFERENCE

Training is finished.

Now we need:

```text
Base Model
     +
Trained LoRA Adapter
     ↓
Fine-tuned Model
     ↓
Generate Answer
```

---

# 29. Load Base Model

```python
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

We load the **same base model architecture/model** used during training.

---

# 30. Load LoRA Adapter

```python
model = PeftModel.from_pretrained(
    base_model,
    "./tinyllama-lora-final"
)
```

This is a very important line.

It combines:

```text
base_model
    +
./tinyllama-lora-final
    ↓
LoRA-enabled model
```

The adapter cannot normally be treated as a complete standalone LLM.

You load it on top of the corresponding base model.

---

# 31. Load Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(
    "./tinyllama-lora-final"
)
```

The tokenizer saved during training is loaded.

---

# 32. Evaluation Mode

```python
model.eval()
```

Switches the model to evaluation/inference mode.

This is important because layers such as dropout should behave differently during inference.

---

# 33. Interactive Loop

```python
while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("Exiting...")
        break
```

This simply creates a terminal chatbot:

```text
You:
 ↓
Question
 ↓
Generate answer
 ↓
Print answer
 ↓
Ask again
```

---

# 34. Tokenize User Question

```python
inputs = tokenizer(
    question,
    return_tensors="pt"
).to(model.device)
```

The process is:

```text
User text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
PyTorch tensors
   ↓
Model device
```

### `return_tensors="pt"`

Return PyTorch tensors.

### `.to(model.device)`

Move the input to the same device as the model.

This prevents errors such as:

```text
Expected all tensors to be on the same device
```

---

# 35. Generate

```python
with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )
```

This is the actual inference step.

---

## `torch.no_grad()`

```python
with torch.no_grad():
```

No gradients are required during inference.

Benefits:

```text
less memory
less computation
```

---

# 36. `max_new_tokens`

```python
max_new_tokens=100
```

Maximum number of **new tokens** generated.

It does not mean maximum total sequence length.

---

# 37. Sampling

```python
do_sample=True
```

Enables probabilistic sampling.

If:

```python
do_sample=False
```

generation becomes more deterministic.

---

# 38. Temperature

```python
temperature=0.7
```

Controls randomness.

Conceptually:

```text
Lower temperature
      ↓
more deterministic

Higher temperature
      ↓
more random
```

---

# 39. Repetition Penalty

```python
repetition_penalty=1.1
```

Helps discourage excessive repetition during generation.

---

# 40. EOS Token

```python
eos_token_id=tokenizer.eos_token_id
```

Tells generation which token represents the end of the sequence.

Generation can stop when EOS is produced.

---

# 41. Decode Output

```python
answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
```

Generation produces token IDs.

We convert them back:

```text
Token IDs
   ↓
tokenizer.decode()
   ↓
Human-readable text
```

`skip_special_tokens=True` removes special tokens from the displayed output.

---

# 42. Print Answer

```python
print("\nTinyLlama:", answer)
```

The complete inference loop is therefore:

```text
Question
   ↓
Tokenizer
   ↓
Input IDs
   ↓
LoRA + Base Model
   ↓
generate()
   ↓
Output Token IDs
   ↓
decode()
   ↓
Answer
```

---

# 43. COMPLETE CODE MEMORY MAP

For revision, remember the code in these blocks:

```text
1. Imports
      ↓
2. GPU check
      ↓
3. model_name
      ↓
4. tokenizer
      ↓
5. dataset
      ↓
6. tokenize
      ↓
7. data_collator
      ↓
8. base model
      ↓
9. use_cache=False
      ↓
10. LoraConfig
      ↓
11. get_peft_model
      ↓
12. print_trainable_parameters
      ↓
13. TrainingArguments
      ↓
14. Trainer
      ↓
15. trainer.train()
      ↓
16. save_pretrained()
```

Inference:

```text
1. Load base model
      ↓
2. PeftModel.from_pretrained()
      ↓
3. Load tokenizer
      ↓
4. model.eval()
      ↓
5. tokenizer(question)
      ↓
6. model.generate()
      ↓
7. tokenizer.decode()
```

---

# 44. The Most Important Code to Memorize

You don't need to memorize every line.

Understand these core APIs:

### Load tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Tokenize

```python
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
```

### Load model

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Configure LoRA

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)
```

### Attach LoRA

```python
model = get_peft_model(model, lora_config)
```

### Train

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
```

### Save

```python
model.save_pretrained("./tinyllama-lora-final")
```

### Load adapter

```python
base_model = AutoModelForCausalLM.from_pretrained(model_name)

model = PeftModel.from_pretrained(
    base_model,
    "./tinyllama-lora-final"
)
```

### Generate

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100
)
```

---

# 45. What You Actually Need to Change for a New Project

Usually these are the main things:

### 1. Model

```python
model_name = "YOUR_MODEL"
```

### 2. Dataset

Replace:

```python
texts = [...]
```

with your real dataset.

### 3. Text column

If your dataset has:

```text
question
answer
```

or:

```text
content
```

you need to construct the text you want the causal LM to train on.

### 4. Sequence length

```python
MAX_LENGTH = 256
```

Change according to your data and GPU memory.

### 5. LoRA target modules

```python
target_modules=["q_proj", "v_proj"]
```

These depend on the architecture of the model.

### 6. Training hyperparameters

Main ones:

```python
num_train_epochs
per_device_train_batch_size
gradient_accumulation_steps
learning_rate
```

---

# 46. Training vs Inference — Don't Mix Them

## Training

```text
Raw text
 ↓
Tokenizer
 ↓
Dataset
 ↓
Data collator
 ↓
Base model + LoRA
 ↓
Loss
 ↓
Backpropagation
 ↓
Update LoRA
 ↓
Save adapter
```

## Inference

```text
Question
 ↓
Tokenizer
 ↓
Base model + trained LoRA
 ↓
generate()
 ↓
Decode
 ↓
Answer
```

No:

```python
trainer.train()
```

during inference.

And normally no gradient calculation:

```python
with torch.no_grad():
```

---

# 47. Common Debugging Points

### CUDA error

Check:

```python
torch.cuda.is_available()
```

and:

```python
torch.cuda.get_device_name(0)
```

---

### Out of memory

First reduce:

```python
per_device_train_batch_size
```

or:

```python
MAX_LENGTH
```

You can also increase:

```python
gradient_accumulation_steps
```

to maintain a larger effective batch size.

---

### Padding error

Use:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

---

### Wrong LoRA target modules

If:

```python
target_modules=["q_proj", "v_proj"]
```

doesn't match the model architecture, inspect the model's module names before configuring LoRA.

---

### Training too slow

Check:

```python
torch.cuda.is_available()
```

and verify that the model is actually using the GPU.

---

### Too much memory

Check:

```python
torch_dtype=torch.float16
```

and use a smaller batch/sequence length if necessary.

---

# 48. One-Minute Revision

If you have only one minute before an interview, remember this:

```text
MODEL
AutoModelForCausalLM
        ↓
TOKENIZER
AutoTokenizer
        ↓
DATA
Dataset
        ↓
TOKENIZE
dataset.map()
        ↓
COLLATOR
mlm=False
        ↓
LORA
LoraConfig(...)
        ↓
ATTACH
get_peft_model()
        ↓
TRAIN
Trainer(...)
trainer.train()
        ↓
SAVE
model.save_pretrained()
        ↓
INFERENCE
PeftModel.from_pretrained()
        ↓
generate()
        ↓
decode()
```

---

# 49. Key Code Questions You Should Be Able to Answer

### Q1. Why use `AutoModelForCausalLM`?

Because the task is causal/autoregressive language modeling.

### Q2. Why `mlm=False`?

Because we're training with next-token prediction rather than masked-token prediction.

### Q3. Why use LoRA?

To train a small set of adapter parameters instead of updating the entire base model.

### Q4. What does `get_peft_model()` do?

It attaches the PEFT/LoRA adapters to the base model.

### Q5. Why `print_trainable_parameters()`?

To verify that only a small fraction of parameters are being trained.

### Q6. What does `gradient_accumulation_steps=8` do?

Accumulates gradients over 8 mini-batches before performing an optimizer update.

### Q7. What does `PeftModel.from_pretrained()` do?

Loads the trained LoRA adapter onto the base model.

### Q8. Why `model.eval()`?

Switches the model to inference/evaluation behavior.

### Q9. Why `torch.no_grad()`?

Because gradients aren't needed during inference, reducing memory/computation.

### Q10. What is the difference between `max_length` and `max_new_tokens`?

`max_length` limits the tokenized sequence length during preprocessing, while `max_new_tokens` limits the number of tokens generated during inference.

---

# 50. Final Mental Model

Don't memorize the notebook line-by-line.

Remember the **code architecture**:

```text
              TRAINING
                 │
                 ▼
          ┌──────────────┐
          │ Base LLM     │
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │ Add LoRA     │
          └──────┬───────┘
                 │
                 ▼
       ┌────────────────────┐
       │ Tokenized Dataset  │
       └─────────┬──────────┘
                 │
                 ▼
            Trainer.train()
                 │
                 ▼
          ┌──────────────┐
          │ LoRA Adapter │
          └──────────────┘


              INFERENCE
                 │
       ┌─────────▼─────────┐
       │   Base LLM        │
       │       +           │
       │   LoRA Adapter    │
       └─────────┬─────────┘
                 │
                 ▼
             generate()
                 │
                 ▼
              decode()
                 │
                 ▼
              Answer
```

**The main thing to remember:**  
**Training learns the LoRA adapter. Inference loads the same base model + that adapter and uses `generate()`.**
