from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import tensorflow as tf

# Load GPT-2 model
model_name = "gpt2"  # or the path to your downloaded GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Fine-tune on Shakespeare dataset (replace with your actual dataset loading)
# For simplicity, let's assume you have a list of Shakespearean texts in 'shakespeare_data'
shakespeare_data = [
    "To be or not to be, that is the question.",
    "All the world's a stage, and all the men and women merely players.",
    # Add more Shakespearean texts
]

# Tokenize and format the dataset
input_ids = tokenizer.encode("\n\n".join(shakespeare_data), return_tensors="tf")

# Labels are shifted by one position for language modeling
labels = tf.concat([input_ids[:, 1:], tf.zeros_like(input_ids[:, :1])], axis=-1)

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train the model (replace with your training loop)
# For simplicity, we'll use a single training iteration
model.fit(input_ids, labels)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_shakespeare_model")

# Generate 5 examples
for _ in range(5):
    prompt = "To be or not to be, that is the question."
    input_ids = tokenizer.encode(prompt, return_tensors="tf")

    # Generate text
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode and print generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)