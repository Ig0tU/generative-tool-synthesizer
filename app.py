import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "google/flan-t5-large" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model and Tokenizer ---
tokenizer = None
model = None

try:
    print(f"Attempting to load model: {MODEL_NAME} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Move model to the appropriate device
    model.to(DEVICE)

    # Optional: Use half precision if GPU memory is a concern and using CUDA
    if DEVICE == "cuda":
        model = model.half()
        print("Using half precision on CUDA.")

    print(f"Model {MODEL_NAME} loaded successfully on {DEVICE}")

except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    print("Model loading failed. The application will not be able to synthesize tools.")
    tokenizer = None
    model = None

# --- List of Complex Tasks ---
COMPLEX_TASKS = [
    "Generative synthesis of tools for handling tasks with ambiguous or incomplete specifications",
    "Automated creation of tools for multi-modal data fusion and processing",
    "Dynamic generation of tools for adversarial robustness testing and defense",
    "Creation of tools for automated scientific hypothesis generation and experimental design",
    "Automated synthesis of tools for generating and manipulating abstract symbolic representations",
    "Dynamic generation of tools for real-time, low-latency task execution environments",
    "Creation of tools capable of operating effectively with limited or zero-shot data",
    "Automated synthesis of tools for complex multi-agent system simulation and analysis",
    "Dynamic generation of tools for complex strategic planning under uncertainty",
    "Creation of tools for analyzing and influencing complex emergent system behaviors"
]

def synthesize_tool_concept(selected_task: str) -> str:
    """
    Uses a generative model to synthesize a conceptual description of a tool for the given complex task.
    """
    if model is None or tokenizer is None:
        return "Error: AI model not loaded. Cannot synthesize tool concept. Please check the Space logs for model loading errors."

    prompt = f"""Synthesize a conceptual description for a software tool designed to achieve the following complex task:
Task: "{selected_task}"

Describe the tool's purpose, potential key components, required data inputs, major technical challenges, and a possible high-level approach or architecture.
"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)

        outputs = model.generate(
            inputs.input_ids,
            max_length=700,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    except Exception as e:
        print(f"Error during generation: {e}")
        return f"An error occurred during tool synthesis: {e}"

# --- Gradio Interface ---
if model is not None and tokenizer is not None:
    iface = gr.Interface(
        fn=synthesize_tool_concept,
        inputs=gr.Dropdown(
            choices=COMPLEX_TASKS,
            label="Select a Complex Task for Tool Synthesis",
            value=COMPLEX_TASKS[0]
        ),
        outputs=gr.Markdown(label="Synthesized Tool Concept"),
        title="Generative Tool Synthesizer for Complex Tasks",
        description="Select a complex task from the list to generate a conceptual description of a potential software tool using a large language model. This demonstrates the *idea* of automated tool synthesis, not the creation of functional code.",
        allow_flagging="never"
    )

    if __name__ == "__main__":
        iface.launch()
else:
    print("Gradio interface not launched due to model loading failure.")
