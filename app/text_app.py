import gradio as gr

from src.text_classifier.inference import load_classifier


classifier = None


def load_model():
    global classifier
    try:
        classifier = load_classifier("src/text_classifier/model")
        return "Model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {str(e)}"


def predict(text: str) -> dict:
    if classifier is None:
        load_model()

    if not text.strip():
        return {"predicted_label": "", "probabilities": {}, "confidence": 0.0}

    result = classifier.predict(text)

    label = result["predicted_label"]
    confidence = result["confidence"]
    probs = result["probabilities"]

    prob_str = "\n".join([f"{k}: {v:.2%}" for k, v in probs.items()])

    return {
        "label": label,
        "confidence": f"{confidence:.2%}",
        "probabilities": prob_str,
    }


with gr.Blocks(title="Text Classifier") as demo:
    gr.Markdown("# Text Classification Demo")
    gr.Markdown("Enter text to classify it as positive, negative, or neutral.")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter your text here...",
                lines=5,
            )
            submit_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            label_output = gr.Label(label="Predicted Label")
            confidence_output = gr.Textbox(label="Confidence", lines=1)
            probs_output = gr.Textbox(label="All Probabilities", lines=5)

    submit_btn.click(
        fn=predict,
        inputs=text_input,
        outputs=[label_output, confidence_output, probs_output],
    )

    text_input.submit(
        fn=predict,
        inputs=text_input,
        outputs=[label_output, confidence_output, probs_output],
    )

    demo.load(fn=load_model, inputs=None, outputs=None)


if __name__ == "__main__":
    demo.launch()
