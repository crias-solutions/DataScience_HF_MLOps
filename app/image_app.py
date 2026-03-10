import gradio as gr

from src.image_classifier.inference import load_classifier


classifier = None


def load_model():
    global classifier
    try:
        classifier = load_classifier("src/image_classifier/model")
        return "Model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {str(e)}"


def predict(image) -> dict:
    if classifier is None:
        load_model()

    if image is None:
        return {"label": "", "confidence": "0%", "probabilities": ""}

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp_path = tmp.name()

    try:
        image.save(tmp_path)
        result = classifier.predict(tmp_path)

        label = result["predicted_label"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        prob_str = "\n".join([f"{k}: {v:.2%}" for k, v in probs.items()])

        return {
            "label": label,
            "confidence": f"{confidence:.2%}",
            "probabilities": prob_str,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


with gr.Blocks(title="Image Classifier") as demo:
    gr.Markdown("# Image Classification Demo")
    gr.Markdown("Upload an image to classify it.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            label_output = gr.Label(label="Predicted Label")
            confidence_output = gr.Textbox(label="Confidence", lines=1)
            probs_output = gr.Textbox(label="All Probabilities", lines=5)

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, confidence_output, probs_output],
    )

    demo.load(fn=load_model, inputs=None, outputs=None)


if __name__ == "__main__":
    demo.launch()
