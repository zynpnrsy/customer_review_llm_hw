import gradio as gr
from inference import predict

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Review Sentiment Analysis  
        BERT model trained with manual PyTorch loop  
        *Developed by zeyneppinarsoy*
        """
    )

    with gr.Row():
        with gr.Column():
            review_input = gr.Textbox(
                lines=4,
                placeholder="Enter a review...",
                label="Review Text"
            )

            submit_btn = gr.Button("Analyze Sentiment")

        with gr.Column():
            sentiment_output = gr.Label(label="Prediction")

    gr.Examples(
        examples=[
            "This product is amazing, I loved it!",
            "Very bad quality, totally disappointed.",
            "It's okay, not great but not terrible."
        ],
        inputs=review_input
    )

    submit_btn.click(
        fn=predict,
        inputs=review_input,
        outputs=sentiment_output
    )

demo.launch()
