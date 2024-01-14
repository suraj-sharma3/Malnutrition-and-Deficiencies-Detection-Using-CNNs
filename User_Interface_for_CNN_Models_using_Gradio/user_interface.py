import gradio as gr
from utils import *

# Gradio Interface

def main():
    model_list = ["Malnutrition_Type_Predictor", "Skin_Condition_Predictor", "Eye Condition Predictor", "Iodine Deficiency Predictor" ]
  
    io = gr.Interface(
        fn=predict,
        inputs=[gr.File(label = "Upload an image of a Baby aged between 0 to 5 years", file_types=["image"]),
                gr.Dropdown(label="Select Model", choices=model_list)],
        outputs=[gr.Image(label="Uploaded Image", width=400, height=400),
                 gr.Textbox(label="Chosen CNN Model's Prediction")],
        allow_flagging="manual",
        flagging_options=["Save"],
        title="CNN-Powered Infant Malnutrition Detection System",
        description="Predict the Malnutrition Type and Presence of Vitamin B7, Iron, Vitamin A and Iron Deficiencies",
        theme=gr.themes.Soft()
    )
    io.launch(share=True)

if __name__ == "__main__":
    main()