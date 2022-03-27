import os
import gradio as gr
from predictor import TBClassifier
from dotenv import load_dotenv


if __name__ == '__main__':

  load_dotenv()
  HOST = os.getenv('HOST')
  PORT = int(os.getenv('PORT'))
  MODEL = os.getenv('MODEL')
  IMG_SIZE = int(os.getenv('IMG_SIZE'))
  NORMALIZE_HEATMAP = bool(os.getenv('NORMALIZE_HEATMAP'))
  HEATMAP_THRESHOLD = float(os.getenv('HEATMAP_THRESHOLD'))

  clf = TBClassifier(MODEL, IMG_SIZE, NORMALIZE_HEATMAP, HEATMAP_THRESHOLD)

  gr.close_all()

  gr.Interface(
    lambda gray_img: clf(gray_img), 
    gr.inputs.Image(image_mode='L'), 
    [
      gr.outputs.Label(label='TB probability'), 
      gr.outputs.Image(label='Heatmap')
    ],
    examples=['./examples/normal.png', './examples/tb.png'],
    allow_flagging='never'
    ).launch(
      server_name=HOST,
      server_port=PORT,
    )

