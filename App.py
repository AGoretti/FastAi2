from fastbook import *
from fastai.vision.widgets import *
from time import sleep
import gradio as gr

learn_inf = load_learner(path/'export.pkl')

categories = ('pink lady apple', 'red delicious apple',  'royal gala apple')
def classify_image(img):
    pred,idx,probs = learned.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['maca.jpg', 'maca2.jpg', 'maca3.jpg']
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)    