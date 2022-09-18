import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import TextVectorization

df = pd.read_csv('train.csv')
X = df['comment_text']
y = df[df.columns[2:]].values
MAX_FEATURES = 200000

vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

import gradio as gr

model = tf.keras.models.load_model('toxicity.h5')


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

    return text


interface = gr.Interface(fn=score_comment, capture_session=True,
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                         outputs='text')
interface.launch()