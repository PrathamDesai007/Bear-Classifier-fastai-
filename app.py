#!/usr/bin/env python
# coding: utf-8

# # Bear Classifier
# 
# This is a prototype tool to deploy a model which classifies 3 bear categories namely Black, Grizzly and Teddy (Toys)
# 
# Upload a picture of a bear and click classify to the results



from fastai.vision.all import *
import gradio as gr
import skimage


learn_inf = load_learner('bear_model.pkl')
labels = learn_inf.dls.vocab


def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn_inf.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3), title = "Bear Classifier",
description = "A Bear Classifier trained with fastai. Created as a demo for Gradio and HuggingFace Spaces.",interpretation='default').launch(share=True)

