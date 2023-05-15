import gradio as gr
import numpy as np
import json
import argparse
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model
from models.pspnet import PSPNet
from mindspore import context
import pt_transform as transform
import cv2
from PIL import Image

context.set_context(mode=context.GRAPH_MODE)


def colorize(gray, palette):
    """ gray: numpy array of the label and 1*3N size list palette """
    color = Image.fromarray(gray).convert('P')
    color.putpalette(palette)
    return color


def transforms(image):
    image = cv2.imread(image, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
    image = np.float32(image)
    h, w, c = image.shape
    image = cur_transform(image)
    image = image.astype(np.float32)
    image = image.reshape(1, 3, 473, 473)

    predict_score = model.predict(Tensor(image)).argmax(1).squeeze()
    predict_score = predict_score.asnumpy()

    gray = np.uint8(predict_score)
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)
    color = colorize(gray, colors)

    return color


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_classes = 21

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    cur_transform = transform.Compose([
        transform.Crop([473, 473], crop_type='center', padding=mean, ignore_label=255),
        transform.Normalize(mean=mean, std=std, is_train=True)])
    colors = np.loadtxt("voc2012_colors.txt").astype('uint8')

    with open('./label_dic_modify.json', 'r') as f:
        class_names = json.load(f)

    param_dict = load_checkpoint('./best_model.ckpt')
    network = PSPNet(num_classes=args.num_classes)
    load_param_into_net(network, param_dict)

    model = Model(network)
    image = gr.inputs.Image(type="filepath")
    label = gr.outputs.Image(type='numpy')

    gr.Interface(css=".footer {display:none !important}",
                 fn=transforms,
                 inputs=image,
                 live=False,
                 description="Please upload a image in JPG, JPEG or PNG.",
                 title='Image Segmentation by PSPNet',
                 outputs=label,
                 examples=['example_img/1.jpg', 'example_img/2.jpg', 'example_img/3.jpg',
                           'example_img/4.jpg', 'example_img/5.jpg', 'example_img/6.jpg',
                           'example_img/7.jpg']
                 ).launch()
