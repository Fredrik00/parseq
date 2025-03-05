# Requirements listed in requirements/tflite.txt

import os
from os import path

# PyTorch export does not support fused attention as of version 2.0
os.environ['TIMM_FUSED_ATTN'] = '0'

import ai_edge_torch
import yaml

from PIL import Image
from strhub.models.parseq.system import PARSeq
from torchvision import transforms as T

def get_transform(img_size=(32, 128)):
    return T.Compose([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])

# Gray dummy image
def get_dummy_input(img_h=32, img_w=128, n_channels=3):
    transform = get_transform((img_h, img_w))
    image = Image.new('RGB', (img_w, img_h), color=128)
    image = transform(image)
    image = image.view(1, *image.size())
    return image

if __name__ == '__main__':
    model_name = "parseq"
    output_dir = "tflite"
    output_path = path.join(output_dir, f"{model_name}.tflite")

    with open("configs/model/parseq.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print(cfg)
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lightning_model = PARSeq(
        charset, charset, 25, 1, img_size=(32, 128),
        warmup_pct=0.075, weight_decay=0.0, **cfg
    )

    lightning_model.model.export_mode = True
    lightning_model.eval()

    image = get_dummy_input()

    # Test model forward pass for debugging purposes
    lightning_model(image)

    tf_model_dir = path.join(output_dir, model_name)
    edge_model = ai_edge_torch.convert(lightning_model, (image,), strict_export=True, _saved_model_dir=tf_model_dir)
    edge_model.export(output_path)
