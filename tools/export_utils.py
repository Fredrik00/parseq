import yaml
from PIL import Image
from torchvision import transforms as T

from strhub.models.parseq.system import PARSeq
from strhub.models.utils import get_pretrained_weights


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


def prepare_export_model(model_name, export_mode):
    with open(f"configs/model/{model_name}.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print(cfg)
    charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    lightning_model = PARSeq(
        charset, charset, 25, 1, img_size=(32, 128),
        warmup_pct=0.075, weight_decay=0.0, **cfg
    )

    model = lightning_model.model
    state_dict = get_pretrained_weights(model_name)
    model.load_state_dict(state_dict)

    model.export_mode = export_mode
    return model.eval()
