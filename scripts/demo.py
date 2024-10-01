import os
import urllib.request
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor

from models import get_model
from loader import vit_transforms
from SelectiveNet import SelectiveNet

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def test_and_show(img_dir, weight_dir):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    # open and transform image for vit
    image = Image.open(img_dir)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = ToTensor()(image)
    image_vit = vit_transforms(image)
    image_vit = image_vit.unsqueeze(0)
    image_vit = image_vit.to(device)

    # get model and predict
    features = get_model().float().to(device)
    model = SelectiveNet(features, 80).float().to(device)
    model.load_state_dict(torch.load(weight_dir, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        pred = model(image_vit)

    # plot
    plt.imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.title(f"Predicted BMI: {pred[0].item():.5f}, {pred[1].item():.5f}")
    plt.show()

    return pred



class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if __name__ == "__main__":
    if not os.path.exists("../weights"):
        os.makedirs("../weights")
        #weight_dir = "../weights/aug_epoch_7.pt"
        #url = "https://face-to-bmi-weights.s3.us-east.cloud-object-storage.appdomain.cloud/aug_epoch_7.pt"
        #print("dowloading weights...")
        #with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        #    urllib.request.urlretrieve(url, weight_dir, reporthook=t.update_to)

    pred = test_and_show('C:/Users/nguyen/TestProjects/selectivepred/facetobmi/face-to-bmi-vit/data/test_pic_rotate.jpg', 'C:/Users/nguyen/TestProjects/selectivepred/facetobmi/face-to-bmi-vit/weights/aug_epoch_8.pt')