import os
import urllib.request
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor

from selectivenet import SelectiveNet
from model import HeightEstimationNet



def test_and_show(img_dir_fullbody, img_dir_face, weight_dir):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # open and transform image for vit
    image_fullbody = Image.open(img_dir_fullbody)
    image_face = Image.open(img_dir_face)
    if image_fullbody.mode != 'RGB':
        image_fullbody = image_fullbody.convert('RGB')
    if image_face.mode != 'RGB':
        image_face = image_face.convert('RGB')
    image_fullbody = ToTensor()(image_fullbody).to(device)
    image_face = ToTensor()(image_face).to(device)
    image_face = image_face.unsqueeze(0)
    image_fullbody = image_fullbody.unsqueeze(0)
    #image_vit = vit_transforms(image)
    #image_vit = image_vit.unsqueeze(0)
    #image_vit = image_vit.to(device)

    # get model and predict
    features = HeightEstimationNet().to(device)
    model = SelectiveNet(features=features)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_dir, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        pred = model(image_fullbody, image_face)

    # plot
    image_fullbody = image_fullbody.cpu().squeeze(0).detach().numpy().transpose(1, 2, 0)
    plt.imshow(image_fullbody)
    plt.axis("off")
    plt.title(f"Predicted Height: {pred}") #{pred.item():>5f}")
    plt.show()

    return pred.item()



class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if __name__ == "__main__":
    '''
    if not os.path.exists("../weights"):
        os.makedirs("../weights")
        weight_dir = "../weights/aug_epoch_7.pt"
        url = "https://face-to-bmi-weights.s3.us-east.cloud-object-storage.appdomain.cloud/aug_epoch_7.pt"
        print("dowloading weights...")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, weight_dir, reporthook=t.update_to)
    '''
    pred = test_and_show('C:/Users/nguyen/slurmgit/facetobmi_selectivenet-regression/rm999862784_fullbody.jpg', 'C:/Users/nguyen/slurmgit/facetobmi_selectivenet-regression/rm999862784.jpg', 'C:/Users/nguyen/slurmgit/facetobmi_selectivenet-regression/weights/checkpoint_selective_divine-dawn14.pt')
    print(f'Predicted Heigth and Uncertainty estimation: {pred}')