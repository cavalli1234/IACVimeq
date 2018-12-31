from data.img_io import load
from matplotlib.pyplot import imshow, show
from utils.naming import *
from skimage import exposure


def showimgs(imgs):
    for img in imgs:
        imshow(img)
        show()


if __name__ == '__main__':
    img = load(dataset_path())[0]
    img_eq = exposure.equalize_hist(img)
    img_adeq = exposure.equalize_adapthist(img)

    showimgs([img, img_eq, img_adeq])
