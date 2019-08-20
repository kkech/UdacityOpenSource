import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

class SimplePreprocessor:
    def __init__ (self, width, height, inter):
        # store the target image width, height and interpolation

        #image resize
        self.widht = widht
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size
        # ratio
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)
