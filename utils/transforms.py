import cv2
import numpy as np
from torchvision import transforms
import random


class ToCV2Image(object):
    '''
    convert a CHW, range [0., 1.] tensor to a cv2 image
    '''
    def __init__(self, in_color='rgb',
                 out_color='bgr'):
        assert in_color in ['rgb', 'bgr']
        assert out_color in ['rgb', 'bgr']
        self.in_color = in_color
        self.out_color = out_color
        
    def __call__(self, im_tensor):
        cv2_img = (im_tensor.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        if self.in_color != self.out_color:
            cv2_img = cv2.img[:, :, ::-1]
        return cv2_img
        

class JointRandHrzFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def flip_single(self, image):
        return cv2.flip(image, 1)
    
    def __call__(self, img):
        assert isinstance(img, (np.ndarray, list, tuple))
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                flipped =  self.flip_single(img)
            else:
                flipped = []
                for each in img:
                    flipped.append(self.flip_single(each))
            return flipped
        else:
            return img


class JointRandVertFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def flip_single(self, image):
        return cv2.flip(image, 0)
    
    def __call__(self, img):
        assert isinstance(img, (np.ndarray, list, tuple))
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                flipped =  self.flip_single(img)
            else:
                flipped = []
                for each in img:
                    flipped.append(self.flip_single(each))
            return flipped
        else:
            return img

    
class JointResize(object):
    def __init__(self, size, interpolation='bilinear'):
        assert isinstance(size, (tuple, int))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        map_dict = {'bilinear': cv2.INTER_LINEAR,
                    'bicubic': cv2.INTER_CUBIC,
                    'nearest': cv2.INTER_NEAREST
                   }
        assert interpolation in map_dict.keys()
        self.inter_flag = map_dict[interpolation]

    def resize_single(self, image):
        return cv2.resize(image, self.size, interpolation=self.inter_flag)


    def __call__(self, img):
        assert isinstance(img, (np.ndarray, list, tuple))
        if isinstance(img, np.ndarray):
            resized = self.resize_single(img)
        else:
            resized = []
            for image in img:
                resized.append(self.resize_single(image))
        return resized

    
class JointToTensor(object):
    def __init__(self):
        self.to_tensor_single = transforms.ToTensor()
        
    def __call__(self, img):
        assert isinstance(img, (np.ndarray, list, tuple))
        if isinstance(img, np.ndarray):
            im_tensor = self.to_tensor_single(img)
        else:
            im_tensor = []
            for image in img:
                im_tensor.append(self.to_tensor_single(image))
        return im_tensor

    
class JointNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.normalize_single = transforms.Normalize(mean, std, inplace)
        
    def __call__(self, img):
        assert isinstance(img, (np.ndarray, list, tuple))
        if isinstance(img, np.ndarray):
            normalized = self.normalize_single(img)
        else:
            normalized = []
            for image in img:
                normalized.append(self.normalize_single(image))
        return normalized

    
class JointRandCrop(object):
    def __init__(self, size):
        pass
    def __call__(self, img):
        pass
    

# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/2
class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

    
class Binarize(object):
    def __init__(self, threshold=125):
        assert isinstance(threshold, (int, float))
        self.threshold = threshold
        
    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        return (img > self.threshold).astype(img.dtype)
    
