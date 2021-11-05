import cv2
import numpy as np
import math
import copy

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts


class Resize(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        image = cv2.resize(image, (self.size,
                                   self.size))
        scales = np.array([self.size / w, self.size / h])

        if polygons is not None:
            for polygon in polygons:
                polygon.center[0] = polygon.center[0]*scales[0]
                polygon.center[1] = polygon.center[1] * scales[1]
                polygon.points = polygon.points * scales

        return image, polygons



class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, polygons



class BaseTransform(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Resize(size),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)
class Padding(object):

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image, polygons=None):
        if np.random.randint(2):
            return image, polygons

        height, width, depth = image.shape
        ratio = np.random.uniform(1, 2)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
          (int(height * ratio), int(width * ratio), depth),
          dtype=image.dtype)
        expand_image[:, :, :] = self.fill
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        if polygons is not None:
            for polygon in polygons:
                polygon.points[:, 0] = polygon.points[:, 0] + left
                polygon.points[:, 1] = polygon.points[:, 1] + top
                polygon.center[0] = polygon.center[0] + left
                polygon.center[1] = polygon.center[1] + top
        return image, polygons


class Rotate(object):
    def __init__(self, up=30):
        self.up = up

    def rotate(self, center, pt, pt_c, theta):  # 二维图形学的旋转
        xr, yr = center
        yr = -yr
        x, y = pt[:, 0], pt[:, 1]
        xc= pt_c[0]
        yc= pt_c[1]
        y = -y
        yc= -yc

        theta = theta / 360 * 2 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        _x = xr + (x - xr) * cos - (y - yr) * sin
        _y = yr + (x - xr) * sin + (y - yr) * cos

        _xc = xr + (xc - xr) * cos - (yc - yr) * sin
        _yc = yr + (xc - xr) * sin + (yc - yr) * cos

        return _x, -_y,_xc,-_yc

    def __call__(self, img, polygons=None):
        if np.random.randint(2):
            return img, polygons
        angle = np.random.uniform(-self.up, self.up)  #
        rows, cols = img.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])
        center = cols / 2.0, rows / 2.0
        if polygons is not None:
            for polygon in polygons:
                x, y,xc,yc = self.rotate(center, polygon.points,polygon.center, angle)
                pts = np.vstack([x, y]).T
                polygon.points = pts
                polygon.center=[xc,yc]
        return img, polygons


class RandomMirror(object):
    def __init__(self):
        pass

    def __call__(self, image, polygons=None):
        if np.random.randint(2):
            image = np.ascontiguousarray(image[:, ::-1])
            _, width, _ = image.shape
            for polygon in polygons:
                polygon.center[0]=width-polygon.center[0]
                polygon.points[:, 0] = width - polygon.points[:, 0]
        return image, polygons


class RandomResizedLimitCrop(object):
    def __init__(self, size, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.extra=Resize(size)

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if np.random.random() < 0.5:
                w, h = h, w

            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, image, polygons=None):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)

        cropped = image[i:i + h, j:j + w, :]
        scales = np.array([self.size[0] / w, self.size[1] / h])
        polygons_=copy.copy(polygons)
        if polygons is not None:
            for polygon in polygons_:
                new_c_x = (polygon.center[0]-j)*scales[0]
                new_c_y = (polygon.center[1]-i)*scales[1]
                '''if new_c_x<0 or new_c_x>=h or new_c_y<0 or new_c_y>=w:

                    return self.extra(image,polygons)
                else:

                    '''
                polygon.center=[new_c_x,new_c_y]

                polygon.points[:, 0] = (polygon.points[:, 0] - j) * scales[0]
                polygon.points[:, 1] = (polygon.points[:, 1] - i) * scales[1]

        img = cv2.resize(cropped, self.size)
        return img, polygons_

class Augmentation(object):

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            #Resize(size),
            Padding(),
            RandomResizedLimitCrop(size=size, scale=(0.24, 1.0), ratio=(0.33, 3)),
            # RandomBrightness(),
            # RandomContrast(),
            RandomMirror(),
            Rotate(),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)