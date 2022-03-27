import numpy as np
import cv2
import onnxruntime as ort
from albumentations import Compose, Resize, Normalize


MEAN = 0.4984
SD = 0.2483


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def overlay_cam(img, cam, normalize=False, threshold=0, weight=0.5, img_max=255.):
        """
        Red is the most important region
        Args:
            img: numpy array (h, w) or (h, w, 3)
        """
            
        if len(img.shape) == 2:
            h, w = img.shape
            img = img.reshape(h, w, 1)
            img = np.repeat(img, 3, axis=2)

        h, w, c = img.shape

        # normalize the cam
        x = sigmoid(cam)
        x[x<threshold] = 0

        if normalize:
            x = x - x.min()
            x = x / x.max()
        # resize the cam
        x = cv2.resize(x, (w, h))

        if normalize:
            x = x - x.min()
            x = x / x.max()

        # coloring the cam
        x = cv2.applyColorMap(np.uint8(255 * (1 - x)), cv2.COLORMAP_JET)
        x = np.float32(x) / 255.

        # overlay
        x = img / img_max + weight * x
        x = x / x.max()
        x = np.uint8(255 * x)
        return x


class TBClassifier():


    def __init__(self, onnx_model_path, img_size, normalize_heatmap, heatmap_threshold):
        self.ort_session = ort.InferenceSession(onnx_model_path)
        self.img_size = img_size
        self.normalize_heatmap = normalize_heatmap
        self.heatmap_threshold = heatmap_threshold
        self.transform = Compose([
                                  Resize(img_size, img_size, cv2.INTER_CUBIC),
                                  Normalize(MEAN, SD)
                                ])
        self.sigmoid = sigmoid
        self.overlay_cam = overlay_cam


    def __call__(self, gray_img):
        transformed = self.transform(image=gray_img)['image']
        outputs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: transformed[None, None]},)
        prob = self.sigmoid(outputs[0]).item()
        prob = f'{prob*100:.2f} %'
        cam = self.overlay_cam(gray_img, outputs[1][0][0], self.normalize_heatmap, self.heatmap_threshold)
        return prob, cam


    
    
