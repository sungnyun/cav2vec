import os, random, cv2
import albumentations as A
import numpy as np
import torchvision
import torch
from skimage.util import random_noise
from pathlib import Path

class Visual_Corruption_Modeling:
    def __init__(self, d_image=None, d_mask=None, occ_type="coco"):
        self.pixelate = False
        self.occ_type = occ_type

        if self.occ_type == "coco":
            occlusion_patch_dir = Path(os.getcwd()).parent.parent.parent.parent.absolute()
            if d_image is None:
                d_image = Path(occlusion_patch_dir, 'occlusion_patch/object_image_sr')
            if d_mask  is None:
                d_mask = Path(occlusion_patch_dir, 'occlusion_patch/object_mask_x4')

            assert os.path.exists(d_image), "Please download coco_object.7z first"
            self.d_image = d_image
            self.d_mask = d_mask
            self.aug = get_occluder_augmentor()
            self.occlude_imgs = os.listdir(d_image)
        
        elif self.occ_type == "hands":
            occlusion_patch_dir = Path(os.getcwd()).parent.parent.parent.parent.absolute()
            if d_image is None:
                d_image = Path(occlusion_patch_dir, 'occlusion_patch/11k-hands_sr')
            if d_mask  is None:
                d_mask = Path(occlusion_patch_dir, 'occlusion_patch/11k-hands_masks')

            assert os.path.exists(d_image), "Please download 11k-hand image dataset first"
            self.d_image = d_image
            self.d_mask = d_mask
            self.aug = get_occluder_augmentor()
            self.occlude_imgs = os.listdir(d_image)

        elif self.occ_type == "pixelate":
            occlusion_patch_dir = Path(os.getcwd()).parent.parent.parent.parent.absolute()
            if d_image is None:
                d_image = Path(occlusion_patch_dir, 'occlusion_patch/object_image_sr')
            if d_mask  is None:
                d_mask = Path(occlusion_patch_dir, 'occlusion_patch/object_mask_x4')

            assert os.path.exists(d_image), "Please download coco_object.7z first"
            self.d_image = d_image
            self.d_mask = d_mask
            self.aug = get_occluder_augmentor()
            self.occlude_imgs = os.listdir(d_image)
            self.pixelate_snr = 5
        
        else:
            raise NotImplementedError

    def get_occluders(self):
        if self.occ_type == "pixelate":
            use_img = "apple_213935_0.jpeg"
            occlude_img = [img for img in self.occlude_imgs if use_img in img][0]
        else:
            occlude_img = np.random.choice(self.occlude_imgs)

        if 'jpeg' in occlude_img:
            occlude_mask = occlude_img.replace('jpeg', 'png')
        elif 'jpg' in occlude_img:
            occlude_mask = occlude_img.replace('jpg', 'png')

        ori_occluder_img = cv2.imread(os.path.join(self.d_image, occlude_img), -1)
        ori_occluder_img = cv2.cvtColor(ori_occluder_img, cv2.COLOR_BGR2RGB)

        occluder_mask = cv2.imread(os.path.join(self.d_mask, occlude_mask))
        occluder_mask = cv2.cvtColor(occluder_mask, cv2.COLOR_BGR2GRAY)

        occluder_mask = cv2.resize(occluder_mask, (ori_occluder_img.shape[1], ori_occluder_img.shape[0]),
                                    interpolation=cv2.INTER_LANCZOS4)

        occluder_img = cv2.bitwise_and(ori_occluder_img, ori_occluder_img, mask=occluder_mask)

        transformed = self.aug(image=occluder_img, mask=occluder_mask)
        occluder_img, occluder_mask = transformed["image"], transformed["mask"]

        if self.occ_type == 'hands':
            occluder_size = 96
            occluder_img = cv2.resize(occluder_img, (occluder_size, occluder_size), interpolation= cv2.INTER_LANCZOS4)
            occluder_img = cv2.rotate(occluder_img, cv2.ROTATE_180)
            occluder_mask = cv2.resize(occluder_mask, (occluder_size, occluder_size), interpolation= cv2.INTER_LANCZOS4)
            occluder_mask = cv2.rotate(occluder_mask, cv2.ROTATE_180)

        else:
            occluder_size = random.choice(range(20, 46))
            occluder_img = cv2.resize(occluder_img, (occluder_size, occluder_size), interpolation= cv2.INTER_LANCZOS4)
            occluder_mask = cv2.resize(occluder_mask, (occluder_size, occluder_size), interpolation= cv2.INTER_LANCZOS4)

        return occlude_img, occluder_img, occluder_mask

    def noise_sequence(self, img_seq, freq=1, return_indices=False):
        if freq == 1:
            len = img_seq.shape[0]
            occ_len = np.random.randint(int(len * 0.1), int(len * 0.5))
            start_fr = np.random.randint(0, len-occ_len)
            occ_indices = []

            if occ_len == 0:
                if return_indices:
                    return img_seq, np.full(img_seq.shape[0], False)
                return img_seq

            raw_sequence = img_seq[start_fr:start_fr+occ_len]
            prob = np.random.rand()
            if prob < 0.3:
                var = np.random.rand() * 0.2
                raw_sequence = np.expand_dims(raw_sequence, 3)
                raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True) * 255
                raw_sequence = raw_sequence.squeeze(3)
                occ_indices = list(range(start_fr, start_fr+occ_len))
            elif prob < 0.6:
                blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
                raw_sequence = np.expand_dims(raw_sequence, 3)
                raw_sequence = blur(torch.tensor(raw_sequence).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
                raw_sequence = raw_sequence.squeeze(3)
                occ_indices = list(range(start_fr, start_fr+occ_len))
            else:
                pass

            img_seq[start_fr:start_fr + occ_len] = raw_sequence

        else:
            len_global = img_seq.shape[0]
            len = img_seq.shape[0] // freq
            occ_indices = []
            for j in range(freq):
                try:
                    occ_len = np.random.randint(int(len_global * 0.3), int(len_global * 0.5))
                    start_fr = np.random.randint(0, len*j + len - occ_len)
                    if start_fr < len*j:
                        assert 1==2
                except:
                    occ_len = len // 2
                    start_fr = len * j

                if occ_len == 0:
                    if return_indices:
                        return img_seq, np.full(img_seq.shape[0], False)
                    return img_seq

                raw_sequence = img_seq[start_fr:start_fr + occ_len]
                prob = np.random.rand()
                if prob < 0.3:
                    var = np.random.rand() * 0.2
                    raw_sequence = np.expand_dims(raw_sequence, 3)
                    raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True) * 255
                    raw_sequence = raw_sequence.squeeze(3)
                    occ_indices += list(range(start_fr, start_fr+occ_len))
                elif prob < 0.6:
                    blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
                    raw_sequence = np.expand_dims(raw_sequence, 3)
                    raw_sequence = blur(torch.tensor(raw_sequence).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
                    raw_sequence = raw_sequence.squeeze(3)
                    occ_indices += list(range(start_fr, start_fr+occ_len))
                else:
                    pass

                img_seq[start_fr:start_fr + occ_len] = raw_sequence

        if return_indices:
            occ = np.full(img_seq.shape[0], False)
            occ_indices = np.unique(np.array(occ_indices, dtype=np.int64))
            occ[occ_indices] = True
            return img_seq, occ
        return img_seq


    def occlude_sequence(self, img_seq, landmarks, yx_min, freq=1, return_indices=False):
        if freq == 1:
            occlude_img, occluder_img, occluder_mask = self.get_occluders()

            len = img_seq.shape[0]
            start_pt_idx = np.random.randint(48, 67)
            offset = np.random.randint(10, 30)
            occ_len = np.random.randint(int(len * 0.1), int(len * 0.5))
            start_fr = np.random.randint(0, len-occ_len)
            occ_indices = []

            for i in range(occ_len):
                fr = cv2.cvtColor(img_seq[i+start_fr], cv2.COLOR_GRAY2RGB)
                x, y = landmarks[i + start_fr][start_pt_idx]

                if self.occ_type == "pixelate":
                    fr = self.image_pixelate(fr, occluder_img, int(y - yx_min[i + start_fr][0] - offset), int(x - yx_min[i + start_fr][1] - offset),
                                                pixel_size=self.pixelate_snr, pixelate_part = 'whole')
                else:
                    alpha_mask = np.expand_dims(occluder_mask, axis=2)
                    alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
                    if self.occ_type == 'hands':
                        fr = self.overlay_image_hands(fr, occluder_img, int(y - yx_min[i + start_fr][0] - offset), int(x - yx_min[i + start_fr][1] - offset), alpha_mask)
                    else:
                        fr = self.overlay_image_alpha(fr, occluder_img, int(y - yx_min[i + start_fr][0] - offset), int(x - yx_min[i + start_fr][1] - offset), alpha_mask)

                img_seq[i + start_fr] = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
                occ_indices.append(i+start_fr)

        else:
            len_global = img_seq.shape[0]
            len = img_seq.shape[0] // freq
            occ_indices = []
            for j in range(freq):
                occlude_img, occluder_img, occluder_mask = self.get_occluders()

                start_pt_idx = np.random.randint(48, 67)
                offset = np.random.randint(10, 30)
                try:
                    occ_len = np.random.randint(int(len_global * 0.3), int(len_global * 0.5))
                    start_fr = np.random.randint(0, len*j + len - occ_len)
                    if start_fr < len*j:
                        assert 1==2
                except:
                    occ_len = len // 2
                    start_fr = len * j

                for i in range(occ_len):
                    fr = cv2.cvtColor(img_seq[i + start_fr], cv2.COLOR_GRAY2RGB)
                    x, y = landmarks[i + start_fr][start_pt_idx]

                    if self.occ_type == "pixelate":
                        fr = self.image_pixelate(fr, occluder_img, int(y - yx_min[i + start_fr][0] - offset), int(x - yx_min[i + start_fr][1] - offset),
                                                    pixel_size=self.pixelate_snr, pixelate_part = 'whole')
                    else:
                        alpha_mask = np.expand_dims(occluder_mask, axis=2)
                        alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
                        if self.occ_type == 'hands':
                            fr = self.overlay_image_hands(fr, occluder_img, int(y - yx_min[i + start_fr][0] - offset), int(x - yx_min[i + start_fr][1] - offset), alpha_mask)
                        else:
                            fr = self.overlay_image_alpha(fr, occluder_img, int(y - yx_min[i + start_fr][0] - offset), int(x - yx_min[i + start_fr][1] - offset), alpha_mask)

                    img_seq[i + start_fr] = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
                    occ_indices.append(i+start_fr)

        if return_indices:
            occ = np.full(img_seq.shape[0], False)
            occ_indices = np.unique(np.array(occ_indices, dtype=np.int64))
            occ[occ_indices] = True
            return img_seq, occlude_img, occ
        return img_seq, occlude_img

    def overlay_image_alpha(self, img, img_overlay, y, x, alpha_mask):
        """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

        `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
        """
        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return img

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]

        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha
        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
        return img
    
    def overlay_image_hands(self, img, img_overlay, y, x, alpha_mask): # default x, y is wrong fixed it y, x
        """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.
        `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
        """
        # Pixelate - !HOTFIX: fixed to the below of center
        y1, y2 = 20, 96
        x1, x2 = 0, 96

        # Overlay ranges - !HOTFIX: fixed to the below of cente        
        y1o, y2o = 0, 76
        x1o, x2o = 0, 96

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return img

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]

        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha
        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
        return img
    
    def image_pixelate(self, img, img_overlay, y, x, pixel_size=10, pixelate_part='whole'):
        
        if pixelate_part == 'whole':
            height, width = img.shape[:2]
            # make low resolution image
            small_img = cv2.resize(img, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
            
            # rescale back to original size
            img = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)

        elif pixelate_part == 'mouth':
            # Image ranges
            # y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
            # x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

            # # Overlay ranges
            # y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
            # x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

            # # Exit if nothing to do
            # if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            #     return img

            # Pixelate - !HOTFIX: fixed to the center
            y1, y2 = 28, 68
            x1, x2 = 28, 68

            img_crop = img[y1:y2, x1:x2]
            small_img = cv2.resize(img_crop, (img_crop.shape[1] // pixel_size, img_crop.shape[0] // pixel_size), interpolation=cv2.INTER_LINEAR)
            pixelated_crop = cv2.resize(small_img, (img_crop.shape[1], img_crop.shape[0]), interpolation=cv2.INTER_NEAREST)

            img[y1:y2, x1:x2] = pixelated_crop
        return img

def get_occluder_augmentor():
    """
    Occludor augmentor
    """
    aug=A.Compose([
        A.AdvancedBlur(),
        A.OneOf([
            A.ImageCompression (quality_lower=70,p=0.5),
            ], p=0.5),
        A.Affine  (
            scale=(0.8,1.2),
            rotate=(-15,15),
            shear=(-8,8),
            fit_output=True,
            p=0.7
        ),
        A.RandomBrightnessContrast(p=0.5,brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=False),
        ])
    return aug
