import cv2
import os
import numpy as np
from matplotlib.pyplot import imshow

# Definir rutas de carpetas
path_cimage_esm = 'test/cimages/extrasmall'
path_cimage_sm = 'test/cimages/small'
path_cimage_med = 'test/cimages/medium'
path_cimage_large = 'test/cimages/large'
path_cimage_elarge = 'train/cimages/extralarge'

path_cmask_esm = 'train/cmasks/extrasmall'
path_cmask_sm = 'train/cmasks/small'
path_cmask_med = 'train/cmasks/medium'
path_cmask_large = 'train/cmasks/large'
path_cmask_elarge = 'train/cmasks/extralarge'

dir_seg_train = 'train/segmentation_masks'
dir_img_train = 'train/images'

input_height = 256
input_width = 256
output_height = 256
output_width = 256

def maximum(a, b, c):
    if (a >= b) and (a >= c):
        return a
    elif (b >= a) and (b >= c):
        return b
    else:
        return c

ldseg = os.listdir(dir_seg_train)

esmall = small = med = large = elarge = 0

for s in ldseg:
    if "pre" in s:
        seg_pre = cv2.imread(os.path.join(dir_seg_train, s))
        seg_post = cv2.imread(os.path.join(dir_seg_train, s.replace("pre", "post")))

        color_seg_pre = seg_pre.copy()
        color_seg_post = seg_post.copy()

        img_is_train_pre = cv2.imread(os.path.join(dir_img_train, s))
        img_is_train_post = cv2.imread(os.path.join(dir_img_train, s.replace("pre", "post")))

        gray = cv2.cvtColor(seg_pre, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
        inverted_binary = cv2.bitwise_not(binary)

        contours, _ = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if 5 < cv2.contourArea(c) < 10300000:
                x, y, w, h = cv2.boundingRect(c)

                cropped_mask_pre = binary[y:y+h, x:x+w]
                cropped_mask_post = color_seg_post[y:y+h, x:x+w]
                cropped_image_pre = img_is_train_pre[y:y+h, x:x+w]
                cropped_image_post = img_is_train_post[y:y+h, x:x+w]

                cropped_image_pre = cv2.bitwise_and(cropped_image_pre, cropped_image_pre, mask=cropped_mask_pre)
                tmp_pre = cv2.cvtColor(cropped_image_pre, cv2.COLOR_BGR2GRAY)
                _, alpha_pre = cv2.threshold(tmp_pre, 0, 255, cv2.THRESH_BINARY)
                b_pre, g_pre, r_pre = cv2.split(cropped_image_pre)
                rgba_pre = [b_pre, g_pre, r_pre, alpha_pre]
                cropped_image_pre = cv2.merge(rgba_pre, 4)

                cropped_image_post = cv2.bitwise_and(cropped_image_post, cropped_image_post, mask=cropped_mask_pre)
                tmp_post = cv2.cvtColor(cropped_image_post, cv2.COLOR_BGR2GRAY)
                _, alpha_post = cv2.threshold(tmp_post, 0, 255, cv2.THRESH_BINARY)
                b_post, g_post, r_post = cv2.split(cropped_image_post)
                rgba_post = [b_post, g_post, r_post, alpha_post]
                cropped_image_post = cv2.merge(rgba_post, 4)

                # Color mean thresholds
                upper_threshold = 230
                med_threshold = 150
                down_threshold = 110
                udown_threshold = 30

                n_r = n_g = n_b = 0
                f_channel = s_channel = t_channel = 0

                for i in cropped_mask_post[..., 0].flatten():
                    if i != 0:
                        f_channel += i
                        n_b += 1
                for i in cropped_mask_post[..., 1].flatten():
                    if i != 0:
                        s_channel += i
                        n_g += 1
                for i in cropped_mask_post[..., 2].flatten():
                    if i != 0:
                        t_channel += i
                        n_r += 1

                n = maximum(n_r, n_g, n_b)
                if n != 0:
                    mean_b = f_channel / n
                    mean_g = s_channel / n
                    mean_r = t_channel / n
                else:
                    mean_b = mean_g = mean_r = 0

                n_colors = 0
                if (
                    (mean_r < udown_threshold and mean_g > upper_threshold and mean_b < udown_threshold) or
                    (mean_r > upper_threshold and mean_g < udown_threshold and mean_b < udown_threshold) or
                    (mean_r > upper_threshold and med_threshold > mean_g > down_threshold and mean_b < udown_threshold) or
                    (mean_r > upper_threshold and mean_g > upper_threshold and mean_b < udown_threshold)
                ):
                    n_colors = 1

                def save_crop(path_img, path_mask, index):
                    base_img_name = f'CROPPED_IMAGE{index}_{s}'
                    base_mask_name = f'CROPPED_MASK{index}_{s}'
                    cv2.imwrite(os.path.join(path_img, base_img_name), cv2.resize(cropped_image_pre, (input_height, input_width)))
                    cv2.imwrite(os.path.join(path_img, base_img_name.replace("pre", "post")), cv2.resize(cropped_image_post, (input_height, input_width)))
                    print(os.path.join(path_img, base_img_name))

                    cv2.imwrite(os.path.join(path_mask, base_mask_name), cv2.resize(cropped_mask_pre, (output_height, output_width)))
                    cv2.imwrite(os.path.join(path_mask, base_mask_name.replace("pre", "post")), cv2.resize(cropped_mask_post, (output_height, output_width)))
                    print(os.path.join(path_mask, base_mask_name))

                if n_colors == 1:
                    area = cv2.contourArea(c)
                    if area <= 500:
                        save_crop(path_cimage_esm, path_cmask_esm, esmall)
                        esmall += 1
                    elif area <= 1000:
                        save_crop(path_cimage_sm, path_cmask_sm, small)
                        small += 1
                    elif area <= 2000:
                        save_crop(path_cimage_med, path_cmask_med, med)
                        med += 1
                    elif area <= 209800:
                        save_crop(path_cimage_large, path_cmask_large, large)
                        large += 1
                    elif area <= 10300000:
                        save_crop(path_cimage_elarge, path_cmask_elarge, elarge)
                        elarge += 1
