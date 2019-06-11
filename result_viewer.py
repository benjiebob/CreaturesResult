import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
import torch

from smal.smal3d_renderer import SMAL3DRenderer
from smal.joint_catalog import SMALJointInfo

ANIMAL_NAME = "maggie"
IMAGE_DIR = "/scratch/bjb56/data/animal_data/cosker/dog_sequences/{0}".format(ANIMAL_NAME)
RESULTS_DIR = "/scratch/bjb56/experiments/cosker/cosker-{0}/st10_ep0".format(ANIMAL_NAME)
IMAGE_SIZE = 256

def crop_silhouette(sil_img, target_size):
    assert len(sil_img.shape) == 2, "Silhouette image is not HxW"

    sil_h, sil_w = sil_img.shape
    pad_sil = np.zeros((sil_h * 4, sil_w * 4))
    pad_sil[sil_h * 2 : sil_h * 3, sil_w * 2 : sil_w * 3] = sil_img

    foreground_pixels = np.where(pad_sil > 0) # This is supposed to find foreground pixels in any input range
    y_min, y_max, x_min, x_max = np.amin(foreground_pixels[0]), np.amax(foreground_pixels[0]), np.amin(foreground_pixels[1]), np.amax(foreground_pixels[1])

    square_half_side_length = int(1.5 * (max(x_max - x_min, y_max - y_min) / 2))
    centre_y = y_min + int((y_max - y_min) / 2)
    centre_x = x_min + int((x_max - x_min) / 2)

    square_sil = pad_sil[centre_y - square_half_side_length : centre_y + square_half_side_length, centre_x - square_half_side_length : centre_x + square_half_side_length]
    sil_resize = cv2.resize(square_sil, (target_size, target_size), interpolation = cv2.INTER_NEAREST)
    
    return sil_resize

def draw_joints(image, joints, joint_info):
    image = (image * 255.0).astype(np.uint8)
    ret_image = image.copy()

    for joint_id, (y_co, x_co) in enumerate(joints):
        color = joint_info.joint_colors[joint_id]
        cv2.drawMarker(ret_image, (x_co, y_co), (int(color[0]), int(color[1]), int(color[2])), joint_info.marker_type[joint_id], 8, thickness = 3)

    return ret_image / 255.0

def main():
    model_renderer = SMAL3DRenderer(IMAGE_SIZE, 20).cuda()
    joint_info = SMALJointInfo()
    input_files = sorted(os.listdir(IMAGE_DIR))

    plt.ion()
    plt.figure()

    for frame_id, frame in enumerate(input_files):
        sil_img_orig = scipy.misc.imread(os.path.join(IMAGE_DIR, frame), mode = 'RGB')[:, :, 0] / 255.0
        pkl_file = os.path.join(RESULTS_DIR, "{0:04}.pkl".format(frame_id))

        sil_img = crop_silhouette(sil_img_orig, IMAGE_SIZE)
        
        params_cuda = {}
        with open(pkl_file, 'rb') as f:
            smal_params = pkl.load(f)
            for k, v in smal_params.items():
                params_cuda[k] = torch.from_numpy(v).unsqueeze(0).cuda()
 
        with torch.no_grad():
            rendered_images, rendered_silhouettes, rendered_joints = model_renderer(params_cuda)

            rendered_image_np = rendered_images[0].permute(1, 2, 0).cpu().numpy()
            rendered_sil_np = rendered_silhouettes.expand_as(rendered_images)[0][0].cpu().numpy()
            joints_np = rendered_joints[0].cpu().numpy()

            rendered_image_joints_overlay = draw_joints(rendered_image_np, joints_np, joint_info)
            
            error_img = np.abs(sil_img - rendered_sil_np)


        plt.suptitle("Result Viewer - Frame: {0:04}".format(frame_id))
        plt.subplot(131)
        plt.title("Input Silhouette")
        plt.imshow(sil_img)
        plt.subplot(132)
        plt.title("Rendered Image")
        plt.imshow(rendered_image_joints_overlay)
        plt.subplot(133)
        plt.title("Silhouette Error")
        plt.imshow(error_img)
        
        plt.draw()
        plt.pause(0.01)

        
if __name__ == "__main__":
    main()