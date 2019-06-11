import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import torch.nn as nn
import neural_renderer as nr
import torch.nn.functional as F
from smal.smal_torch_batch import SMALModel
from smal.joint_catalog import SMALJointInfo

import pickle as pkl

import numpy as np
import cv2
import matplotlib.pyplot as plt

class SMAL3DRenderer(nn.Module):
    def __init__(self, image_size, n_betas, z_distance = 5.0, elevation = 89.9, azimuth = 0.0):
        super(SMAL3DRenderer, self).__init__()
        
        self.smal_model = SMALModel()
        self.image_size = image_size
        self.smal_info = SMALJointInfo()

        self.n_betas = n_betas

        self.renderer = nr.Renderer(camera_mode='look_at')
        self.renderer.eye = nr.get_points_from_angles(z_distance, elevation, azimuth)

        self.renderer.image_size = image_size
        self.renderer.light_intensity_ambient = 1.0
        self.renderer.viewing_angle = 10

        with open("smal/dog_texture.pkl", 'rb') as f:
            self.textures = pkl.load(f).cuda()

    def forward(self, batch_params):
        batch_size = batch_params['joint_rotations'].shape[0]
        global_rotation = batch_params['global_rotation'].clone()

        verts, joints_3d = self.smal_model(
            torch.cat([batch_params['betas'], torch.zeros(batch_size, 41 - self.n_betas).cuda()], dim = 1), # Pad remaining shape parameters with zeros
            torch.cat((global_rotation, batch_params['joint_rotations'].view(batch_size, -1)), dim = 1),
            batch_params['trans'])
    
        batch_size = batch_params['betas'].shape[0]
        faces = self.smal_model.faces.unsqueeze(0).expand(batch_size, -1, -1)
        textures = self.textures.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)

        rendered_joints = self.renderer.render_points(joints_3d[:, self.smal_info.include_classes])
        rendered_silhouettes = self.renderer.render_silhouettes(verts, faces)
        rendered_silhouettes = rendered_silhouettes.unsqueeze(1)

        rendered_images = self.renderer.render(verts, faces, textures)    
        rendered_images = torch.clamp(rendered_images[0], 0.0, 1.0)

        return rendered_images, rendered_silhouettes, rendered_joints