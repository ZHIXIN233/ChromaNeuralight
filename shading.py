# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

from brdf import BRDFFactory
from light import LightFactory
import torch
from torch import nn
from torch import Tensor
# from utils import CameraPose
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lietorch import SO3

class ShadingModel(nn.Module):
    def __init__(
        self,
        brdf: str = "Microfacet",
        light: str = "Gaussian2D",
        albedo: float = 100.,
        device: str = "cpu",
        color_mode: str = "Grayscale",
    ) -> None:
        super(ShadingModel, self).__init__()
        self.color_mode = color_mode
        self.channels = 3 if color_mode == "RGB" else 1
        self.light = LightFactory.get_light(light, channels=self.channels)
        self.brdf = BRDFFactory.get_brdf(brdf)
        self.albedo_log = nn.Parameter(torch.tensor(albedo), requires_grad=True)
        self.ambient_light_log = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        # target's own coordinate system used as world coordinate. Right-Down-Forward. Hardcoded here that normal pointing from the origin of the target to camera.
        self.normal = torch.tensor([0.,0.,-1.], device=device)
        self.scaling_factor = nn.Parameter(torch.tensor(1.0), requires_grad=False) # When calibrating, should not optimize this scaling factor

    @property
    def albedo(self):
        return torch.exp(self.albedo_log)
    
    def set_albedo(self, albedo: float, require_grad: bool = True) -> None:
        self.albedo_log = nn.Parameter(torch.log(torch.tensor(albedo)), requires_grad=require_grad)
    
    @property
    def ambient_light(self):
        return torch.exp(self.ambient_light_log)
    
    def set_ambient_light(self, ambient_light: float, require_grad: bool = True) -> None:
        self.ambient_light_log = nn.Parameter(torch.log(torch.tensor(ambient_light)), requires_grad=require_grad)

    def forward(self, pts: Tensor, rvec_w2c: Tensor, t_w2c: Tensor)-> Tensor:
        '''
        Arguments:
            pts: 3D points in world coordinate.
        '''

        R_w2c = SO3.exp(rvec_w2c)
        R_c2w = R_w2c.inv()
        t_c2w = -R_c2w.act(t_w2c)

        p_l_in_w = R_c2w.act(self.light.t_l2c())+t_c2w
        light_dir = p_l_in_w-pts
        view_dir = t_c2w-pts

        reflectance = self.brdf(view_dir, self.normal[None, None], light_dir)

        pts_in_cam = R_w2c.act(pts)+t_w2c

        incident_light = self.light(pts_in_cam) # convert to camera coordinate first
        reflected_light = self.albedo*reflectance*(incident_light+self.ambient_light)
        return torch.clamp(reflected_light, 0.0, 255.0)

    def forward_mono(self, pts: Tensor, rvec_w2c: Tensor, t_w2c: Tensor)-> Tensor:
        R_w2c = SO3.exp(rvec_w2c)
        R_c2w = R_w2c.inv()
        t_c2w = -R_c2w.act(t_w2c)

        p_l_in_w = R_c2w.act(self.light.t_l2c())+t_c2w
        light_dir = p_l_in_w-pts
        view_dir = t_c2w-pts

        reflectance = self.brdf(view_dir, self.normal[None, None], light_dir)

        pts_in_cam = R_w2c.act(pts)+t_w2c

        if hasattr(self.light, "mono_intensity"):
            incident_light = self.light.mono_intensity(pts_in_cam)
        else:
            incident_light = self.light(pts_in_cam)
            if incident_light.ndim == 3:
                incident_light = incident_light.mean(dim=-1)
        reflected_light = self.albedo*reflectance*(incident_light+self.ambient_light)
        return torch.clamp(reflected_light, 0.0, 255.0)
