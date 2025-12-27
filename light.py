# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
from typing import TypeAlias, Literal
from torch import Tensor
from lietorch import SO3, LieGroupParameter

class LightBaseLie(nn.Module):
    name = "Light Base"
    def __init__(
        self,
        t_x: float = -0.2266,
        t_y: float = -0.0022,
        t_z: float = 0.0761,
        r_x: float = -0.0027,
        r_y: float = 0.36,
        r_z: float = -0.027,
        channels: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        # Translation vector and Rotation vector are light-to-camera transformation.
        # Light is using same convention to camera coordinate: x-right, y-down, z-forward
        # Apply following transformation will transform [ ] from light coordinate to camera coordiante
        self._t_vec = nn.Parameter(torch.tensor([[[t_x, t_y, t_z]]], device="cuda:0", dtype=torch.float64), requires_grad=True) #[-0.4, 0.0, 0.0]
        
        _r_vec = torch.tensor([[[r_x, r_y, r_z]]], device="cuda:0", dtype=torch.float64)
        self._r_l2c_SO3 = LieGroupParameter(SO3.exp((_r_vec)))

        self.gamma_log = nn.Parameter(torch.tensor(-1.59), requires_grad=True)
        self.tau_log = nn.Parameter(torch.tensor(-1.59), requires_grad=True)
        self.light_color_log = nn.Parameter(torch.zeros(self.channels, device="cuda:0"), requires_grad=True)

    def set_t_vec(self, t_tuple: tuple[float, float, float], require_grad: bool = True)->None:
        self._t_vec = nn.Parameter(torch.tensor([[[t_tuple[0], t_tuple[1], t_tuple[2]]]], device="cuda:0", dtype=torch.float64), requires_grad=require_grad)

    def set_r_vec(self, r_tuple: tuple[float, float, float])->None:
        _r_vec = torch.tensor([[[r_tuple[0], r_tuple[1], r_tuple[2]]]], device="cuda:0", dtype=torch.float64)
        self._r_l2c_SO3 = LieGroupParameter(SO3.exp((_r_vec)))

    def set_sigma(self, sigma: list[float], require_grad: bool = True)->None:
        pass

    @property
    def gamma(self):
        return torch.exp(self.gamma_log)

    def set_gamma(self, gamma: float, require_grad: bool = True)->None:
        self.gamma_log = nn.Parameter(torch.log(torch.tensor(gamma)), requires_grad=require_grad)

    @property
    def tau(self):
        return torch.exp(self.tau_log)

    def set_tau(self, tau: float, require_grad: bool = True)->None:
        self.tau_log = nn.Parameter(torch.log(torch.tensor(tau)), requires_grad=require_grad)

    @property
    def light_color(self):
        return torch.exp(self.light_color_log)

    def set_light_color(self, light_color: list[float] | tuple[float, float, float], require_grad: bool = True)->None:
        if len(light_color) != self.channels:
            raise ValueError("Light color channels mismatch with current light setting.")
        self.light_color_log = nn.Parameter(torch.log(torch.tensor(light_color, device=self.light_color_log.device)), requires_grad=require_grad)
    
    def c2l(self, pts)-> Tensor:
        return self._r_l2c_SO3.inv().act(pts)
    
    def l2c(self, pts)-> Tensor:
        return self._r_l2c_SO3.act(pts)
    
    def t_c2l(self)-> Tensor:
        return -self.c2l(self._t_vec)

    def t_l2c(self)-> Tensor:
        return self._t_vec
    
    def lorenzian(self, x_in_l: Tensor)->Tensor:
        '''
            Lorentzian Function Model, used to model light fall off with distance
            Args:
                x_in_l: 3D point in the light's coordinate system
            Return:
                i_falloff: fall off of light intensity
        '''
        dist_square = torch.sum(x_in_l*x_in_l, dim=-1)
        i_falloff = 1/torch.pow(self.tau+dist_square, self.gamma)
        return i_falloff

    def apply_color(self, intensity: Tensor)->Tensor:
        if self.channels == 1:
            if intensity.ndim == 2:
                intensity = intensity[..., None]
            return intensity * self.light_color
        if intensity.ndim == 3:
            if intensity.shape[1] == self.channels and intensity.shape[-1] != self.channels:
                intensity = intensity.permute(0, 2, 1)
            if intensity.shape[-1] == self.channels:
                return intensity * self.light_color.view(1, 1, self.channels)
            return intensity[..., None] * self.light_color
        return intensity[..., None] * self.light_color
    
    @property
    def name(self)->str:
        return self.name

class LightMLPBase(LightBaseLie):
    name: str = "MLP"
    def __init__(self, channels: int = 1) -> None:
        super().__init__(channels=channels)

    def create_mlp(self, width: int = 20, input_dim: int = 1, output_dim: int = 1, depth: int = 2)->nn.Sequential:
        layers = [nn.Linear(input_dim, width), nn.Softplus()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(width, output_dim))
        layers.append(nn.Softplus())
        mlp = nn.Sequential(*layers)
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        return mlp
    
    def forward(self, pts: Tensor)-> Tensor:
        '''
        Args:
            pts: points in camera coordinate
        '''
        x_in_l = self.c2l(pts)+self.t_c2l()
        i_falloff = self.lorenzian(x_in_l).to(torch.float32)
        i_mlp = self.mlp_process(x_in_l).squeeze(-1)
        return self.apply_color(i_falloff*i_mlp)
    
    def mlp_process(self, x_in_l: Tensor)->Tensor:
        raise NotImplementedError("MLP Process not implemented")
                
        
class LightMLP1D(LightMLPBase):
    name = "1D MLP"
    # Not using due to nosies and peaks in MLP especially at transit of edge.
    def __init__(self, channels: int = 1) -> None:
        super().__init__(channels=channels)
        self.mu = nn.Parameter(torch.tensor(0.0, device="cuda:0", dtype=torch.float32), requires_grad=True)
        self.log_sigma = nn.Parameter(torch.tensor(-1.0, device="cuda:0", dtype=torch.float32), requires_grad=True)
        self.trunk = nn.Sequential(
            nn.Linear(5, 20),
            nn.Softplus(),
            nn.Linear(20, 20),
            nn.Softplus(),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(20, 1),
            nn.Softplus(),
        )
        self.chroma_head = nn.Sequential(
            nn.Linear(20, 3),
        )
        for module in [self.trunk, self.intensity_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        for m in self.chroma_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        self.chroma_clamp_enabled = False
        self.chroma_clamp = 0.15
        self.last_delta = None
    
    def mlp_process(self, x_in_l: Tensor)->Tensor:
        '''
            2D Gaussian model
            Args:
                x_in_l: 3D point in the light's coordinate system
        '''
        x_in_l = x_in_l/x_in_l[..., 2, None]
        x_y = (torch.square(x_in_l[..., 0])+torch.square(x_in_l[..., 1])).to(torch.float32)[..., None]
        x = torch.sqrt(x_y)

        # Positional encoding under test
        x_y1 = torch.cos(x)
        x_y2 = torch.cos(2*x)
        x_y3 = torch.cos(4*x)
        x_y4 = torch.cos(8*x)
        x_y  = torch.concat([x, x_y1, x_y2, x_y3, x_y4], dim=-1)

        h = self.trunk(x_y)

        sigma = torch.exp(self.log_sigma)
        envelope = torch.exp(- (x - self.mu) ** 2 / (2 * sigma ** 2))

        i_mono = self.intensity_head(h)

        i_mono = i_mono * envelope
        if self.channels == 1:
            return i_mono
        delta = self.chroma_head(h)
        delta = delta - delta.mean(dim=-1, keepdim=True)

        delta = delta.clamp(-20, 20)

        if self.chroma_clamp_enabled:
            delta = torch.clamp(delta, -self.chroma_clamp, self.chroma_clamp)
        self.last_delta = delta
        gain = torch.exp(delta)
        return i_mono * gain

    @property
    def sigma(self)->Tensor:
        return torch.exp(self.log_sigma)

    def set_mu(self, mu: float, require_grad: bool = True)->None:
        self.mu = nn.Parameter(torch.tensor(mu, device=self.mu.device, dtype=torch.float32), requires_grad=require_grad)

    def set_log_sigma(self, log_sigma: float, require_grad: bool = True)->None:
        self.log_sigma = nn.Parameter(torch.tensor(log_sigma, device=self.log_sigma.device, dtype=torch.float32), requires_grad=require_grad)

    def set_sigma(self, sigma: list[float], require_grad: bool = True)->None:
        self.set_mu(sigma[0], require_grad=require_grad)
        self.set_log_sigma(sigma[1], require_grad=require_grad)

    def forward(self, pts: Tensor)-> Tensor:
        x_in_l = self.c2l(pts)+self.t_c2l()
        i_falloff = self.lorenzian(x_in_l).to(torch.float32)
        i_mlp = self.mlp_process(x_in_l)
        if self.channels == 1:
            i_mlp = i_mlp.squeeze(-1)
            return i_falloff*i_mlp
        if i_mlp.ndim == 3 and i_mlp.shape[1] == 3 and i_mlp.shape[-1] != 3:
            i_mlp = i_mlp.permute(0, 2, 1)
        i_rgb = i_falloff[..., None] * i_mlp
        return self.apply_color(i_rgb)

    def mono_intensity(self, pts: Tensor)->Tensor:
        x_in_l = self.c2l(pts)+self.t_c2l()
        i_falloff = self.lorenzian(x_in_l).to(torch.float32)
        x_in_l = x_in_l/x_in_l[..., 2, None]
        x_y = (torch.square(x_in_l[..., 0])+torch.square(x_in_l[..., 1])).to(torch.float32)[..., None]
        x_y = torch.sqrt(x_y)

        x_y1 = torch.cos(x_y)
        x_y2 = torch.cos(2*x_y)
        x_y3 = torch.cos(4*x_y)
        x_y4 = torch.cos(8*x_y)
        x_y  = torch.concat([x_y, x_y1, x_y2, x_y3, x_y4], dim=-1)
        envelope = torch.exp(-0.5 * torch.square((x_y[..., 0] - self.mu) / (self.sigma + 1e-6)))[..., None]
        h = self.trunk(x_y)
        i_mono = self.intensity_head(h).squeeze(-1) * envelope
        return i_falloff * i_mono

    def set_chroma_clamp(self, enabled: bool, value: float)->None:
        self.chroma_clamp_enabled = enabled
        self.chroma_clamp = value

    def set_chroma_finetune(self, enabled: bool)->None:
        for param in self.trunk.parameters():
            param.requires_grad = not enabled
        for param in self.intensity_head.parameters():
            param.requires_grad = not enabled
        for param in self.chroma_head.parameters():
            param.requires_grad = enabled


# Experimental
class LightMLP2D(LightMLPBase):
    name = "2D MLP"
    # Not using due to nosies and peaks in MLP especially at transit of edge.
    def __init__(self, channels: int = 1) -> None:
        super().__init__(channels=channels)
        self.mlp = self.create_mlp(width=128, input_dim=22, output_dim=1, depth=4)
    
    def mlp_process(self, x_in_l: Tensor)->Tensor:
        '''
            2D Gaussian model
            Args:
                x_in_l: 3D point in the light's coordinate system
        '''
        x_in_l = x_in_l/x_in_l[..., 2, None]
        x_y = torch.abs((x_in_l[..., 0:2])).to(torch.float32)
        xy1 = torch.cos(x_y)
        xy2 = torch.cos(2*x_y)
        xy3 = torch.cos(4*x_y)
        xy4 = torch.cos(8*x_y)
        xy5 = torch.cos(16*x_y)
        xy1_ = torch.sin(x_y)
        xy2_ = torch.sin(2*x_y)
        xy3_ = torch.sin(4*x_y)
        xy4_ = torch.sin(8*x_y)
        xy5_ = torch.sin(16*x_y)
        # x_y = torch.concat([x_y, torch.sum(x_y, dim=-1, keepdim=True)], dim=-1)
        x_y  = torch.concat([x_y, xy1, xy2, xy3, xy4, xy5, xy1_, xy2_, xy3_, xy4_, xy5_], dim=-1)
        i_mlp = self.mlp(x_y)
        return i_mlp
    
class PointLightSource(LightBaseLie):
    name = "Point Light Souce"
    def __init__(self, channels: int = 1) -> None:
        super().__init__(channels=channels)

    def forward(self, pts: Tensor)->Tensor:
        x_in_l = self.c2l(pts)+self.t_c2l()
        return self.apply_color(self.lorenzian(x_in_l))
    

# Experimental
class Light2DGaussian(LightBaseLie):
    name = "Gaussian2D"
    def __init__(self, channels: int = 1) -> None:
        super().__init__(channels=channels)
        self.sigma = nn.Parameter(Tensor([13.4, 14.763]), requires_grad=True) # 2D distribution of light intensity

    def set_sigma(self, sigma: list[float], require_grad: bool = True)->None:
        self.sigma = nn.Parameter(Tensor([sigma[0], sigma[1]]), requires_grad=require_grad)

    def forward(self, pts: Tensor)-> Tensor:
        '''
        Args:
            pts: points in camera coordinate
        '''
        x_in_l = self.c2l(pts)+self.t_c2l()
        i_falloff = self.lorenzian(x_in_l)
        i_gaussian2d = self.gaussian2d(x_in_l)
        return self.apply_color(i_falloff*i_gaussian2d)
    
    def gaussian2d(self, x_in_l: Tensor)->Tensor:
        '''
            2D Gaussian model
            Args:
                x_in_l: 3D point in the light's coordinate system
        '''
        x_in_l = x_in_l/x_in_l[..., 2, None]
        x = torch.arctan(torch.abs(x_in_l[..., 0]))
        y = torch.arctan(torch.abs(x_in_l[..., 1]))
        i_gaussian2d = torch.exp(-(x*self.sigma[0])**2-(y*self.sigma[1])**2)
        return i_gaussian2d
    

class Light1DGaussian(LightBaseLie):
    name = "Gaussian 1D"
    def __init__(self, channels: int = 1) -> None:
        super().__init__(channels=channels)
        # Gaussian std
        self.sigma = nn.Parameter(torch.tensor(11.0), requires_grad=True) # 2D distribution of light intensity

        # self.max_cap = nn.Parameter(torch.tensor(0.55), requires_grad=True)

    def set_sigma(self, sigma: list[float], require_grad: bool = True)->None:
        self.sigma = nn.Parameter(torch.tensor(sigma[0]), requires_grad=require_grad)

    def forward(self, pts: Tensor)-> Tensor:
        '''
            Args:
                pts: points in camera coordinate
        '''
        x_in_l = self.c2l(pts)+self.t_c2l()
        i_falloff = self.lorenzian(x_in_l)
        i_gaussian1d = self.gaussian1d(x_in_l)
        return self.apply_color(i_falloff*i_gaussian1d)
    
    def gaussian1d(self, x_in_l: Tensor)->Tensor:
        '''
            2D Gaussian model
            Args:
                x_in_l: 3D point in the light's coordinate system
        '''
        x_in_l = x_in_l/x_in_l[..., 2, None]
        x = torch.arctan(torch.sqrt(x_in_l[..., 0]**2+x_in_l[..., 1]**2))
        i_gaussian1d = torch.exp(-(x*self.sigma)**2)
        # i_gaussian1d = torch.clamp(i_gaussian1d, torch.tensor(0.0, device="cuda:0"), F.sigmoid(self.max_cap))
        return i_gaussian1d
    

class LightFactory:
    Mode: TypeAlias = Literal['Gaussian2D', 'Gaussian1D', "PointLightSource", "1DMLP", "2DMLP"]
    @staticmethod
    def get_light(Light_type: Mode, channels: int = 1):
        if Light_type == "Gaussian2D":
            return Light2DGaussian(channels=channels)
        elif Light_type == "Gaussian1D":
            return Light1DGaussian(channels=channels)
        elif Light_type == "PointLightSource":
            return PointLightSource(channels=channels)
        elif Light_type == "1DMLP":
            return LightMLP1D(channels=channels)
        elif Light_type == "2DMLP":
            return LightMLP2D(channels=channels)
        else:
            raise ValueError(f"Light type {Light_type} not recognized!")
