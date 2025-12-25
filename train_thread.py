# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

import logging
from PySide6.QtCore import QThread, Signal
import torch

from config import GUIConfig, TrainingConfig
import torch
from config import TrainingConfig
from shading import ShadingModel
import torch.nn as nn
from tqdm import tqdm
from lietorch import SO3

class TrainingThread(QThread):
    update_images = Signal(object, object)
    update_shading_model_param = Signal(object, object, object, object, object, object, object)
    update_progress_bar = Signal(int)
    training_stopped = Signal()

    def __init__(
        self,
        shading_model: ShadingModel,
        training_dataloader,
        render,
        train_params,
        num_epochs,
        lr,
        chroma_finetune: bool = False,
        consistency_weight: float = 0.0,
        chroma_reg_weight: float = 0.0,
        chroma_clamp_enabled: bool = False,
        chroma_clamp_value: float = 0.15,
        radial_decay_weight: float = 0.0,
        radial_decay_threshold: float = 1.2,
        monotonic_weight: float = 0.0,
    ):
        super().__init__()
        self.training_config = TrainingConfig()
        self.GUI_config = GUIConfig()
        self.shading_model = shading_model
        self.training_dataloader = training_dataloader
        self.render = render
        self._is_running = True
        self.train_params = train_params
        self.num_epochs = num_epochs
        self.lr = lr
        self.chroma_finetune = chroma_finetune
        self.consistency_weight = consistency_weight
        self.chroma_reg_weight = chroma_reg_weight
        self.chroma_clamp_enabled = chroma_clamp_enabled
        self.chroma_clamp_value = chroma_clamp_value
        self.radial_decay_weight = radial_decay_weight
        self.radial_decay_threshold = radial_decay_threshold
        self.monotonic_weight = monotonic_weight

    def run(self):
        logging.basicConfig(filename='train.log', level=logging.INFO, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        for i in range(self.GUI_config.training_update_times):
            logging.info("training times: " + str(i))
            if not self._is_running:
                self.training_stopped.emit()
                return
            self.start_training(i)
            logging.info("current shading model: " + self.shading_model.light.name)
            logging.info("shading model light parameters after training: ")
            logging.info("albedo: " + str(self.shading_model.albedo))
            logging.info("ambient: " + str(self.shading_model.ambient_light))
            logging.info("gamma: " + str(self.shading_model.light.gamma))
            logging.info("tau: " + str(self.shading_model.light.tau))
            if hasattr(self.shading_model.light, 'sigma'):
                logging.info("sigma: " + str(self.shading_model.light.sigma))
            logging.info("_t_vec: " + str(self.shading_model.light._t_vec))
            logging.info("_r_l2c_SO3: " + str(self.shading_model.light._r_l2c_SO3.log()))
            imgs_raw, imgs_rendered = self.render()
            if self._is_running:
                self.update_images.emit(imgs_raw, imgs_rendered)
                if hasattr(self.shading_model.light, 'sigma'):
                    if self.shading_model.light.sigma.ndim == 0:
                        self.update_shading_model_param.emit(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.light.tau, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [self.shading_model.light.sigma, 0])
                    else:
                        self.update_shading_model_param.emit(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.light.tau, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [self.shading_model.light.sigma[0], self.shading_model.light.sigma[1]])
                else:
                    self.update_shading_model_param.emit(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.light.tau, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [0, 0])
        self.training_stopped.emit()
                    
    def get_learning_param(self, checked_param):
        match checked_param:
            case "Albedo":
                return [{'params': [self.shading_model.albedo_log], 'lr': self.lr['Albedo'], "name": "albedo"}]
            case "gamma":
                return [{'params': [self.shading_model.light.gamma_log], 'lr': self.lr['gamma'], "name": "gamma"}]
            case "tau":
                return [{'params': [self.shading_model.light.tau_log], 'lr': self.lr['tau'], "name": "tau"}]
            case "Ambient":
                return [{'params': [self.shading_model.ambient_light_log], 'lr': self.lr['Ambient'], "name": "ambient"}]
            case "Rotation":
                return [{'params': [self.shading_model.light._r_l2c_SO3], 'lr': self.lr['Rotation'], "name": "r_vec"}]
            case "Translation":
                return [{'params': [self.shading_model.light._t_vec], 'lr': self.lr['Translation'], "name": "t_vec"}]
            case "\u03C3_x":
                return [{'params': [self.shading_model.light.sigma], 'lr': self.lr['\u03C3_x'], 'name': 'sigma'}]
            case "Trunk":
                if hasattr(self.shading_model.light, 'trunk') and hasattr(self.shading_model.light, 'intensity_head'):
                    return [
                        {'params': self.shading_model.light.trunk.parameters(), 'lr': self.lr['Trunk'], 'name': 'mlp_trunk'},
                        {'params': self.shading_model.light.intensity_head.parameters(), 'lr': self.lr['Trunk'], 'name': 'mlp_intensity'},
                    ]
                if hasattr(self.shading_model.light, 'mlp'):
                    return [{'params': self.shading_model.light.mlp.parameters(), 'lr': self.lr['Trunk'], 'name': 'mlp0'}]
                return []
            case "Light Color":
                return [{'params': [self.shading_model.light.light_color_log], 'lr': self.lr['Light Color'], 'name': 'light_color'}]
            case "Chroma Head":
                if hasattr(self.shading_model.light, 'chroma_head'):
                    return [{'params': self.shading_model.light.chroma_head.parameters(), 'lr': self.lr['Chroma Head'], 'name': 'chroma_head'}]
                return []
            case _:
                return []


    def start_training(self, round: int):       
        num_epoch = (int)(self.num_epochs / self.GUI_config.training_update_times)

        loss_fn = nn.L1Loss()
        l = []
        if hasattr(self.shading_model.light, "set_chroma_clamp"):
            self.shading_model.light.set_chroma_clamp(self.chroma_clamp_enabled, self.chroma_clamp_value)
        for train_param in self.train_params:
            l += self.get_learning_param(train_param)
        if (
            "Chroma Head" in self.train_params
            and hasattr(self.shading_model.light, "chroma_head")
            and not any(p is self.shading_model.light.chroma_head for g in l for p in g.get("params", []))
        ):
            l.append(
                {
                    "params": self.shading_model.light.chroma_head.parameters(),
                    "lr": self.lr["Chroma Head"],
                    "name": "chroma_head",
                }
            )
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.shading_model = self.shading_model.to(self.training_config.device)
        for epoch in tqdm(range(num_epoch)):
            if not self._is_running:
                self.training_stopped.emit()
                return
            self.update_progress_bar.emit(round * (int)(self.num_epochs / self.GUI_config.training_update_times) + epoch)
            for itr, (pts, intensities, rvec_w2c, tvec_w2c, _, _) in enumerate(self.training_dataloader):
                rendered_intensities = self.shading_model(pts, rvec_w2c, tvec_w2c)
                loss = loss_fn(rendered_intensities, intensities)
                mono_pred = None
                if self.consistency_weight > 0.0 and hasattr(self.shading_model, "forward_mono"):
                    mono_pred = self.shading_model.forward_mono(pts, rvec_w2c, tvec_w2c)
                    gray_pred = rendered_intensities
                    if rendered_intensities.ndim == 3:
                        gray_pred = rendered_intensities.mean(dim=-1)
                    loss = loss + self.consistency_weight * loss_fn(gray_pred, mono_pred)
                if self.chroma_reg_weight > 0.0 and hasattr(self.shading_model.light, "last_delta"):
                    if self.shading_model.light.last_delta is not None:
                        loss = loss + self.chroma_reg_weight * torch.mean(self.shading_model.light.last_delta**2)
                mono_pred = mono_pred if mono_pred is not None else rendered_intensities
                if (self.radial_decay_weight > 0.0 or self.monotonic_weight > 0.0) and mono_pred is not None:
                    R_w2c = SO3.exp(rvec_w2c)
                    pts_in_cam = R_w2c.act(pts) + tvec_w2c
                    radius = torch.sqrt(pts_in_cam[..., 0] ** 2 + pts_in_cam[..., 1] ** 2) / torch.clamp_min(
                        pts_in_cam[..., 2].abs(), 1e-8
                    )
                    if self.radial_decay_weight > 0.0:
                        far_mask = radius > self.radial_decay_threshold
                        if far_mask.any():
                            mono_for_decay = mono_pred
                            if mono_for_decay.ndim == 3:
                                mono_for_decay = mono_for_decay.mean(dim=-1)
                            decay_loss = torch.mean(torch.abs(mono_for_decay[far_mask]))
                            loss = loss + self.radial_decay_weight * decay_loss
                    if self.monotonic_weight > 0.0:
                        mono_for_mono = mono_pred
                        if mono_for_mono.ndim == 3:
                            mono_for_mono = mono_for_mono.mean(dim=-1)
                        radius_flat = radius.reshape(radius.shape[0], -1)
                        mono_flat = mono_for_mono.reshape(mono_for_mono.shape[0], -1)
                        _, sort_idx = torch.sort(radius_flat, dim=-1)
                        mono_sorted = torch.gather(mono_flat, dim=-1, index=sort_idx)
                        if mono_sorted.shape[-1] > 1:
                            diff = mono_sorted[..., :-1] - mono_sorted[..., 1:]
                            monotonic_loss = torch.relu(diff).mean()
                            loss = loss + self.monotonic_weight * monotonic_loss
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def stop(self):
        self._is_running = False
