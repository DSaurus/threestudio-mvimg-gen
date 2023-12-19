from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("mvimg-gen-stable-zero123-system")
class Zero123Simple(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        pass

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self.guidance(**batch["random_camera"])
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["imgs_final"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
