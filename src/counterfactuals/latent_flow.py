import torch
import pytorch_lightning as pl
from modules import LatentDecoder
from diffusion_model import FlowMatcher
from cabrnet.archs.generic.model import CaBRNet



def get_classifier():
    model_arch = "src/counterfactuals/intermediate_model_arch.yml"
    model_state_dict = "trained_models/protopnet_cub200_resnet50_s1337/final/model_state.pth"
    
    classifier = CaBRNet.build_from_config(config=model_arch, state_dict_path=model_state_dict)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


class LatentFlow(pl.LightningModule):
    def __init__(
        self,
        decoder: LatentDecoder,
        flow_matcher: FlowMatcher,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.decoder = decoder
        self.flow_matcher = flow_matcher
        self.lr = lr

        self.classifier = get_classifier()

        self.save_hyperparameters(ignore=['decoder', 'flow_matcher'])

    def on_save_checkpoint(self, checkpoint):
        """ Remove the classifier from the checkpoint to save space. """
        state_dict = checkpoint["state_dict"]
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("classifier.")]
        for k in keys_to_remove:
            del state_dict[k]

    def step(self, batch, *, return_upsampled=False):
        img, _ = batch

        with torch.no_grad():
            features = self.classifier.extractor(img)['convnet']

        upsampled_features = self.decoder(features)
        flow_loss = self.flow_matcher.loss(img, upsampled_features)

        if return_upsampled:
            return flow_loss, upsampled_features
        return flow_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr
        )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("Train/Loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:  
            imgs, _ = batch
            loss, upsampled_features = self.step(batch, return_upsampled=True)
            self.log_images(imgs, upsampled_features)
        else:
            loss = self.step(batch)
        self.log("Val/Loss", loss, prog_bar=True)
        return loss
    
    def log_images(self, imgs, upsampled_features):
        n_show = min(5, imgs.size(0))
        epoch = self.current_epoch
        tensorboard = self.logger.experiment
        for j in range(n_show):
            real_img = imgs[j].unsqueeze(0)
            upsampled = upsampled_features[j].unsqueeze(0)
            flow_img = self.flow_matcher.sample_image(upsampled)
            # Real image
            if epoch == 0:
                tensorboard.add_image(
                    f'Real_Img/Sample_{j}',
                    real_img.cpu(), 
                    epoch,
                    dataformats='CHW'
                )
            # Reconstructed image
            tensorboard.add_image(
                f'Recon_Img/Sample_{j}',
                flow_img.cpu(),
                epoch,
                dataformats='CHW'
            )


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    from pytorch_lightning.cli import LightningCLI
    LightningCLI(
        LatentFlow,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


    
