import torch
import torch.nn.functional as F

from .model import Encoder, Decoder
from .distributions import DiagonalGaussianDistribution


class AutoencoderKL(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 learn_logvar=False,
                 scale_factor=1.0
                 ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.scale_factor = scale_factor
        
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        
        self.z_dim = ddconfig["z_channels"]
        self.downsample_factor = 2**(self.decoder.num_resolutions-1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        target_dict = {}
        for k in keys:
            if k.startswith("first_stage_model"):
                target_dict[k.split('first_stage_model.')[-1]] = sd[k] 
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(target_dict, strict=True)
        print(f"Restored from {path}")

    @torch.no_grad()
    def encode(self, x, sample_posterior=True):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z * self.scale_factor

    @torch.no_grad()
    def encode_dist(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return moments
        
    @torch.no_grad()
    def decode(self, z):
        z = z / self.scale_factor
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    @torch.no_grad()
    def forward(self, input, sample_posterior=True):
        z = self.encode(input, sample_posterior)
        dec = self.decode(z)
        return dec

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False,  **kwargs):
        log = dict()
        x_T = kwargs['xT']
        x = batch[self.image_key]
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
            
        log["inputs"] = x
        log["condition"] = x_T
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

