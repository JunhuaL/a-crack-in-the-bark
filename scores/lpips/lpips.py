import torch
import torch.nn as nn
import warnings

from collections import namedtuple
from torchvision import models as tv
from torch.utils.data import DataLoader
from datasets.img_dataset import ImgLoader, PairedDataset

### Taken LPIPS score model and implementation taken from https://github.com/richzhang/PerceptualSimilarity
### 
def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)

def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)

class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple(
            "AlexnetOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"]
        )
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out

class LPIPS(nn.Module):
    def __init__(
        self,
        pretrained=True,
        net="alex",
        version="0.1",
        lpips=True,
        spatial=False,
        pnet_rand=False,
        pnet_tune=False,
        use_dropout=True,
        model_path=None,
        eval_mode=True,
        verbose=True,
    ):
        """Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(LPIPS, self).__init__()
        warnings.filterwarnings("ignore")
        if verbose:
            pass

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type == "alex":
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self.pnet_type == "squeeze":  # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
            self.lins += [self.lin5, self.lin6]
        self.lins = nn.ModuleList(self.lins)

        if pretrained:
            if model_path is None:
                import os

                model_path = os.path.abspath(
                    os.path.join(
                        "weights", 'v%s' % (version), '%s.pth' % (net)
                    )
                )

            self.load_state_dict(
                torch.load(model_path, map_location="cpu"), strict=False
            )

        if eval_mode:
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if (
            normalize
        ):  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1))
            if self.version == "0.1"
            else (in0, in1)
        )
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
            for kk in range(self.L)
        ]

        val = 0
        for l in range(self.L):
            val += res[l]

        if retPerLayer:
            return (val, res)
        else:
            return val

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_model(mode, device):
    perceptual_model = LPIPS(net=mode).to(device)
    return perceptual_model

def compute_lpips(img1, img2, model, device):
    img1 = (img1/255).to(device)
    img2 = (img2/255).to(device)
    score = model(img1, img2, normalize=True).cpu().item()
    return score

def compute_lpips_given_paths(clean_path, attacked_path, model, batch_size, device):
    clean_imgs = ImgLoader(clean_path)
    attacked_imgs = ImgLoader(attacked_path)
    img_pairs = PairedDataset(clean_imgs,attacked_imgs)

    scores = []
    for attacked_imgs,clean_imgs,_ in img_pairs:
        attacked_imgs = (attacked_imgs/255).to(device)
        clean_imgs = (clean_imgs/255).to(device)

        score = model(clean_imgs,attacked_imgs, normalize=True).cpu().item()
        scores.append(score)
    return scores
    
    
