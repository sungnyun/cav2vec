# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,sys
import logging
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import utils, checkpoint_utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    # TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm
from copy import deepcopy
from .ema_module import EMAModule, EMAModuleConfig

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from resnet import ResEncoder
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    from utils import compute_mask_indices
    from encoder import TransformerEncoder
    from decoder import TransformerDecoder

else:
    from .hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from .resnet import ResEncoder
    from .utils import compute_mask_indices
    from .encoder import TransformerEncoder
    from .decoder import TransformerDecoder

from omegaconf import II

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)


@dataclass
class CAV2vecConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")
    input_modality: str = II("task.input_modality")
    w2v_path: str = field(  # TDOO: change this to w2v_path_pre to distinguish from the w2v_path for fine-tuning
        default="", metadata={"help": "path to hubert pretrained model"}
    )
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={
            "help": "dropout to apply to the features (after feat extr)"
        },
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        },
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    use_mlm_loss: bool = field(
        default=True,  # later, change this to False for pure AV2vec
        metadata={"help": "use masked language modeling (MLM)-style loss of HuBERT for auxiliary and multi-task learning"}
    )
    use_distill_loss: bool = field(
        default=True,
        metadata={"help": "use data2vec-style self-distillation (regression) loss"}
    )
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    resnet_relu_type: str = field(default='prelu', metadata={"help": 'relu type for resnet'})
    resnet_weights: Optional[str] = field(default=None, metadata={"help": 'resnet weights'})
    sim_type: str = field(default='cosine', metadata={"help": 'similarity type'})

    sub_encoder_layers: int = field(default=0, metadata={'help': 'number of transformer layers for single modality'})
    audio_feat_dim: int = field(default=-1, metadata={'help': 'audio feature dimension'})
    modality_dropout: float = field(default=0, metadata={'help': 'drop one modality'})
    audio_dropout: float = field(default=0, metadata={'help': 'drop audio feature'})
    modality_fuse: str = field(default='concat', metadata={'help': 'fusing two modalities: add,concat'})
    selection_type : str = field(default='same_other_seq', metadata={'help': 'type of selectig images, same_other_seq: replace masked span with span from another sequence, same_seq: repace masked span with span of the same sequence'})
    masking_type : str = field(default='input', metadata={'help': 'input or feature masking'})

    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})
    rel_score: bool = field(
            default=False, 
            metadata={'help': 'using relscore'},
        )
    
    # ema
    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )
    occ_strategy: int = field(
        default=1,
        metadata={
            "help": "mask indices and occlusion indices configuration"
        },
    )
    uocc_strategy: int = field(
        default=1,
        metadata={
            "help": "mask indices and occlusion indices configuration"
        },
    )
    add_clean_loss: bool = field(
        default=False,
        metadata={
            "help": "include clean frames in the distillation and mlm losses"
        }
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = True ###########
    instance_norm_targets: bool = False ###########
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining

class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None, cfg=None, rel_score=False):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, cfg.encoder_embed_dim)
        self.encoder = TransformerEncoder(cfg) if cfg.encoder_layers > 0 else None
        self.rel_score = rel_score
        if self.rel_score:
            raise NotImplementedError

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))

        if self.encoder is not None:
            x = self.encoder(x)[0].transpose(1, 2)
        else:
            x = x.transpose(1, 2)
            
        # x: [B, F, T]
        if self.rel_score:  # change
            x = x * self.score(x) + x 
        else:
            pass
        
        return x

@register_model("cav2vec", dataclass=CAV2vecConfig)
class CAV2vecModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: CAV2vecConfig,
        task_cfg: AVHubertPretrainingConfig,
        dictionaries: List[Dictionary],
        **kwargs
    ) -> None:
        super().__init__()
        logger.info(f"AV2VecModel Config: {cfg}")

        feature_ds_rate = 1
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate
        self.rel_score = kwargs.get('rel_score', False)
        sub_cfg = deepcopy(cfg)
        sub_cfg.encoder_layers = sub_cfg.sub_encoder_layers
        resnet = ResEncoder(relu_type=cfg.resnet_relu_type, weights=cfg.resnet_weights)
        self.feature_extractor_audio = SubModel(resnet=None, input_dim=cfg.audio_feat_dim, cfg=sub_cfg, rel_score=self.rel_score)
        self.feature_extractor_video = SubModel(resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg, rel_score=self.rel_score)
        self.modality_dropout, self.audio_dropout = cfg.modality_dropout, cfg.audio_dropout
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == 'concat':
            self.embed = cfg.encoder_embed_dim * 2
        elif self.modality_fuse == 'add':
            self.embed = cfg.encoder_embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.ema = None
        self.cfg = cfg
        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale
        self.final_pred_proj = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim) 
        self.final_pred_proj_asp = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)
        self.final_pred_proj_vsp = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)
        self.final_pred_proj_uasp = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)
        self.final_pred_proj_uvsp = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)

        self.use_mlm_loss = cfg.use_mlm_loss
        self.mask_prob_image, self.mask_prob_audio = cfg.mask_prob_image, cfg.mask_prob_audio
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length_image, self.mask_length_audio = cfg.mask_length_image, cfg.mask_length_audio
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.sim_type = cfg.sim_type
        self.selection_type = cfg.selection_type
        self.masking_type = cfg.masking_type

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.audio_feat_dim).uniform_() if self.masking_type == 'input' else torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            ) if self.use_mlm_loss else None
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim) if self.use_mlm_loss else None

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info(
                "cannot find dictionary. assume will be used for fine-tuning"
            )
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

        if cfg.w2v_path:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path
            )
            new_state = dict()
            for k, v in state["model"].items():
                if k == "label_embs_concat":  # km labels do not match
                    continue
                new_state[k] = v
            self.load_state_dict(new_state, strict=False)
            logger.info(f">> checkpoint loaded from {cfg.w2v_path}")

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,  # TODO: check if fp32 is okay
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")

        self.ema = EMAModule(
            self.encoder if self.cfg.ema_transformer_only else self,
            ema_config,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.final_pred_proj is not None:
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)

        self.num_updates = num_updates

    @classmethod
    def build_model(cls, cfg: CAV2vecConfig, task: AVHubertPretrainingTask):
        """Build a new model instance."""
        rel_score = False
        kwargs = {
            "rel_score": rel_score
        }
        model = CAV2vecModel(cfg, task.cfg, task.dictionaries, **kwargs)
        return model

    def apply_input_mask(self, x, padding_mask, target_list, skip_indices=None):
        B, C, T = x.shape[:3]
        is_audio = True if len(x.shape) == 3 else False
        if is_audio:
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        else:
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
        if mask_prob > 0:

            mask_indices, starts, ends, batch_indexes = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices_np = mask_indices
            mask_indices = torch.from_numpy(mask_indices).to(x.device)

            if skip_indices is not None:
                new_batch_indexes, new_starts, new_ends = [], [], []
                for batch_index, start, end in zip(batch_indexes, starts, ends):
                    if skip_indices[batch_index, start] and skip_indices[batch_index, end-1]: # avoid masks on occluded parts
                        mask_indices[batch_index, start:end] = False
                        continue
                    new_batch_indexes.append(batch_index)
                    new_starts.append(start)
                    new_ends.append(end)
                batch_indexes = np.asarray(new_batch_indexes)
                starts = np.asarray(new_starts)
                ends = np.asarray(new_ends)

            x = x.transpose(1, 2).contiguous() # [B, T, C, H, W]
            if B == 1:
                x[mask_indices] = 0
            elif is_audio:
                x[mask_indices] = self.mask_emb
            elif self.selection_type == 'same_other_seq':
                perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
                x_perm = x[perm]
                x[mask_indices] = x_perm[mask_indices]
            elif self.selection_type == 'same_seq':  # masking by substitution (for video frames)
                batch_indexes_, other_indexes = [], []
                if len(batch_indexes) > 0:
                    for batch_index, start, end in zip(batch_indexes, starts, ends):
                        length = end-start
                        other_start = np.setdiff1d(np.arange(T), np.arange(max(0, start-length), end))
                        if len(other_start) > 0:
                            other_start = np.random.choice(other_start, size=1)
                        else:
                            other_start = 0
                        other_end = other_start + length
                        other_indexes.append(np.arange(other_start, other_end).clip(max=T-1))
                        batch_indexes_.append(np.zeros([length], dtype=np.int64)+batch_index)
                    batch_indexes, other_indexes = np.concatenate(batch_indexes_), np.concatenate(other_indexes)
                    x[mask_indices] = x[batch_indexes, other_indexes]

            x = x.transpose(1, 2).contiguous()
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            logger.info(f"No mask channel prob for input masking")
        return x, mask_indices

    def apply_feature_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        assert self.mask_prob_audio == self.mask_prob_image and self.mask_length_audio == self.mask_length_image, f"masking prob/length for image/audio be same for feature masking"
        mask_prob, mask_length = self.mask_prob_audio, self.mask_length_image
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices, _, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_targets(
            self, features: torch.Tensor, mask_indices: torch.Tensor, target_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        trim_feat_tsz = None
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            if mask_indices is not None:
                mask_indices = mask_indices[..., :feat_tsz]
            trim_feat_tsz = feat_tsz
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, mask_indices, target_list, trim_feat_tsz

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def compute_logits(self, feats, emb_mat):
        # feats: [B, T, F], emb_mat: [V, F]
        if self.sim_type == 'dot':
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.sim_type == 'cosine':
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(dim=-1) # [B*T, V]
            denom = (feats_**2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (emb_mat**2).sum(dim=-1).sqrt().unsqueeze(dim=0) # [B*T, V]
            logits = (nom/denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.logit_temp
        return logits
    
    def compute_targets(self, layer_results):
        target_layer_results = [l[2] for l in layer_results]
        permuted = False
        if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
            target_layer_results = [
                tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
            ]
            permuted = True

        if self.cfg.batch_norm_target_layer:
            target_layer_results = [
                F.batch_norm(
                    tl.float(), running_mean=None, running_var=None, training=True
                )
                for tl in target_layer_results
            ]

        if self.cfg.instance_norm_target_layer:
            target_layer_results = [
                F.instance_norm(tl.float()) for tl in target_layer_results
            ]

        if permuted:
            target_layer_results = [
                tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
            ]

        if self.cfg.group_norm_target_layer:
            target_layer_results = [
                F.layer_norm(tl.float(), tl.shape[-2:])
                for tl in target_layer_results
            ]

        if self.cfg.layer_norm_target_layer:
            target_layer_results = [
                F.layer_norm(tl.float(), tl.shape[-1:])
                for tl in target_layer_results
            ]

        y = sum(target_layer_results) / len(target_layer_results)

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y.float(), y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.float().permute(1, 2, 0)).transpose(1, 2)  # TBC -> BCT -> BTC
            permuted = True

        if not permuted:
            y = y.transpose(0, 1)

        return y

    @torch.no_grad()
    def forward_ema_teacher(self, clean_audio, clean_video, padding_mask, trim_feat_tsz=None):
        features_clean_audio = self.forward_features(clean_audio, modality='audio')
        features_clean_video = self.forward_features(clean_video, modality='video')
        if self.modality_fuse == 'concat':
            features_clean = torch.cat([
                torch.cat([features_clean_audio, features_clean_video], dim=1),
                torch.cat([features_clean_audio, features_clean_video * 0], dim=1),
                torch.cat([features_clean_audio * 0, features_clean_video], dim=1),
            ], dim=0)     # [3B, 2F, T]
            padding_mask = padding_mask.repeat(3, 1)
        elif self.modality_fuse == 'add':
            features_clean = torch.cat([
                features_clean_audio + features_clean_video,
                features_clean_audio,
                features_clean_video,
            ], dim=0)
        if trim_feat_tsz is not None:
            features_clean = features_clean[..., :trim_feat_tsz]

        self.ema.model.eval()
        features_clean = features_clean.transpose(1, 2)
        features_clean = self.layer_norm(features_clean)
        if self.post_extract_proj is not None:
            features_clean = self.post_extract_proj(features_clean)

        # extract clean features
        y = features_clean
        y, layer_results = self.ema.model.extract_features(
            y,
            padding_mask=padding_mask,
            min_layer=self.cfg.encoder_layers - self.average_top_k_layers,
        )
        # normalize & average
        y = self.compute_targets(layer_results)
        return y

    @torch.no_grad()
    def forward_ema_teacher_with_drop(self, clean_audio, clean_video, padding_mask, trim_feat_tsz=None, modality='avsp'):
        features_clean_audio = self.forward_features(clean_audio, modality='audio')
        features_clean_video = self.forward_features(clean_video, modality='video')
        if self.modality_fuse == 'concat':
            if modality == 'avsp':
                features_clean = torch.cat([features_clean_audio, features_clean_video], dim=1)
                padding_mask = padding_mask
            elif modality == 'vsp':
                features_clean = torch.cat([
                    torch.cat([features_clean_audio, features_clean_video], dim=1),
                    torch.cat([features_clean_audio * 0, features_clean_video], dim=1),
                ], dim=0)     # [2B, 2F, T]
                padding_mask = padding_mask.repeat(2, 1)
            elif modality == 'asp':
                features_clean = torch.cat([
                    torch.cat([features_clean_audio, features_clean_video], dim=1),
                    torch.cat([features_clean_audio, features_clean_video * 0], dim=1),
                ], dim=0)     # [2B, 2F, T]
                padding_mask = padding_mask.repeat(2, 1)
        elif self.modality_fuse == 'add':
            raise NotImplementedError
        
        if trim_feat_tsz is not None:
            features_clean = features_clean[..., :trim_feat_tsz]

        self.ema.model.eval()
        features_clean = features_clean.transpose(1, 2)
        features_clean = self.layer_norm(features_clean)
        if self.post_extract_proj is not None:
            features_clean = self.post_extract_proj(features_clean)

        # extract clean features
        y = features_clean
        y, layer_results = self.ema.model.extract_features(
            y,
            padding_mask=padding_mask,
            min_layer=self.cfg.encoder_layers - self.average_top_k_layers,
        )
        # normalize & average
        y = self.compute_targets(layer_results)
        return y

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_video = source['audio'], source['video']
        src_clean_audio, src_clean_video = source['clean_audio'], source['clean_video']
        video_occ_indices, audio_occ_indices = source['video_occ_indices'], source['audio_occ_indices']

        if mask and self.masking_type == 'input':
            av_occ_indices = torch.logical_or(video_occ_indices, audio_occ_indices)
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list, skip_indices=av_occ_indices)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list, skip_indices=av_occ_indices)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)

            # uni-modal targets: occ - masked parts (occs can overlap)
            asp_indices = torch.logical_and(video_occ_indices, ~mask_indices)
            vsp_indices = torch.logical_and(audio_occ_indices, ~mask_indices)
            uasp_indices = torch.logical_and(video_occ_indices, ~mask_indices)
            uvsp_indices = torch.logical_and(audio_occ_indices, ~mask_indices)
        else:
            # src_audio, src_video, mask_indices = src_audio, src_video, None
            raise NotImplementedError

        features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
        features_video = self.forward_features(src_video, modality='video')

        MODALITY_DROP_FLAG = False
        AUDIO_DROP_FLAG, VIDEO_DROP_FLAG = False, False
        modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
        if self.training:
            if modality_drop_prob < self.modality_dropout:
                MODALITY_DROP_FLAG = True
                if audio_drop_prob < self.audio_dropout:
                    features_audio = 0 * features_audio
                    AUDIO_DROP_FLAG = True
                else:
                    features_video = 0 * features_video
                    VIDEO_DROP_FLAG = True
        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        if target_list is not None:
            # ignore the mask_indices and trim_feat_tsz -> they are not trimmed
            features, mask_indices, target_list, trim_feat_tsz = self.forward_targets(features, mask_indices, target_list)
        else:
            trim_feat_tsz = None

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        if self.masking_type == 'feature' and mask:
            x, mask_indices = self.apply_feature_mask(features, padding_mask, target_list)
        else:
            x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        pred_targets = {}
        if not MODALITY_DROP_FLAG:
            y = self.forward_ema_teacher_with_drop(
                src_clean_audio, src_clean_video, 
                padding_mask=padding_mask,
                trim_feat_tsz=trim_feat_tsz,
                modality='avsp'
            )
            y_avsp = y
            pred_targets['avsp'] = (self.final_pred_proj(x[mask_indices]), y_avsp[mask_indices])

        elif AUDIO_DROP_FLAG:
            y = self.forward_ema_teacher_with_drop(
                src_clean_audio, src_clean_video, 
                padding_mask=padding_mask,
                trim_feat_tsz=trim_feat_tsz,
                modality='asp'
            )
            y_avsp, y_asp = y.chunk(2)
            pred_targets['avsp'] = (self.final_pred_proj(x[mask_indices_video]), y_avsp[mask_indices_video])
            if uasp_indices is not None:
                pred_targets['uasp'] = (self.final_pred_proj_uasp(x[uasp_indices]), y_asp[uasp_indices])

        elif VIDEO_DROP_FLAG:
            y = self.forward_ema_teacher_with_drop(
                src_clean_audio, src_clean_video, 
                padding_mask=padding_mask,
                trim_feat_tsz=trim_feat_tsz,
                modality='vsp'
            )
            y_avsp, y_vsp = y.chunk(2)
            pred_targets['avsp'] = (self.final_pred_proj(x[mask_indices_audio]), y_avsp[mask_indices_audio])
            if uvsp_indices is not None:
                pred_targets['uvsp'] = (self.final_pred_proj_uvsp(x[uvsp_indices]), y_vsp[uvsp_indices])

        sz = self.encoder_embed_dim
        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        loss = dict()
        sample_size = 0
        for key, (x_m, y_m) in pred_targets.items():
            if self.loss_beta == 0:
                loss_ = F.mse_loss(x_m.float(), y_m.float(), reduction="none").sum(dim=-1)
                sample_size += loss_.numel()
                loss['loss_' + key] = loss_.sum() * scale
            else:
                loss_ = F.smooth_l1_loss(
                    x_m.float(), y_m.float(), reduction="none", beta=self.loss_beta
                ).sum(dim=-1)
                sample_size += loss_.numel()
                loss['loss_' + key] = loss_.sum() * scale

        if self.ema is not None:
            ema_decay = self.ema.get_decay() * 1000
        else:
            ema_decay = 0

        if self.use_mlm_loss:
            label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
            proj_x = self.final_proj(x)
            if self.untie_final_proj:
                proj_x_list = proj_x.chunk(len(self.num_classes), dim=-1)
            else:
                proj_x_list = [proj_x for _ in self.num_classes]
            logit_list = [self.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in zip(proj_x_list, label_embs_list, self.num_classes)] # [[B*T, V]]
            mask, unmask = torch.logical_and(mask_indices, ~padding_mask).view(-1), torch.logical_and(~mask_indices, ~padding_mask).view(-1) # [B*T]
            logit_m_list, logit_u_list = [logit[mask] for logit in logit_list], [logit[unmask] for logit in logit_list]
            target_m_list, target_u_list = [target.view(-1)[mask].long() for target in target_list], [target.view(-1)[unmask].long() for target in target_list]
        else:
            logit_m_list, logit_u_list, target_m_list, target_u_list = [], [], [], []

        result = {
                    "regression": loss,
                    "sample_size": sample_size,
                    "ema_decay": ema_decay,
                    "logit_m_list": logit_m_list,
                    "logit_u_list": logit_u_list,
                    "target_m_list": target_m_list,
                    "target_u_list": target_u_list,
                    "padding_mask": padding_mask,
                    "features_pen": features_pen,
                }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def extract_finetune(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None):
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list=None)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list=None)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video) # mask_indices not used in fine-tuning
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
            features_video = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1))
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = features_video.new_zeros(features_video.size(0), self.encoder_embed_dim, features_video.size(-1))
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video') # edit
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T] # edit

        # feature fusion after audio, visual front-ends
        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        # feature post-projection after fusion
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        x = features
        mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        return x, padding_mask

    def extract_finetune_features(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None, output_clean=False):
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list=None)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list=None)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video) # mask_indices not used in fine-tuning
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
            features_video = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1))
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = features_video.new_zeros(features_video.size(0), self.encoder_embed_dim, features_video.size(-1))
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]

        if output_clean:
            assert 'clean_audio' in source
            src_clean_audio = source['clean_audio']

            if src_clean_audio is not None:
                if mask and self.masking_type == 'input':
                    src_clean_audio, mask_indices_clean_audio = self.apply_input_mask(src_clean_audio, padding_mask, target_list=None)
                    # mask_indices = torch.logical_or(mask_indices_clean_audio, mask_indices_video) # mask_indices not used in fine-tuning
                else:
                    src_clean_audio = src_clean_audio
                features_clean_audio = self.forward_features(src_clean_audio, modality='audio') # features: [B, F, T]
            else:
                features_clean_audio = None
        else:
            features_clean_audio = None

        return {'video': features_video, 'audio': features_audio, 'clean_audio': features_clean_audio}
    
    def extract_finetune_encoder(self, features, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None):
        # feature fusion after audio, visual front-ends
        features_audio, features_video = features['audio'], features['video']
        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        # feature post-projection after fusion
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        x = features
        mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        return x, padding_mask

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []
        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
        self.final_pred_proj = None
        self.final_pred_proj_asp = None
        self.final_pred_proj_vsp = None
        self.final_pred_proj_uasp = None
        self.final_pred_proj_uvsp = None
        self.final_pred_proj_uaasp = None
        self.ema = None

    def get_logits(self, net_output, is_masked=True):
        raise NotImplementedError

    def get_targets(self, net_output, is_masked=True):
        raise NotImplementedError

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits