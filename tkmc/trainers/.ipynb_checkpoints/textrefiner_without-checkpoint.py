import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES
from tqdm import tqdm
import math
from dassl.utils import save_checkpoint
from clip.model import VisionTransformer, convert_weights
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
_tokenizer = _Tokenizer()

class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )
    def forward(self, input_feat): # input_feat:[B, d] [B, N, d]
        
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))
        
        return final_feat.squeeze(-1).squeeze(-1)
        
def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.PROMPTKD.TEACHER_NAME
    
    if backbone_name == "ViT-B/16":
        model_path = './clip/ViT-B-16.pt'
    elif backbone_name == "ViT-L/14":
        model_path = './clip/ViT-L-14.pt'
    elif backbone_name == "ViT-B/32":
        model_path = './clip/ViT-B-32.pt'
    else:
        print('enter the wrong teacher name.')
    
    print(f"CLIP Teacher name is {backbone_name}")
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # We default use PromptSRC to pretrain our teacher model
    design_details = {"trainer": 'IVLP',
                        "vision_depth": 9,
                        "language_depth": 9,
                        "vision_ctx": 4,
                        "language_ctx": 4}
    
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    model_path = './clip/ViT-B-16.pt'
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.PROMPTKD.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.PROMPTKD.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, is_teacher):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTKD.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTKD.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.trainer_name = cfg.TRAINER.NAME
        self.train_modal = cfg.TRAINER.MODAL
        
        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTKD.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        
        print(f'classnames size is {len(classnames)}')

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens

        if self.train_modal == "base2novel":
            self.register_buffer("token_prefix", embedding[:math.ceil(self.n_cls / 2), :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:math.ceil(self.n_cls / 2), 1 + n_ctx:, :])  # CLS, EOS

            self.register_buffer("token_prefix2", embedding[math.ceil(self.n_cls / 2):, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[math.ceil(self.n_cls / 2):, 1 + n_ctx:, :])  # CLS, EOS
            
        elif self.train_modal == "cross":
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            
            self.register_buffer("token_prefix2", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # print(f'label is {label}')
        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.trainer_name == "PromptKD" and self.train_modal == "base2novel":
            # print(f'n_cls is {self.n_cls}')
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts
class MultiHeadAttention(nn.Module):    #第二版新增注意力机制
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final linear layer
        output = self.out(output)

        return output, attention_weights

class Memory(nn.Module):
    def __init__(self, clip_model,feature_dim = 512, memory_size = 25, reduction = 4,num_heads=8):
        super().__init__()
        self.device = clip_model.dtype
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.text_fine_cache = F.normalize(torch.rand(self.memory_size, feature_dim), dim = -1)
        self.text_fine_cache = self.text_fine_cache.to(self.device)
        self.text_fine_cache = self.text_fine_cache.cuda()
        self.alpha = 0.2
        self.momentum = 0.8
        
        self.extractor = nn.Linear(2 * feature_dim , feature_dim, bias=False)
        self.extractor = self.extractor.to(self.device)
        self.extractor = self.extractor.cuda()

        self.writeTF = lambda x: x.clone()
        
        self.mutitransformer=MultiHeadAttention(feature_dim,num_heads)
        self.mutitransformer=self.mutitransformer.to(self.device)
        self.mutitransformer=self.mutitransformer.cuda()

    def forward(self,text_token=None, image_token=None,image_features=None):    #看一下这边个是什么东东
        fine_feature = self.read(text_token)
        
        text_fine_feature = torch.cat((text_token, fine_feature), dim = -1)
        text_fine_feature = self.alpha * self.extractor(text_fine_feature) + text_token

        image_store=image_features.unsqueeze(1)   #the image_fuse_module
        image_fine_feature=self.imageread(image_features)
       
        image_update=image_fine_feature.unsqueeze(1)
        out_put,_=self.mutitransformer(image_store,image_update,image_store)
        out_put=out_put.squeeze(1)
       # image_fine_feature=self.alpha*image_fine_feature+image_features
        if self.training:
            self.write(image_token)
            normalized_text_features = F.normalize(text_fine_feature, dim = -1)
            normalized_image_features=F.normalize(out_put,dim=-1)
            loss = F.l1_loss(normalized_text_features, text_token, reduction='mean') 
            loss2=  F.l1_loss(normalized_image_features,image_features , reduction='mean') 
        else:
            loss = 0.0
            loss2=0.0
        return text_fine_feature, out_put,loss,loss2

    def get_score(self, query, mem): 
        score = query @ mem.t() 
        score_query = F.softmax(score, dim = 0)
        score_mem = F.softmax(score, dim = 1)
        return score_query, score_mem
    
    def read(self, x):
        base_features = F.normalize(x, dim = -1) 
        base_features = x

        C, d = x.shape
        if self.training:
            self.text_fine_cache = self.text_fine_cache.detach()
        _, softmax_score_cache = self.get_score(base_features, self.text_fine_cache)
        fine_feature = softmax_score_cache @ self.text_fine_cache  # (N, d)
    
        return fine_feature

    def write(self, x):
        m, d = self.text_fine_cache.shape
        ratio = 0.2
        base_features = x.clone()
        base_features = self.writeTF(base_features)

        base_features = base_features.reshape(-1, d) # (B * P, d)
        base_features = F.normalize(base_features, dim = -1) 
        
        softmax_score_query, softmax_score_cache = self.get_score(base_features, self.text_fine_cache) #(B*P, 50)
        _, updating_indices = torch.topk(softmax_score_cache, 1, dim=1)
        
        updated_cache = self.text_fine_cache.clone().detach()
        for i in range(m):
            idx = torch.nonzero(updating_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                score = (softmax_score_query[idx, i] / torch.max(softmax_score_query[:, i]))
                updated_cache[i] = self.momentum * self.text_fine_cache[i] + (1 - self.momentum) * torch.sum(score * base_features[idx.squeeze(1)], dim=0)
        
        updated_cache = F.normalize(updated_cache, dim = -1)
        self.text_fine_cache = updated_cache.to(self.device)
    def  imageread(self,x):   #加工图片   ,还是当image_token处理可能会好一点     ,这里处理的是教师模块给的数据
    
        base_features = F.normalize(x, dim = -1) 
        if self.training:
            self.text_fine_cache = self.text_fine_cache.detach()     #加载上轮加载后的文本存储
        _, softmax_attention_cache = self.get_score(base_features, self.text_fine_cache)  #   (B*P,50)
        # print(softmax_attention_cache.size())
        # print(base_features.size())
        # softmax_attention_cache=softmax_attention_cache.unsqueeze(-1)
        # base_features=base_features.unsqueeze(2)
     #   base_features= softmax_attention_cache.T@base_features
        # base_features=(base_features*softmax_attention_cache) .sum(dim=2)
        base_features=softmax_attention_cache@self.text_fine_cache   #这个是能跑的通的，但是会有问题
        return base_features

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
        self.VPT_image_trans = self.VPT_image_trans.cuda()
        convert_weights(self.VPT_image_trans)

        self.mid_image_trans = Feature_Trans_Module_two_layer(512, 768)
        self.mid_image_trans = self.mid_image_trans.cuda()
        convert_weights(self.mid_image_trans)
       
        self.cfg = cfg
        
    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()
    
        image_features, image_fine = self.image_encoder(image.type(self.dtype), all_layer_outputs=True)
        image_features = self.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        image_fine_list=[]
        B, d = image_features.shape
        image_fine_features = [x[0] for x in image_fine] # B,N,d
        image_fine_attns =  [x[1] for x in image_fine] # B,1,N
        top_image_fine_list, layers= [], [-1]

        _, _, before_d = image_fine_features[0].shape
        for layer in layers:
            x = image_fine_features[layer]
            x = x.reshape(-1, before_d)
            x = self.mid_image_trans(x)
            x = x.reshape(B, -1, d)
            image_fine_feature = x
            image_fine_list.append(image_fine_feature)

            k = 5
            _, indices = torch.topk(image_fine_attns[layer], k = k,dim=-1)
            indices += 1
            indices = torch.cat((torch.zeros(B,1,dtype=torch.int64).cuda(),indices), dim=1)
            top_image_fine_feature = torch.gather(x, dim=1, index=indices.unsqueeze(-1).expand(B,k+1,d))
            
            avg_image_feature = torch.mean(image_fine_feature, dim = 1, keepdim=True)
            top_image_fine_feature = torch.cat((top_image_fine_feature, avg_image_feature), dim = 1)

            top_image_fine_list.append(top_image_fine_feature.reshape(-1, d))
            
        top_image_fine_list = torch.cat(top_image_fine_list)
        image_fine_list = torch.cat(image_fine_list)
        
        return image_features, logit_scale, image_fine_list, top_image_fine_list

 
class CustomCLIP_teacher(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, True)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model).cuda()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
  
    def forward(self, image=None, label=None):
        prompts = self.prompt_learner()

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        
        image_features = self.image_encoder(image.type(self.dtype)) 
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logits = logit_scale * image_features @ text_features.t()
        
        return image_features, text_features, logits


@TRAINER_REGISTRY.register()
class PromptKD(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTKD.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.PROMPTKD.PREC == "fp32" or cfg.TRAINER.PROMPTKD.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)
    
        print("Building memory")
        self.memory = Memory(clip_model,feature_dim=768)
        if cfg.TRAINER.MODAL == "base2novel":
            model_path = './teacher_model/'+str(cfg.DATASET.NAME)+'/VLPromptLearner/model-best.pth.tar'
        elif cfg.TRAINER.MODAL == "cross":
            model_path = './teacher_model/ImageNet-xd/VLPromptLearner_large/model.pth.tar-20'
            
        self.train_modal = cfg.TRAINER.MODAL
        
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_prefix2" in state_dict:
            del state_dict["prompt_learner.token_prefix2"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
        if "prompt_learner.token_suffix2" in state_dict:
            del state_dict["prompt_learner.token_suffix2"]
        
        self.model_teacher.load_state_dict(state_dict, strict=False)
        self.model_teacher.to(self.device)
        self.model_teacher.eval()
        
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "image_trans" in name or "extractor" in name or "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer

        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)
        self.trainable_list.append(self.memory)

        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        self.register_model("Memory", self.memory, self.optim, self.sched)
        


        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTKD.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.temperature = cfg.TRAINER.PROMPTKD.TEMPERATURE
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        with torch.no_grad():
            tea_image_features, tea_text_features, tea_logits = self.model_teacher(image)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.PROMPTKD.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            image_ft, logit_scale, image_fine, top_image_fine = model(image, label)
            fine_tea_text_features,fine_image_features, L_mem,L_img = self.memory(tea_text_features.detach(), image_fine,image_ft)
            image_ft=image_ft*0.95+fine_image_features*0.05
            stu_logits = logit_scale * image_ft @ fine_tea_text_features.t()
            
            top_stu_logits = logit_scale * top_image_fine @ tea_text_features.t()
            
            L_ukd = F.kl_div(
                F.log_softmax(stu_logits / self.temperature, dim=1),
                F.softmax(tea_logits / self.temperature, dim=1),
                reduction='sum',
            ) * (self.temperature * self.temperature) / stu_logits.numel()  # 求平均

            L_sem = F.kl_div(
                F.log_softmax(top_stu_logits / self.temperature, dim=1),
                F.softmax(tea_logits / self.temperature, dim=1).repeat_interleave(repeats = 7 ,dim = 0),
                reduction='sum',
            ) * (self.temperature * self.temperature) / top_stu_logits.numel()  # 求平均

            
       
 
            loss = self.cfg.TRAINER.PROMPTKD.KD_WEIGHT * L_ukd + self.cfg.TRAINER.PROMPTKD.KD_WEIGHT / 100 * L_sem + \
                                                                self.cfg.TRAINER.PROMPTKD.KD_WEIGHT / 50 * L_mem+ \
                                                                 self.cfg.TRAINER.PROMPTKD.KD_WEIGHT/100*L_img    #原来是100
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item(),
                        "L_ukd": self.cfg.TRAINER.PROMPTKD.KD_WEIGHT * L_ukd.item(),
                        "L_sem": self.cfg.TRAINER.PROMPTKD.KD_WEIGHT / 100 * L_sem.item(),
 
                        "L_mem": self.cfg.TRAINER.PROMPTKD.KD_WEIGHT / 100 * L_mem.item(),
                        "L_img": self.cfg.TRAINER.PROMPTKD.KD_WEIGHT / 100 * L_img.item()
                        }

        if (self.batch_idx + 1) == self.num_batches:
            self.sched.step()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]
                
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]
                
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            if 'Mem' in name:
                self._models[name].text_fine_cache = checkpoint["memory_item"]

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)
            
            with torch.no_grad():
                tea_image_features, tea_text_features, tea_logits = self.model_teacher(image, label)

            image_ft, logit_scale, image_fine, _ = self.model(image, label)



            tea_text_features, tea_image_features,L_mem ,_= self.memory(tea_text_features, image_fine,image_ft)
            image_ft=0.95*image_ft+0.05*tea_image_features
            if self.train_modal == "base2novel":
                if split == "val":
                    output = logit_scale * image_ft @ tea_text_features[:math.ceil(self.n_cls / 2),:].t()
                elif split == "test":
                    output = logit_scale * image_ft @ tea_text_features[math.ceil(self.n_cls / 2):,:].t()
            elif self.train_modal == "cross" :
                output = logit_scale * image_ft @ tea_text_features.t()
            
            self.evaluator.process(output, label) 

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            if 'Mem' in name:
                print("save memory item...")
                save_checkpoint(
                    {
                        "memory_item": self._models[name].text_fine_cache,
                        "state_dict": model_dict,
                        "epoch": epoch + 1,
                        "optimizer": optim_dict,
                        "scheduler": sched_dict,
                        "val_result": val_result
                    },
                    osp.join(directory, name),
                    is_best=is_best,
                    model_name=model_name,
                )
            else:
                save_checkpoint(
                    {
                        "state_dict": model_dict,
                        "epoch": epoch + 1,
                        "optimizer": optim_dict,
                        "scheduler": sched_dict,
                        "val_result": val_result
                    },
                    osp.join(directory, name),
                    is_best=is_best,
                    model_name=model_name,
                )

 