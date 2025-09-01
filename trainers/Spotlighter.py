import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import convert_weights
from tqdm import tqdm
from dassl.utils import save_checkpoint
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


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype
        

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    # zero-shot clip
    def encode_text(self, text):
        text = text.cuda()
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
                
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init


        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]


        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):    #这边是不用改的,选取典型的特点它这边k为5
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)     #这个是不用改的
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts    #不改
        self.image_encoder = clip_model.visual    #这个是不用改的
        self.text_encoder = TextEncoder(clip_model)    #这个是不用改的
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.mid_image_trans = Feature_Trans_Module_two_layer(512, 512)
        self.mid_image_trans = self.mid_image_trans.cuda()
        convert_weights(self.mid_image_trans)
        
        
       
    def forward(self, image):
        image_features, image_fine = self.image_encoder(image.type(self.dtype), all_layer_outputs=True)

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        image_fine_list = []
        top_image_fine_list = []
        top_image_top_fine_list=[]
        top_image_low_fine_list=[]
        B, d = image_features.shape
        image_fine_features = [x[0] for x in image_fine] # B,N,d
        image_fine_attns =  [x[1] for x in image_fine] # B,1,N
        layers=[-1] 
        _, _, before_d = image_fine_features[0].shape
        loss = 0.0
        for i, layer in enumerate(layers):
            x = image_fine_features[layer]
            x = x.reshape(-1, before_d)

            if self.training:
                x = self.mid_image_trans(x)
                x = x.reshape(B, -1, d)

                image_fine_feature = x
                image_fine_list.append(image_fine_feature)
                
                
                k = 5
                _, indices = torch.topk(image_fine_attns[layer], k = k ,dim=-1)   #（64，5）   the size of batch is 64
              #  print({"indices1":indices.size()})
                indices_top=indices[:,:2]     #2
                indices_low =indices[:,2:]   #3
                #indices += 1
                #
               # indices_low+=1
                indices = torch.cat((torch.zeros(B,1,dtype=torch.int64).cuda(),indices), dim=1) #(64,6)
                indices_top=torch.cat((torch.zeros(B,1,dtype=torch.int64).cuda(),indices_top), dim=1)    #(64,3)
                indices_low=torch.cat((torch.zeros(B,1,dtype=torch.int64).cuda(),indices_low),dim=1)      #（64，）
               #print({"indices2":indices.size()})
                top_image_fine_feature = torch.gather(x, dim=1, index=indices.unsqueeze(-1).expand(B, k+1, d))
                top_image_top_fine_feature = torch.gather(x, dim=1, index=indices_top.unsqueeze(-1).expand(B, 2+1, d))  #64
                top_image_low_fine_feature = torch.gather(x, dim=1, index=indices_low.unsqueeze(-1).expand(B, 3+1, d))   #64  
              #  print({"0":len(top_image_fine_feature)})    #(64)
                avg_image_feature = torch.mean(x, dim = 1, keepdim = True)
                top_image_fine_feature = torch.cat((top_image_fine_feature, avg_image_feature), dim = 1)
                top_image_top_fine_feature=torch.cat(( top_image_top_fine_feature,avg_image_feature),dim=1)
                top_image_low_fine_feature=torch.cat(( top_image_low_fine_feature,avg_image_feature),dim=1)
                top_image_fine_list.append(top_image_fine_feature.reshape(-1, d))
                top_image_top_fine_list.append(top_image_top_fine_feature.reshape(-1, d))
                top_image_low_fine_list.append(top_image_low_fine_feature.reshape(-1, d))
               # print({"1":len(top_image_fine_list)})

        
        if len(image_fine_list) > 0:
            image_fine_list = torch.cat(image_fine_list)
            top_image_fine_list = torch.cat(top_image_fine_list)
            top_image_top_fine_list=torch.cat(top_image_top_fine_list)
            top_image_low_fine_list=torch.cat(top_image_low_fine_list)
            #print({"2":len(top_image_fine_list)})   #(448)
        
        return text_features, image_features, logit_scale, image_fine_list, top_image_fine_list,top_image_top_fine_list,top_image_low_fine_list

#加一个多头的模块
class MultiHeadAttention(nn.Module):
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
    
class TextMemory(nn.Module):
    def __init__(self, clip_model,feature_dim = 512, memory_size = 25,num_heads=8, reduction = 4, frozen_text_embedding=None, alpha=0.2,momentum=0.8):
        super().__init__()
        self.device = clip_model.dtype
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.text_fine_cache = F.normalize(torch.rand(self.memory_size, feature_dim), dim = -1)
        self.text_fine_cache = self.text_fine_cache.to(self.device)
        self.text_fine_cache = self.text_fine_cache.cuda()
        self.mutitransformer=MultiHeadAttention(feature_dim,num_heads)
        self.mutitransformer=self.mutitransformer.to(self.device)
        self.mutitransformer=self.mutitransformer.cuda()
        self.alpha = alpha
        self.momentum = momentum

        if frozen_text_embedding is not None:
            self.frozen_text_embedding = frozen_text_embedding

        
        self.extractor = nn.Linear(2 * feature_dim , feature_dim, bias=False)     #mlp
        self.extractor = self.extractor.to(self.device)
        self.extractor = self.extractor.cuda()


        self.writeTF = lambda x: x.clone()
        
    def forward(self, text_token=None, image_token=None,image_feature=None):     #image_feature是正常
        fine_cache_feature = self.read(text_token)         #choose top-k module，像对齐名不详选择

        text_cache_feature = torch.cat((text_token, fine_cache_feature), dim = -1)
        text_cache_feature = self.alpha * self.extractor(text_cache_feature) + text_token     #text_fuse_module
        image_store=image_feature.unsqueeze(1)
        image_fine_feature=self.imageread(image_feature)     #choose top-k module
        #这边加一个多头注意力在这，作为image_fuse_modul
        #print("image_fine_feature",image_token.size())
        #image_fine_feature=image_fine_feature.unsqueeze(1)    
        image_update=image_fine_feature.unsqueeze(1)
        out_put,_=self.mutitransformer(image_store,image_update,image_update)
        out_put=out_put.squeeze(1)
        if self.training:
            _ = self.write(image_token)
            normalized_text_features = F.normalize(text_cache_feature, dim = -1)
            normalized_image_features=F.normalize(out_put,dim=-1)
            loss1 = F.l1_loss(normalized_text_features, text_token, reduction='mean') 
            loss2=  F.l1_loss(normalized_image_features,image_feature , reduction='mean') 
        else:
            loss1 = 0.0
            loss2 = 0.0
        return text_cache_feature,out_put,loss1,loss2

    def get_score(self, query, mem):
        score = query @ mem.t() 
        score_query = F.softmax(score, dim = 0)
        score_mem = F.softmax(score, dim = 1)
        return score_query, score_mem
    
    def read(self, x):
        base_features = F.normalize(x, dim = -1) 
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

        loss = 0.0
        self.text_fine_cache = updated_cache.to(self.device)
        return loss
    def diversityloss(self, mem):
        # it is same with orthonomal constraints.
        cos_sim = torch.matmul(mem,torch.t(mem))
        margin = 0 
        cos_sim_pos = cos_sim-margin
        cos_sim_pos[cos_sim_pos<0]=0
        loss = (torch.sum(cos_sim_pos)-torch.trace(cos_sim_pos))/(self.memory_size*(self.memory_size-1))
        return loss
    
    def  imageread(self,x):   #加工图片   输入的是通过mlp层的图像特征
        # x=torch.tensor(x)
        # x=x.to(self.device)
        # x=x.cuda()
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

 


@TRAINER_REGISTRY.register()
class Spotlighter(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)   #the backbone of the model    这个是要注意

        print("Building memory")
        self.memory = TextMemory(clip_model,feature_dim=512, memory_size=cfg.TRAINER.TF.MEMORY_SIZE, alpha=cfg.TRAINER.TF.ALPHA)
        #self.image_memory = ImageMemory(clip_model,feature_dim=512, image_memory_size=cfg.TRAINER.TF.MEMORY_SIZE, alpha=cfg.TRAINER.TF.ALPHA)
        if cfg.TRAINER.TF.BALANCE:
            self.balance = cfg.TRAINER.TF.BALANCE
        if cfg.TRAINER.TF.DISTILL:
            self.distill = cfg.TRAINER.TF.DISTILL
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name or 'mid_image_trans' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
            
        print("check hyper-parameter balance: ", self.balance)
        print("check hyper-parameter distill: ", self.distill)
        print("check hyper-parameter alpha: ", self.memory.alpha)
        print("check hyper-parameter momentum: ", self.memory.momentum)
#        print("check hyper-parameter momentum: ", self.image_memory.momentum)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)         #这个是要注意
        self.trainable_list.append(self.memory)        #这个是要注意
        #self.trainable_list.append(self.image_memory)
        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model, self.optim, self.sched)   #   这个是要注意
        self.register_model("Memory", self.memory, self.optim, self.sched)
       # self.register_model("ImageMemory", self.image_memory, self.optim, self.sched)

        enabled = set()
        for name, param in self.trainable_list.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:         #下面是文章的关键
            text_features, image_features, logit_scale, image_fine, top_image_fine,top_top_image_fine,top_low_image_fine = self.model(image)  #top-2和top-3
            
            fine_text_features,fine_image_features, loss3 ,loss4= self.memory(text_features, image_fine,image_features)

            #fine_image_features, loss4 = self.image_memory(text_features, image_features)
            image_features=0.95*image_features+0.05*fine_image_features
            output = logit_scale * image_features @ fine_text_features.t()
            top_local_logit = logit_scale * top_image_fine @ text_features.t()
            top_top_local_logit=logit_scale * top_top_image_fine @ text_features.t()
            top_low_local_logit=logit_scale * top_low_image_fine @ text_features.t()
            # print(output.size())
            #print({"label":label.size()})
            # print({"top_2":top_top_image_fine.size()})
            # print({"top_3":top_low_image_fine.size()})
            # print({"top_5":top_image_fine.size()})
            # print({"logit_scale":logit_scale.size()})
            # print({"text_features":text_features.size()})
            # print({"top_local_logit":top_local_logit.size()})
            # print({"top_top_local_logit":top_top_local_logit.size()})
            # print({"top_low_local_logit":top_low_local_logit.size()})
            #print({"label":label.size()})
            # output=output.mean(dim=1)
            # output=output.float()
            loss1 = F.cross_entropy(output, label)
            
            loss2 = F.cross_entropy(top_local_logit, label.repeat_interleave(dim=0, repeats=7))

            loss_top=F.cross_entropy(top_top_local_logit,label.repeat_interleave(dim=0,repeats=4))
            loss_low=F.cross_entropy(top_low_local_logit,label.repeat_interleave(dim=0,repeats=5))
            
            loss = loss1 + loss2 / self.balance + loss3 * self.distill+ loss4 *self.distill+loss_low/(self.balance)+loss_top/(self.balance)
          # loss = loss1 + loss2 / self.balance + loss3 * self.distill+ loss4 *self.distill 
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "global": loss1.item(),
            "local": loss2.item() / self.balance,
            "top":loss_low.item()/(self.balance),
            "low":loss_top.item()/(self.balance),
            "feature": loss3.item() * self.distill,
            "image_feature": loss4.item() * self.distill*20,

            "acc": compute_accuracy(output, label)[0].item(),
        }
        # loss_summary = {
        #     "loss": loss.item(),
        #     "global": loss1.item(),
        #     "local": loss2.item() / self.balance,
        #     "top":loss_low.item()/(self.balance*100),
        #     "low":loss_top.item()/(self.balance*100),
        #     "feature": loss3.item() * self.distill,
        #     "image_feature": loss4.item() * self.distill*20,

        #     "acc": compute_accuracy(output, label)[0].item(),
        # }


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

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            
            if "prompt_learner.tokenized_prompts" in state_dict:
                del state_dict["prompt_learner.tokenized_prompts"]

            if "frozen_text_embedding" in state_dict:
                del state_dict["frozen_text_embedding"]

            # Ignore abstract layer
            keys=[key for key in state_dict.keys() if 'mid_image_trans' in key]
            state_dict = {k:v for k,v in state_dict.items() if k not in keys}

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            if 'TextMemory' in name:
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
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)

            text_features, image_features, logit_scale, image_fine, _,_,_ = self.model(image)
            fine_text_features,fine_image_features ,_,_ = self.memory(text_features, image_fine,image_features)
          #  fine_images_features,_=self.image_memory(text_features,image_features)
            image_features=0.9*image_features+0.1*fine_image_features
            # print("fine:",fine_image_features.size())
            # print(image_features.size())
            output = logit_scale * image_features @ fine_text_features.t()
           # output=output.mean(dim=1)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        

        return list(results.values())[0]
    
    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
                self.training = True
            elif mode in ["test", "eval"]:
                self._models[name].eval()
                self.training = False
            else:
                raise KeyError
            
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

            if 'TextMemory' in name:
                print("save memory item...")
                save_checkpoint(
                    {
                        "memory_item": self._models[name].text_fine_cache,
                       # "memory_item": self._models[name].image_fine_cache,
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
            # elif "ImageMemory" in name:    #问题在于imagememory
            #     print("save memory item...")
            #     save_checkpoint(
            #         {
            #             #"memory_item": self._models[name].text_fine_cache,
            #             "memory_item": self._models[name].image_fine_cache,
            #             "state_dict": model_dict,
            #             "epoch": epoch + 1,
            #             "optimizer": optim_dict,
            #             "scheduler": sched_dict,
            #             "val_result": val_result
            #         },
            #         osp.join(directory, name),
            #         is_best=is_best,
            #         model_name=model_name,
            #     )
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