from flash_attn.models.bert import BertEncoder
from transformers import BertConfig

import torch.nn as nn
import numpy as np
import torch


HITANNO_DEFAULT_PARAMS = {
    "embed_dim": 128,
    "nhead1": 16, 
    "dim_feedforward1": 128,
    "num_encoder_layers1": 1,
    "nhead2": 16, 
    "dim_feedforward2": 128,
    "num_encoder_layers2": 1,
    "hidden_dim": 64,}


class HitAnno(nn.Module):
    def __init__(self,
                 batch_size,
                 peak_num_list,
                 peak_num_all,
                 output_size,
                 embed_dim=HITANNO_DEFAULT_PARAMS["embed_dim"], 
                 nhead1=HITANNO_DEFAULT_PARAMS["nhead1"], 
                 dim_feedforward1=HITANNO_DEFAULT_PARAMS["dim_feedforward1"],
                 num_encoder_layers1=HITANNO_DEFAULT_PARAMS["num_encoder_layers1"],
                 nhead2=HITANNO_DEFAULT_PARAMS["nhead2"], 
                 dim_feedforward2=HITANNO_DEFAULT_PARAMS["dim_feedforward2"],
                 num_encoder_layers2=HITANNO_DEFAULT_PARAMS["num_encoder_layers2"],
                 hidden_dim=HITANNO_DEFAULT_PARAMS["hidden_dim"],
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 use_flash_attn=False,):
        super(HitAnno, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.parameter_dict = {"embed_dim": embed_dim,
                               "transEncoder1": {"nhead1": nhead1, "dim_feedforward1": dim_feedforward1, "num_encoder_layers1": num_encoder_layers1},
                               "transEncoder2": {"nhead2": nhead2, "dim_feedforward2": dim_feedforward2, "num_encoder_layers2": num_encoder_layers2},
                               "mlp": {"hidden_dim": hidden_dim, "output_size": output_size}}
        self.CLS_TOKEN = 0
        self.PAD_TOKEN = 1
        self.SEP_TOKEN = 2
        class_num = len(peak_num_list)
        self.peak_num_list = peak_num_list
        peak_num = min(peak_num_list)
        self.peak_num = peak_num
        self.peak_num_all = peak_num_all
        self.class_num = class_num
        self.input_embedding = nn.Embedding(peak_num_all + 3 + self.class_num, embed_dim, padding_idx=1)
        self.posi_embedding = nn.Embedding(2, embed_dim)
        transEncdoer1_config = BertConfig(hidden_size=embed_dim, num_attention_heads=nhead1, intermediate_size=dim_feedforward1, num_hidden_layers=num_encoder_layers1, use_flash_attn=use_flash_attn)
        self.transEncoder1 = BertEncoder(transEncdoer1_config)
        transEncdoer2_config = BertConfig(hidden_size=embed_dim, num_attention_heads=nhead2, intermediate_size=dim_feedforward2, num_hidden_layers=num_encoder_layers2, use_flash_attn=use_flash_attn)
        self.transEncoder2 = BertEncoder(transEncdoer2_config)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )
        self.input_idx_mat = np.zeros((batch_size * class_num, 1 + peak_num), dtype=np.int32)
        peak_ptr = 3 + class_num
        for i in range(class_num):
            self.input_idx_mat[i, 0] = i + 3
            for j in range(1, peak_num + 1):
                self.input_idx_mat[i, j] = peak_ptr
                peak_ptr += 1
        for i in range(1, batch_size):
            self.input_idx_mat[class_num * i: class_num * (i + 1)] = self.input_idx_mat[class_num * (i - 1): class_num * i]
        self.input_idx_mat = torch.tensor(self.input_idx_mat, dtype=torch.long).to(self.device)

    def forward(self, input: torch.Tensor):
        train_embedding = self.input_embedding(self.input_idx_mat[:input.shape[0] * self.class_num])
        posi_embed_mat = self.posi_embedding(input.type(torch.long))
        train_embedding[:, 1:] += posi_embed_mat.reshape((-1, posi_embed_mat.shape[-2], posi_embed_mat.shape[-1]))

        sub_cls_output = self.transEncoder1(train_embedding)

        cls_embedding = self.input_embedding(torch.zeros((input.shape[0]), dtype=torch.long, device=self.device)).unsqueeze(dim=1)
        compond_embeddings = torch.cat([cls_embedding, sub_cls_output[:,0,:].reshape((input.shape[0], self.class_num, -1))], dim=1)

        transformer_output = self.transEncoder2(compond_embeddings)
        cell_embedding = transformer_output[:, 0, :]
        output = self.mlp(cell_embedding)
        return output, cell_embedding
    
    def save_parameters(self, col_list, path: str):
        torch.save({
                    "state_dict": self.state_dict(),
                    "batch_size": self.batch_size,
                    "peak_num_list": self.peak_num_list,
                    "peak_num_all": self.peak_num,
                    "col_list": col_list,
                    "model_parameter": self.parameter_dict,
                    }, 
                    path)
        return
