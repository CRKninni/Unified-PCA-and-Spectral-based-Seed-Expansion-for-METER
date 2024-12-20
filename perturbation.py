# import gradio as gr
import torch
import copy
import numpy as np
import json
import pytorch_lightning as pl
import csv

from PIL import Image

from meter.config import ex
from meter.modules import METERTransformerSS

from meter.transforms import clip_transform
from meter.datamodules.datamodule_base import get_pretrained_tokenizer

from object_discovery import *
import spectral.vqa_data as vqa_data
import spectral.vqa_utils as utils

import random
from tqdm import tqdm
from perturbation_helper import get_image_relevance, get_text_relevance
import gc
import spectral.vqa_data as vqa_data
import spectral.vqa_utils as utils
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig
from meter.modules.bert_model import BertCrossLayer
from meter.modules import swin_transformer as swin
from meter.modules import heads, objectives, meter_utils
from meter.modules.clip_model import build_model, adapt_position_encoding
from meter.modules.swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig
from meter.modules.roberta_model import RobertaModel
from meter.modules.layers import *
from sklearn.metrics import auc
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
VQA_URL = 'spectral/vqa_dict.json'

class METERTransformerSS1(pl.LightningModule):
    def __init__(self, config, COCO_val_path):
        
        super().__init__()
        self.save_hyperparameters()
        self.COCO_VAL_PATH = COCO_val_path
        self.tokenizer = get_pretrained_tokenizer(config["tokenizer"])
        f = open(VQA_URL)
        self.vqa_answers = json.load(f)

        self.vqa_dataset = vqa_data.VQADataset(splits="valid")

        self.pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        self.pert_acc = [0] * len(self.pert_steps)


    
        self.is_clip = (not 'swin' in config['vit'])

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        resolution_after=config['image_size']

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(
                        pretrained=True, config=self.hparams.config,
                    )

                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])

            torch.distributed.barrier()

        if self.is_clip:
            self.vit_model = build_model(config['vit'], resolution_after=resolution_after)
        else:
            self.vit_model = getattr(swin, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config,
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        if 'roberta' in config['tokenizer']:
            self.text_transformer = RobertaModel.from_pretrained(config['tokenizer'])
            # self.text_transformer = RobertaModel(config['tokenizer'])
        else:
            self.text_transformer = BertModel.from_pretrained(config['tokenizer'])

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"]*2)
            self.itm_score.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential( #baka
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            # self.vqa_classifier = nn.Sequential( #baka custom layers for lrp
            #     Linear(hs * 2, hs * 2),
            #     LayerNorm(hs * 2,  eps=1e-12),
            #     GELU(),
            #     Linear(hs * 2, vs),
            # )
            self.vqa_classifier.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=False)


        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 4, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["snli"] > 0:
            self.snli_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.snli_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        meter_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=False)


    def infer_mega(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        image_embeds = self.vit_model(img)


        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        x, y = text_embeds, image_embeds
        all_text_feats, all_image_feats = [], []
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]
            all_text_feats.append(x.detach().clone())
            all_image_feats.append(y.detach().clone())
            # print(y.shape)

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        # print("helloooooooooooooooooooooooooooooooooooooo")
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "all_text_feats": all_text_feats,
            "all_image_feats": all_image_feats
        }


        return ret



    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)

        image_embeds = self.vit_model(img)

        return text_embeds, image_embeds, text_masks, image_token_type_idx



    def meter_vqa(self, text_embeds, image_embeds, text_masks, image_token_type_idx):

        device = text_embeds.device
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, text_masks.size(), device)
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]


        text_embeds = self.cross_modal_text_transform(text_embeds)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=self.device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device=self.device)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
    

        x, y = text_embeds, image_embeds
        all_text_feats, all_image_feats = [], []
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]
            all_text_feats.append(x.detach().clone())
            all_image_feats.append(y.detach().clone())

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)

        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)


        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "all_text_feats": all_text_feats,
            "all_image_feats": all_image_feats
        }


        return ret
    
    def whatever (self, url, text, device="cuda"):
        try:

            image = Image.open(url)
            orig_shape = np.array(image).shape
            img = clip_transform(size=576)(image)
            img = img.unsqueeze(0).to(device)

        except:
            return False

        batch = {"text": [text], "image": [img]}

        encoded = self.tokenizer(batch["text"])
        batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
        return batch

    def perturbation_image(self, item, cam_image, is_positive_pert=False):
        if is_positive_pert:
            cam_image = cam_image * (-1)
        
        
        image_file_path = item['img_id']
        batch = self.whatever(image_file_path, item['sent'])
        text_embeds, image_embeds, text_masks, image_token_type_idx = self.infer(batch)

        for step_idx, step in enumerate(self.pert_steps):
            cam_pure_image = cam_image[1:]
            image_len = cam_pure_image.shape[0]
            curr_num_boxes = int((1 - step) * image_len)
            _, top_bboxes_indices = cam_pure_image.topk(k=curr_num_boxes, dim=-1)
            top_bboxes_indices = top_bboxes_indices.cpu().data.numpy()


            # add back [CLS] token
            top_bboxes_indices = [0] +\
                                 [top_bboxes_indices[i] + 1 for i in range(len(top_bboxes_indices))]
            # text tokens must be sorted for positional embedding to work
            top_bboxes_indices = sorted(top_bboxes_indices)

            curr_features = image_embeds[:, top_bboxes_indices, :]
            # curr_features_masks = extend_image_masks[:, top_bboxes_indices, :]

            ret = self.meter_vqa(text_embeds, curr_features, text_masks, image_token_type_idx)
            vqa_logits = self.vqa_classifier(ret["cls_feats"])
            answer = self.vqa_answers[str(vqa_logits.argmax().item())]

            accuracy = item["label"].get(answer, 0)
            
            self.pert_acc[step_idx] += accuracy
            


        return self.pert_acc
    


    def perturbation_text(self, item, cam_text, is_positive_pert=False):
        if is_positive_pert:
            cam_text = cam_text * (-1)

        image_file_path = item['img_id']
        batch = self.whatever(image_file_path, item['sent'])
        text_embeds, image_embeds, text_masks, image_token_type_idx = self.infer(batch)


        for step_idx, step in enumerate(self.pert_steps):
            cam_pure_text = cam_text[1:-1]
            text_len = cam_pure_text.shape[0]
            curr_num_tokens = int((1 - step) * text_len)
            _, top_bboxes_indices = cam_pure_text.topk(k=curr_num_tokens, dim=-1)
            top_bboxes_indices = top_bboxes_indices.cpu().data.numpy()

            top_bboxes_indices = [0, cam_text.shape[0] - 1] +\
                                 [top_bboxes_indices[i] + 1 for i in range(len(top_bboxes_indices))]
            top_bboxes_indices = sorted(top_bboxes_indices)

            curr_text_features = text_embeds[:, top_bboxes_indices, :]
            curr_text_masks = text_masks[:, top_bboxes_indices]


            ret = self.meter_vqa(curr_text_features, image_embeds, curr_text_masks, image_token_type_idx)
            vqa_logits = self.vqa_classifier(ret["cls_feats"])
            answer = self.vqa_answers[str(vqa_logits.argmax().item())]
            accuracy = item["label"].get(answer, 0)
            self.pert_acc[step_idx] += accuracy
 
            
        return self.pert_acc


def main1(_config):
    COCO_path = _config['COCO_path']
    _config = copy.deepcopy(_config)

    loss_names = {
        "itm": 0,
        "mlm": 1,
        "mpp": 0,
        "vqa": 1,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
    }

    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    url = 'spectral/vqa_dict.json'
    f = open(url)
    id2ans = json.load(f)

    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    model = METERTransformerSS(_config)
    model.setup("test")
    model.eval()
    
    model_pert = METERTransformerSS1(_config, COCO_path)
    model_pert.setup("test")
    model_pert.eval()


    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)
    model_pert.to(device)
    vqa_dataset = vqa_data.VQADataset(splits="valid")


    items = vqa_dataset.data
    random.seed(1234)
    r = list(range(len(items)))
    random.shuffle(r)
    pert_samples_indices = r[:3303]

    iterator = tqdm([vqa_dataset.data[i] for i in pert_samples_indices])

    test_type = _config["test_type"]
    method_name = _config["method_name"]
    is_positive_pert = _config["is_positive_pert"]
    modality = _config["modality"]
    IMG_SIZE = 576


    print("running {0} pert test for {1} modality with method {2}".format(test_type, modality, method_name))
    
    def infer(url,text):
        image = Image.open(url)
        img = clip_transform(size=IMG_SIZE)(image)
        img = img.unsqueeze(0).to(device)
        batch = {"text": [text], "image": [img]}
        
        encoded = tokenizer(batch["text"])
        text_tokens = tokenizer.tokenize(batch["text"][0])
        batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
        
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        ret = model.infer(batch)
        vqa_logits = model.vqa_classifier(ret["cls_feats"])
        
        answer = id2ans[str(vqa_logits.argmax().item())]
        output = vqa_logits
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads = []
        cams = []

        for i in range(len(model.cross_modal_image_layers)):
            grad = model.cross_modal_image_layers[i].attention.self.get_attn_gradients().detach()
            cam = model.cross_modal_image_layers[i].attention.self.get_attention_map().detach()

            grads.append(grad)
            cams.append(cam)
        
        grads_text = []
        cams_text = []
        for i in range(len(model.cross_modal_text_layers)):
            grad = model.cross_modal_text_layers[i].attention.self.get_attn_gradients().detach()
            cam = model.cross_modal_text_layers[i].attention.self.get_attention_map().detach()
            grads_text.append(grad)
            cams_text.append(cam)
        
        return grads, cams, ret, grads_text, cams_text
        
        

    all_results = []
    for index, item in enumerate(iterator):
        
        item['img_id'] = COCO_path + item['img_id'] + '.jpg'
        grads, cams, ret,grads_text,cams_text = infer(item['img_id'], item['sent'])
        R_t_i = None
        if modality == "text":
            text_relevance = get_text_relevance(ret['text_feats'].squeeze(), grads_text, cams_text)
            cams_text = torch.cat((torch.zeros(1).to(device), torch.tensor(text_relevance).to(device),torch.zeros(1).to(device)))
            cams_text = (cams_text - cams_text.min()) / (cams_text.max() - cams_text.min())
            curr_pert_result = model_pert.perturbation_text(item, cams_text, is_positive_pert)
        else:
            image_relevance = get_image_relevance(ret, grads, cams)
            R_t_i = torch.cat((torch.zeros(1).to(device), torch.tensor(image_relevance).to(device)))
            cam_image = R_t_i
            cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min())
            curr_pert_result = model_pert.perturbation_image(item, cam_image, is_positive_pert)
            
        curr_pert_result = [round(res / (index + 1) * 100, 2) for res in curr_pert_result]
        
        all_results.append(curr_pert_result)

        iterator.set_description("Acc: {}".format(curr_pert_result))
        
        del R_t_i
        gc.collect()
        torch.cuda.empty_cache()
        

    y_values = np.mean(all_results,axis=0)

    x_values = model_pert.pert_steps

    new_auc = auc(x_values, y_values)

    print(f"Final AUC: {new_auc}")
    print(f"y values {y_values}")
    print(f"xvalues {x_values}")
        
        
        
        



if __name__ == "__main__":
    @ex.automain
    def main(_config):
        main1(_config)