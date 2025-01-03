import torch
import cv2
import copy
import requests
import io
import numpy as np
import json
import urllib.request
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image

from meter.config import ex
from meter.modules import METERTransformerSS

from meter.transforms import clip_transform
from meter.datamodules.datamodule_base import get_pretrained_tokenizer

import matplotlib.pyplot as plt
from object_discovery import *

from perturbation_helper import get_image_relevance, get_text_relevance



def main1(_config, item, model = None, viz = True, is_pert = False, tokenizer = None):

    if is_pert:
        img_path = item['img_id'] + '.jpg'
        question = item['sent']
    else:
        img_path, question = item

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

    if not is_pert:
        tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    

    with urllib.request.urlopen(
        "https://github.com/dandelin/ViLT/releases/download/200k/vqa_dict.json"
    ) as url:
        id2ans = json.loads(url.read().decode())


    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    if not is_pert:
        model = METERTransformerSS(_config)
        model.setup("test")
        model.eval()

    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)

    IMG_SIZE = 576


    def infer(url, text):
        # loading the image
        try:
            if "http" in url:
                res = requests.get(url)
                image = Image.open(io.BytesIO(res.content)).convert("RGB")
            else:
                image = Image.open(url)
            img = clip_transform(size=IMG_SIZE)(image)
            img = img.unsqueeze(0).to(device)

        except Exception as e:
            print(f"EXCEPTION: {e}")
            return False

        batch = {"text": [text], "image": [img]}

        encoded = tokenizer(batch["text"])
        text_tokens = tokenizer.tokenize(batch["text"][0])
        
        
        batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
        
        
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        if not is_pert:
            ret = model.infer(batch)
        else:
            ret = model.infer_mega(batch)

        vqa_logits = model.vqa_classifier(ret["cls_feats"])

        answer = id2ans[str(vqa_logits.argmax().item())]
        output = vqa_logits
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if is_pert:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)


        model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        grads = []
        cams = []
        grads_text = []
        cams_text = []
        
        for i in range(len(model.cross_modal_image_layers)):
            grads.append(model.cross_modal_image_layers[i].attention.self.get_attn_gradients().detach())
            cams.append(model.cross_modal_image_layers[i].attention.self.get_attention_map().detach())
            
        for i in range(len(model.cross_modal_text_layers)):
            grads_text.append(model.cross_modal_text_layers[i].attention.self.get_attn_gradients().detach())
            cams_text.append(model.cross_modal_text_layers[i].attention.self.get_attention_map().detach())


        image_rel = get_image_relevance(ret,grads, cams)
        text_rel = get_text_relevance(ret,grads_text, cams_text)
        
        image_rel = torch.tensor(image_rel).to(device)
        text_rel = torch.tensor(text_rel).to(device)
    
        text_rel = F.pad(text_rel, (1, 1), value=0)
        
        return answer, text_rel, image_rel, img, text_tokens, tokens
    



    result, text_relevance, image_relevance, image, text_tokens, tokens = infer(img_path, question)


    if viz:
        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=IMG_SIZE, mode='bilinear')
        image_relevance = image_relevance.reshape(IMG_SIZE, IMG_SIZE).cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())


        def show_cam_on_image(img, mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            return cam


        image = image[0].permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        vis = show_cam_on_image(image, image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)


        fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
        axs[0].imshow(vis)
        axs[0].axis('off')
        axs[0].set_title('Image Relevance')

        ti = axs[1].imshow(text_relevance.unsqueeze(dim = 0).numpy())
        axs[1].set_title("Word Impotance")
        plt.sca(axs[1])
        plt.xticks(np.arange(len(text_tokens) + 2), [ '[CLS]' ] + text_tokens + [ '[SEP]' ])
        plt.colorbar(ti, orientation = "horizontal", ax = axs[1])
        plt.show()


    if is_pert:
        return text_relevance, image_relevance
    else:
        return text_relevance, image_relevance, result
    


if __name__ == '__main__':
    @ex.automain
    def main (_config):
        test_img = _config['img']
        test_question = _config['question']

        if test_img == '' or test_question == '':
            print("Provide an image and a corresponding question for VQA")

        else:
            item = (test_img, test_question)
            _, _, answer = main1(_config, item, viz = True)
            print(f"QUESTION: {test_question}\nANSWER: {answer}")

    
# python demo_vqa_ours.py with num_gpus=0 load_path="/home/charan/ExplanableAI/METER-spectral-interpretability/meter_clip16_288_roberta_vqa.ckpt" test_only=True img="/home/charan/ExplanableAI/CoCo/val2014/COCO_val2014_000000000488.jpg" question="What game is been played?"
