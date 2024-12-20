import torch
import copy

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention


def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition


def apply_mm_attention_rules(R_ss, R_qq, R_qs, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_ss_addition = torch.matmul(cam_sq, R_qs)
    return R_sq_addition, R_ss_addition

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    # computing R hat
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    # print(f"self_attn (handle residual): {self_attention.shape} | diag idx: {diag_idx}")
    # assert self_attention[diag_idx, diag_idx].min() >= 0
    # normalizing R hat
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention


class GeneratorBaselines:
    def __init__(self, model, normalize_self_attention, apply_self_in_rule_10):
        self.model = model
        self.normalize_self_attention = normalize_self_attention 
        self.apply_self_in_rule_10 = apply_self_in_rule_10

    def handle_self_attention_image(self, blk):
        cam = blk.attention.self.get_attention_map().detach()
        grad = blk.attention.self.get_attn_gradients().detach()
        cam = avg_heads(cam, grad)
        R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
        self.R_i_i += R_i_i_add
        self.R_i_t += R_i_t_add


    def handle_co_attn_image(self, block):
        
        cam_i_t = block.crossattention.self.get_attention_map().detach()
        grad_i_t = block.crossattention.self.get_attn_gradients().detach()
        cam_i_t = avg_heads(cam_i_t, grad_i_t)
        R_i_t_addition, R_i_i_addition = apply_mm_attention_rules(self.R_i_i, self.R_t_t, self.R_t_i, cam_i_t,
                                                                  apply_normalization=self.normalize_self_attention,
                                                                  apply_self_in_rule_10=self.apply_self_in_rule_10)
        return R_i_t_addition, R_i_i_addition
    

    def handle_self_attention_lang(self, blk):
        cam = blk.attention.self.get_attention_map().detach()
        grad = blk.attention.self.get_attn_gradients().detach()
        # print(grad.shape, cam.shape)
        cam = avg_heads(cam, grad)
        # print(self.R_t_t[0])
        R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
        self.R_t_t += R_t_t_add
        self.R_t_i += R_t_i_add
        # print(f"R_t_t in lang self attn: {self.R_t_t[0]}")


    def handle_co_attn_lang(self, block):
        
        cam_t_i = block.crossattention.self.get_attention_map().detach()
        grad_t_i = block.crossattention.self.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        R_t_i_addition, R_t_t_addition = apply_mm_attention_rules(self.R_t_t, self.R_i_i, self.R_i_t, cam_t_i,
                                                                  apply_normalization=self.normalize_self_attention,
                                                                  apply_self_in_rule_10=self.apply_self_in_rule_10)
        # print(f"R_t_t_addition in lang co attn: {self.R_t_t[0]}")
        
        return R_t_i_addition, R_t_t_addition

    def generate_relevance_maps (self, text_tokens, image_tokens, device):
        # text self attention matrix
        # text_tokens -= 2
        # image_tokens -= 1 

        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

        # text
        blocks = self.model.text_transformer.encoder.layer
        for blk in blocks:
            self.handle_self_attention_lang(blk)


        count = 0
        for text_layer, image_layer in zip(self.model.cross_modal_text_layers, self.model.cross_modal_image_layers):
            self.handle_self_attention_image(image_layer)
            R_i_t_addition, R_i_i_addition = self.handle_co_attn_image(image_layer)
            self.R_i_t += R_i_t_addition
            self.R_i_i += R_i_i_addition

            self.handle_self_attention_lang(text_layer)
            R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(text_layer)
            self.R_t_i += R_t_i_addition
            self.R_t_t += R_t_t_addition

            # print(self.R_t_t[0])
            count += 1
            # if count == 1:
                # break
        self.R_t_t[0, 0] = 0
        self.R_i_i[0, 0] = 0 #baka
        self.R_t_i[0, 0] = 0
        self.R_i_t[0, 0] = 0

        # return self.R_i_t, self.R_t_i #baka
        # return self.R_t_t, self.R_t_i #baka

        return self.R_t_t, self.R_i_i #baka
    
    
    def gradcam(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam

    def generate_attn_gradcam (self, text_tokens, image_tokens, device):
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

        # for text_layer, image_layer in zip(self.model.cross_modal_text_layers, self.model.cross_modal_image_layers):
        text_layer = self.model.cross_modal_text_layers[-1]
        image_layer = self.model.cross_modal_image_layers[-1]

        cam_t_i = text_layer.crossattention.self.get_attention_map().detach()
        grad_t_i = text_layer.crossattention.self.get_attn_gradients().detach()
        cam_t_i = self.gradcam(cam_t_i, grad_t_i)
        self.R_t_i = cam_t_i

        cam = text_layer.attention.self.get_attention_map().detach()
        grad = text_layer.attention.self.get_attn_gradients().detach()
        self.R_t_t = self.gradcam(cam, grad)

        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i




    def generate_transformer_attr (self, text_tokens, image_tokens, device):

        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

        # text self attention
        blocks = self.model.text_transformer.encoder.layer
        for blk in blocks:
            cam = blk.attention.self.get_attention_map().detach()
            grad = blk.attention.self.get_attn_gradients().detach()
            cam = avg_heads(cam, grad)
            self.R_t_t += torch.matmul(cam, self.R_t_t)

        for text_layer, image_layer in zip(self.model.cross_modal_text_layers, self.model.cross_modal_image_layers):
            cam = text_layer.attention.self.get_attention_map().detach()
            grad = text_layer.attention.self.get_attn_gradients().detach()
            cam = avg_heads(cam, grad)
            self.R_t_t += torch.matmul(cam, self.R_t_t)

            cam = image_layer.attention.self.get_attention_map().detach()
            grad = image_layer.attention.self.get_attn_gradients().detach()
            cam = avg_heads(cam, grad)
            self.R_i_i += torch.matmul(cam, self.R_i_i)
        
        cam_t_i = text_layer.crossattention.self.get_attention_map().detach()
        # print(f"cam_t_i shape: {cam_t_i.shape}")
        grad_t_i = text_layer.crossattention.self.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        self.R_t_i = cam_t_i

        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_i_i
    

    def generate_rollout (self, text_tokens, image_tokens, device):

        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

        cams_text = []
        cams_image = []

        for blk in self.model.text_transformer.encoder.layer:
            cam = blk.attention.self.get_attention_map().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_text.append(cam)

        for text_layer in self.model.cross_modal_text_layers:
            cam = text_layer.attention.self.get_attention_map().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_text.append(cam)

        for image_layer in self.model.cross_modal_image_layers:
            cam = image_layer.attention.self.get_attention_map().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_image.append(cam)    

        # for text_layer, image_layer in zip(self.model.cross_modal_text_layers, self.model.cross_modal_image_layers):    
        self.R_t_t = compute_rollout_attention(copy.deepcopy(cams_text))
        self.R_i_i = compute_rollout_attention(cams_image)
        cam_t_i = self.model.cross_modal_text_layers[-1].crossattention.self.get_attention_map().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        self.R_t_i = torch.matmul(self.R_t_t.t(), torch.matmul(cam_t_i, self.R_i_i))


        self.R_t_t[0, 0] = 0
        # self.R_t_i[0, 0] = 0
        return self.R_t_t, self.R_i_i

    # def generate_lrp(self, text_tokens, image_tokens, device):
    #     self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
    #     # image self attention matrix
    #     self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
    #     # impact of images on text
    #     self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
    #     # impact of text on images
    #     self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

    #     cam_t_i = self.model.cross_modal_text_layers[-1].crossattention.self.get_attn_cam().detach()
    #     cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
    #     self.R_t_i = cam_t_i

    #     cam = self.model.cross_modal_text_layers[-1].attention.self.get_attn_cam().detach()
    #     cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
    #     self.R_t_t = cam

    #     self.R_t_t = (self.R_t_t - self.R_t_t.min()) / (self.R_t_t.max() - self.R_t_t.min())
    #     self.R_t_i = (self.R_t_i - self.R_t_i.min()) / (self.R_t_i.max() - self.R_t_i.min())
    #     # disregard the [CLS] token itself
    #     self.R_t_t[0, 0] = 0
    #     return self.R_t_t, self.R_t_i

    def generate_raw_attn (self, text_tokens, image_tokens, device):
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)      

        # cam_t_i = self.model.cross_modal_text_layers[-1].crossattention.self.get_attention_map().detach()
        # cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        # self.R_t_i = cam_t_i

        cam_t_i = self.model.cross_modal_image_layers[-1].attention.self.get_attention_map().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        self.R_t_i = cam_t_i

        cam = self.model.cross_modal_text_layers[-1].attention.self.get_attention_map().detach()
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
        self.R_t_t = cam  

        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i


