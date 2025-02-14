import torch
import clip

class metircs:
    def __init__(self, ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # clip score
        model, preprocess = clip.load('ViT-B/32', device=self.device)
        self.model = model
        self.preprocess = preprocess

    def clip_scores(self, prompt, images):

        text_tokens = clip.tokenize(prompt).to(self.device)
        images = self.preprocess(images.convert("RGB")).unsqueeze(0).to(self.device)  # 预处理图像并添加批次维度，移动到设备上

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化>图像特征
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化文本特征

            similarity_scores = (image_features @ text_features.T).squeeze()  # 计算相似度得分（点积）

        return similarity_scores.mean().item()