import torch
import clip
from PIL import Image
import os


# 加载CLIP模型和预处理函数
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# 替换为实际存放图片的文件夹路径
image_folder_path = "/home/lyw/project/ReNoise-Inversion/rf-solver_ablation_result/new_output_for_edit_clip/output_3_solver_edit_3_20_3_v_injection"

count = 0
score = 0
for file in os.listdir(image_folder_path):
    count += 1
    images = []
    text_descriptions = []
    file_path = os.path.join(image_folder_path, file)
    image = Image.open(file_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)  # 预处理图像并添加批次维度，移动到设备上
    images.append(image)
    # 提取文件名（不含扩展名）作为对应的文本描述（prompt）
    prompt = file.split(".")[0]
    text_descriptions.append(prompt)

    images = torch.cat(images, dim=0)

    text_tokens = clip.tokenize(text_descriptions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化图像特征
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化文本特征

        similarity_scores = (image_features @ text_features.T).squeeze()  # 计算相似度得分（点积）

    score += similarity_scores.mean().item()


# 计算平均CLIP分数
average_score = score / count

print("平均CLIP分数:", average_score)