import os
from PIL import Image
import torch



# 导入之前定义的计算指标的函数
from metric_util import calculate_metrics  # 将这里替换为你实际的模块名，如果在同一个文件中可省略这行


def calculate_average_metrics(folder_path):
    l2_sum = 0
    psnr_sum = 0
    lpips_sum = 0
    count = 0

    for file in os.listdir(folder_path):
        if "reconstructed" in file:
            original_file = file.replace(" reconstructed", "")
            original_image_path = os.path.join(folder_path, original_file)
            reconstruction_image_path = os.path.join(folder_path, file)
        else:
            original_image_path = os.path.join(folder_path, file)
            reconstruction_file = file.split('.')[0] + " reconstructed." + file.split('.')[1]
            reconstruction_image_path = os.path.join(folder_path, reconstruction_file)

        try:
            original_image = Image.open(original_image_path)
            reconstruction_image = Image.open(reconstruction_image_path)
        except:
            print(f"无法打开图片 {original_image_path} 或 {reconstruction_image_path}，跳过本次计算")
            continue

        device = 'cuda'
        metrics = calculate_metrics(original_image, reconstruction_image, device)
        l2_sum += metrics["l2"]
        psnr_sum += metrics["psnr"]
        lpips_sum += metrics["lpips"]
        count += 1

    if count == 0:
        print("没有成功计算任何一组图片指标，请检查图片是否正确及命名格式")
        return None

    average_l2 = l2_sum / count
    average_psnr = psnr_sum / count
    average_lpips = lpips_sum / count

    return {"l2": average_l2, "psnr": average_psnr, "lpips": average_lpips}


# 使用示例，将这里替换为实际存放图片的文件夹路径
folder_path = "/home/lyw/project/ReNoise-Inversion/rf-solver_ablation_result/new_output_for_edit_clip/output_3_solver_inversion_3_20_3_v_injection"
average_metrics = calculate_average_metrics(folder_path)
print(average_metrics)
