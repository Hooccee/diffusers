import pandas as pd
import os
import subprocess
import argparse

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'Intel'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# 定义批量编辑图片的函数
def batch_flux_editing(excel_path, output_dir, feature_path, script_path):
    # 读取Excel文件
    data = pd.read_excel(excel_path)
    
    # 创建一个字典用于跟踪每种编辑类别的行序号
    class_row_counters = {}

    # 跟踪已完成任务的文件
    completed_tasks_file = os.path.join(output_dir, "completed_tasks.txt")
    if os.path.exists(completed_tasks_file):
        with open(completed_tasks_file, "r") as f:
            completed_tasks = set(line.strip() for line in f)
    else:
        completed_tasks = set()
    
    # 遍历Excel表中的每一行
    for index, row in data.iterrows():
        # 提取信息
        edit_class = row['Edit Class']
        source_prompt = row['Source Prompt']
        target_prompt = row['Target Prompt']
        
        # 如果是新出现的编辑类别，初始化其计数
        if edit_class not in class_row_counters:
            class_row_counters[edit_class] = 1
        else:
            class_row_counters[edit_class] += 1
        
        # 根据编辑类别和类内的行序号生成图片路径
        class_row_number = class_row_counters[edit_class]
        source_img_path = f"/data/chx/EditEval_v1/Dataset/{edit_class}/{class_row_number}.jpg"
        save_dir = os.path.join(output_dir, f"{edit_class}_output_{class_row_number}")
        
        # 创建唯一标识符用于记录任务状态
        task_id = f"{edit_class}_{class_row_number}"
        
        # 跳过已完成的任务
        if task_id in completed_tasks:
            print(f"跳过已完成的任务: {task_id}")
            continue
        
        # 检查并创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 设置其他所需参数
        model_path = "/data/chx/FLUX.1-dev"
        guidance_scale = 2
        num_steps = 30
        inject = 0
        dtype = "bfloat16"
        #offload = True

        print(f"正在处理编辑类别: {edit_class} 的第 {class_row_number} 张图片 - 源提示词: '{source_prompt}', 目标提示词: '{target_prompt}'")

        # 构建命令
        command = [
            "python", script_path,
            "--model_path", model_path,
            "--image_path", source_img_path,
            "--output_dir", save_dir,
            "--use_inversed_latents",
            "--guidance_scale", str(guidance_scale),
            "--num_steps", str(num_steps),
            "--shift",
            "--source_prompt", source_prompt,
            "--target_prompt", source_prompt,
            "--dtype", dtype,
            "--feature_path", feature_path,
            "--inject", str(inject)
        ]
        
        # 使用 subprocess 运行外部命令
        try:
            subprocess.run(command, check=True)
            # 记录已完成任务
            with open(completed_tasks_file, "a") as f:
                f.write(f"{task_id}\n")
            print(f"任务完成: {task_id}")
        except subprocess.CalledProcessError as e:
            print(f"任务失败: {task_id}, 错误: {e}")
            break

if __name__ == "__main__":
    # 设置参数
    excel_path = "/data/chx/EditEval_v1/Dataset/editing_prompts_collection.xlsx"
    output_dir = "/data/chx/EditEval_v1/output_RF-Solver-Edit-diffusers_prompt_reco_i0_g2"
    feature_path = "feature_output"  # 可根据需要修改
    script_path = "RF-Solver-Edit.py"  # 编辑脚本的路径

    # 执行批量处理
    batch_flux_editing(excel_path, output_dir, feature_path, script_path)