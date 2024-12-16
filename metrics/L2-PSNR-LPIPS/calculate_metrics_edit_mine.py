import os
from PIL import Image
import torch

# 导入之前定义的计算指标的函数
from metric_util import calculate_metrics  # 将这里替换为你实际的模块名，如果在同一个文件中可省略这行

# 假设这里定义了编辑提示的映射关系，你可以根据实际情况填充具体内容，键为原图prompt，值为编辑prompt
mapping = {'A deserted highway near snow mountains under a partly cloudy sky':
               'A red car on a deserted highway near snow mountains under a partly cloudy sky',
           'A tranquil lake surrounded by dense forests and mountains under a clear blue sky':
               'A canoe on the tranquil lake surrounded by dense forests and mountains under a clear blue sky',
           'A plant near a lamp on a table with a wooden clock on the wall':
               'A book on the table near a plant and a lamp with a wooden clock on the wall',
           'A bench under a tree with fallen leaves on the ground and a historic building in the background':
               'A doll on the bench under a tree with fallen leaves on the ground and a historic building in the background',
           'An empty room with hardwood floors, white walls, and a large window showing trees outside':
               'A blue sofa in the center of an room with hardwood floors, white walls, and a large window showing trees outside',
           'An orange cat sitting attentively on a red-painted curb against a white background':
               'A butterfly near an orange cat sitting attentively on a red-painted curb against a white background',
           'A stack of cookies on a wooden board with a gray background':
               'A glass of milk next to a stack of cookies on a wooden board with a gray background',
           'A camel walking on a desert': 'A flamingo standing on a camel walking on a desert',
           'A llama toy standing next to a potted plant in a cozy room':
               'A floor lamp standing next to a potted plant in a cozy room',
           'A wolf stands in a forest':
               'A rock sits in a forest',
           'A cluster of coconuts attached to a palm tree':
               'A cluster of lanterns on a palm tree',
           'A person holding a handbag':
               'A person holding a box',
           'A car with surfboards on top':
               'A car with bushes on top',
           'A road sign stands beside a rural highway':
               'A mailbox stands beside the rural highway',
           'A white mug filled with yellow dandelions on a table':
               'A white mug filled with cute puppies on the table',
           'A red cabin on a rocky outcrop by the sea':
               'A blue dome on a rocky outcrop by the sea',
           'A wooden bench in front of a lush green hill with trees':
               'A lush green hill with trees',
           'A bicycle with a basket parked against a rustic wall with vines':
               'A rustic wall with vines',
           'A cup of coffee on an unmade bed':
               'An unmade bed',
           'A person sitting at the seaside with rocks':
               'A seaside with rocks',
           'A cup of coffee and a pair of glasses near an open book':
               'An open book',
           'A person with an umbrella walking across a busy city street':
               'A busy city street',
           'A wooden house with a dock on a mountain lake':
               'A mountain lake',
           'Four women in long dresses walk down a hallway':
               'A hallway',
           'A man stands on mountains':
               'A man stands at a seaside',
           'A person sits on a desert':
               'A person sits in a bedroom',
           'A person stands on a snowy mountain range':
               'A person stands in a tropical rainforest',
           'A woman leans on a wooden post at a field':
               'A woman leans on a wooden post at a street',
           'A person sits on a paddleboard on a lake':
               'A person sits on a paddleboard on a desert',
           'A white fox sitting on a snowland':
               'A white fox sitting on a beach',
           'Several Rhinoceroses on an African plain':
               'Several Rhinoceroses on Mars',
           'A silver car parked at a city street':
               'A silver car parked at a dense jungle',
           'A church on top of a mountain surrounded by trees':
               'A watercolor painting of a church on top of a mountain surrounded by trees',
           'A light house sitting on a cliff next to the ocean':
               'A Van Gogh style painting of a light house sitting on a cliff next to the ocean',
           'A cityscape with skyscrapers and a park':
               'A cyberpunk style of a cityscape with skyscrapers and a park',
           'A countryside field with houses and trees':
               'A Japanese anime style of a countryside field with houses and trees',
           'A winding road in the middle of a large green landscape':
               'A winding road in the middle of a large green landscape in winter',
           'A grassy field with yellow flowers near mountains':
               'A grassy field with yellow flowers near mountains in autumn',
           'A woman standing on a beach next to the ocean':
               'A woman standing on a beach next to the ocean at sunset',
           'An orange car parked in a parking lot in front of tall buildings':
               'An orange car parked in a parking lot in front of tall buildings at night',
           'A kitten sitting on a sofa': 'A metal kitten sitting on a sofa',
           'A white horse running in a field':
               'A statue of a horse running in a field',
           'A chocolate cake on a plate with a fork':
               'A plastic chocolate cake on a plate with a fork',
           'A person carrying a leather handbag with a fluffy keychain':
               'A person carrying a glass handbag with a fluffy keychain',
           'A rose in bloom': 'A wooden rose in bloom', 'A black church':
               'An ice church', 'A white horse standing in the grass':
               'A white horse running in the grass',
           'A polar bear standing beside the sea':
               'A polar bear raising its hand',
           'A dog standing on the ground':
               'A dog jumping on the ground',
           'A bird on the tree': 'A bird is flying over the tree'}



def calculate_average_metrics(folder_path):
    lpips_sum = 0
    count = 0

    for source_prompt, target_prompt in mapping.items():
        source_image_path = os.path.join(folder_path, source_prompt + ".jpg")
        target_image_path = os.path.join(folder_path, target_prompt + ".jpg")

        try:
            source_image = Image.open(source_image_path)
            target_image = Image.open(target_image_path)
        except:
            print(f"无法打开图片 {source_image_path} 或 {target_image_path}，跳过本次计算")
            continue

        device = 'cuda'
        metrics = calculate_metrics(source_image, target_image, device)
        lpips_sum += metrics["lpips"]
        count += 1

    if count == 0:
        print("没有成功计算任何一组图片的LPIPS指标，请检查图片是否正确及命名格式和mapping设置")
        return None

    average_lpips = lpips_sum / count

    return {"lpips": average_lpips}



# 使用示例，将这里替换为实际存放图片的文件夹路径
folder_path = "/home/lyw/project/ReNoise-Inversion/rf-solver_ablation_result/new_output_for_edit_lpips/output_3_solver_edit_3_20_3_v_injection"
average_metrics = calculate_average_metrics(folder_path)
print(average_metrics)