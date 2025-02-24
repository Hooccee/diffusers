from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

default_rootpath = '/data/chx/EditEval_v1/Dataset'
default_csvpath = '/data/chx/EditEval_v1/Dataset/editing_prompts_collection.xlsx'
class EditEval_v1_dataset(Dataset):
    def __init__(self, transform=None,
                    csvpath=default_csvpath,
                    rootpath=default_rootpath):
        super(EditEval_v1_dataset, self).__init__()
        self.transform = transform
        self.csvpath = csvpath
        self.rootpath = rootpath
        df = pd.read_excel(self.csvpath)
        
        # 读取数据列
        self.edit_class = df['Edit Class'].tolist()
        self.source_prompt = df['Source Prompt'].tolist()
        self.target_prompt = df['Target Prompt'].tolist()
        
        self.samples = []
        class_counter = {}  # 用于跟踪每个类别的独立计数器
        current_class = None  # 当前处理的类别

        for item, s, t in zip(self.edit_class, self.source_prompt, self.target_prompt):
            # 检测到新类别时（非空值）
            if not pd.isna(item):
                # 统一处理类别名称
                formatted_class = item.strip().lower().replace(' ', '_')
                formatted_class = formatted_class.replace('object_removal', 'object_removel')
                
                # 更新当前类别并重置计数器
                current_class = formatted_class
                if current_class not in class_counter:
                    class_counter[current_class] = 1

            # 生成路径
            if current_class is None:
                raise ValueError("CSV文件中缺少初始类别定义")
            
            path = f"{current_class}/{class_counter[current_class]}.jpg"
            self.samples.append([path, s, t])
            # 递增当前类别的计数器
            class_counter[current_class] += 1
    
    def __getitem__(self, index):
        impath, source_prompt, target_prompt = self.samples[index]
        impath = default_rootpath + '/' + impath
        if not os.path.exists(impath):
            impath = impath.replace('jpg', 'jpeg')
        img = Image.open(impath)
        img = self.transform(img)
        return img, source_prompt, target_prompt

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.samples)