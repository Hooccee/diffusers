from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

default_rootpath = '/teg_amai_yun/share_2938211/bobmin/cqw/Awesome-Diffusion-Model-Based-Image-Editing-Methods/EditEval_v1/Dataset'
default_csvpath = '/teg_amai_yun/share_2938211/bobmin/cqw/Awesome-Diffusion-Model-Based-Image-Editing-Methods/EditEval_v1/Dataset/editing_prompts_collection.xlsx'
class EditEval_v1_dataset(Dataset):
    def __init__(self, transform=None,
                    csvpath=default_csvpath,
                    rootpath=default_rootpath):
        super(EditEval_v1_dataset, self).__init__()
        self.transform = transform
        self.csvpath = csvpath
        self.rootpath = rootpath
        df = pd.read_excel(self.csvpath)
        self.object_addition = df['Object Addition'].tolist()
        self.source_prompt = df['Source Prompt'].tolist()
        self.target_prompt = df['Target Prompt'].tolist()
        self.samples = []
        cur_class = 'Object Addition'
        id = 1
        for item, s, t in zip(self.object_addition, self.source_prompt, self.target_prompt):
            if not pd.isna(item):
                cur_class = item
                id = 1
            cur_class = cur_class.replace('object_removal', 'object_removel')
            path = f"{cur_class.lower().replace(' ','_')}/{id}.jpg"
            path = path.replace('object_removal', 'object_removel')
            id += 1
            self.samples.append([path, s, t])
    
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