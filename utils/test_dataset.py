from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

class dehaze_test_dataset(Dataset):
    def __init__(self, test_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test_hazy=[]

        self.root_hazy=os.path.join(test_dir, 'LQ_padding/')  # LQ/  LQ_padding/
        for i in os.listdir(self.root_hazy):
           self.list_test_hazy.append(i)
        #self.root_hazy = os.path.join(test_dir)


        self.file_len = len(self.list_test_hazy)


    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.root_hazy + self.list_test_hazy[index]).convert('RGB')
        #hazy = hazy.crop((512,0,1024,768))
        #print(hazy.size)
        hazy = self.transform(hazy)
        

        name=self.list_test_hazy[index]
        
        return hazy, name
    def __len__(self):
        return self.file_len





