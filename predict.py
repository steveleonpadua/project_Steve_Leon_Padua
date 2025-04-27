import os
import torch
import matplotlib.pyplot as plt
import random
import dataset
import numpy as np
import torchvision.transforms as tt
import config
import model

device = config.device
classes = config.classes
nn_model = model.CNN_Model()
path = os.path.join(os.getcwd(), 'checkpoints', 'final_weights.pth')
nn_model.load_state_dict(torch.load(path, map_location= device))
nn_model = nn_model.to(device)
nn_model.eval()

def classify_image(pred_path):
    ds = []
    transform = tt.Compose([tt.ToTensor(),tt.Resize((config.resize_x, config.resize_y))])
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    for i in os.listdir(pred_path):
        if i.lower().endswith(image_extensions):
            try:
                img = plt.imread(os.path.join(pred_path, i))
                img = np.array(img,copy=True)
                ds.append(transform(img))
            except Exception as e:
                print(f"Error processing {i}: {e}")
    
    random.shuffle(ds)
    pred_dl = dataset.CustomDataLoader(ds, device='cuda', has_labels=False)
    
    preds = []
    for img in pred_dl:
        out = nn_model(img)
        _, pred = torch.max(out, dim=1)
        preds.append(pred.to('cpu').numpy())
        break
        
    classes_pred = [classes[x] for x in pred]
    
    for img in pred_dl:
        for i in range(0, 10):
            plt.imshow(img[i].to('cpu').permute(1, 2, 0))
            plt.title(classes_pred[i])
            plt.show()
        break
        
    
    return None