import os
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max([topk])
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        #print(pred)
        #print(target)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in range(topk):
            correct_k = correct.view(-1).float().sum(0, keepdim=True)
            #print(correct[:k])
            res.append(correct_k.mul_(100.0 / batch_size))
        #print(res)
        return res


def label_map_fn(label):
    return 0 if label == 4 else 1

class RemapDataset_lsun(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.map = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 0, 12: 1, 13: 1, 14: 1, 15:1, 16:1, 17:1, 18: 1, 19: 1, 20:1, 21:1, 22:1, 23: 1, 24: 1}
        self.idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
        
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        orig_class = self.idx_to_class[label]
        label = self.map[label]
        
        return img, label, orig_class

class RemapDataset_imagenet(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        # self.map = {0: 1, 1: 1, 2: 1, 3: 0, 4: 1}
        self.map = {0: 1, 1: 0, 2: 1}
        self.idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        orig_class = self.idx_to_class[label]
        label = self.map[label]
        
        return img, label, orig_class

class RemapDataset_celebahq(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.map = {0: 1, 1: 1, 2: 1, 3: 0, 4: 1}
        self.idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
        
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        orig_class = self.idx_to_class[label]
        label = self.map[label]
        
        return img, label, orig_class

def normalize_per_image(x):
    B = x.shape[0]
    out = []
    for i in range(B):
        img_i = x[i]
        min_val, max_val = img_i.min(), img_i.max()
        img_norm = (img_i - min_val) / (max_val - min_val + 1e-8)
        out.append(img_norm)
    return torch.stack(out)


def global_min_max_norm(tensor_list):
    combined = torch.cat(tensor_list, dim=0)  # 2N x C x H x W
    min_val = combined.min()
    max_val = combined.max()
    normed = [(x - min_val) / (max_val - min_val + 1e-8) for x in tensor_list]
    return torch.cat(normed, dim=0)

def write_log(logfile_path, text):
    with open(logfile_path, "a") as f:
        f.write(text + "\n")

