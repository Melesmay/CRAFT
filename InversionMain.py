import os
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
import lpips
import numpy
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImageVariationPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from diffusers.models import AutoencoderKL
import torch.nn.functional as F

from util import AverageMeter, accuracy, write_log
from util import RemapDataset_imagenet, RemapDataset_celebahq, RemapDataset_lsun, normalize_per_image, global_min_max_norm
from network import LPIPSClassifierMLP, random_init, ClassCenterBuffer, ContrastiveReconstructionLoss
from torch.cuda.amp import autocast, GradScaler
from InversionTrainer import DiffusionInversionModule
from newInversionTrain import DiffusionReconstructionModule
from diffusers.utils import logging as diffusers_logging
from collections import defaultdict

from sklearn.metrics import average_precision_score

import swanlab
from PIL import Image, ImageFile

import datetime
diffusers_logging.disable_progress_bar() 

swanlab.init(project = 'diffdect_base', requirements_collect=False)

lpips_loss_fn = lpips.LPIPS(net='vgg', spatial=True).to("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
l1_loss_fn = nn.L1Loss()

def train(lpips_clas, CELoss, ConsLoss, opt, loader, inversion_module, center_buffer, freeze_cls = True, warmup_iters = 500, reloss = 'CR', if_perb = True, if_fine = True, num_finetune_steps = 15):
    lpips_clas.train()
    # scaler = GradScaler()
    losses = AverageMeter()
    top = AverageMeter()

    batch_bar = tqdm(loader, dynamic_ncols=True)
    
    accumu_steps = 4  
    
    loss_sum, correct_sum, n_samples = 0.0, 0, 0  
    torch.cuda.empty_cache()    
    for i, data in enumerate(batch_bar):
        img, label = data[0].cuda(), data[1].cuda()
        img = img.to(dtype=next(inversion_module.pipe.vae.encoder.parameters()).dtype) 

        if if_fine:
            if freeze_cls and i == 0:
                for p in lpips_clas.parameters():
                    p.requires_grad_(False)
            if freeze_cls and i == warmup_iters:
                for p in lpips_clas.parameters():
                    p.requires_grad_(True)
                freeze_cls = False

        latents, re_img = inversion_module.process_image(img, label, mode='train', if_perb = if_perb, num_finetune_steps=num_finetune_steps) 
        center_buffer.update(latents.detach(), label) 

        if i % 100 == 0: 

            img_vis = (img[:4] + 1) / 2
            re_img_vis = (re_img[:4] + 1) / 2
                
            merged_vis = global_min_max_norm([img_vis, re_img_vis])
                
            save_image(merged_vis, f'save/debug_step_{label[0]}{label[1]}{label[2]}{label[3]}.png')

        # LPIPS distence
        dist = inversion_module.lpips_loss_fn(img, re_img).mean([1,2,3])
        dist_norm = torch.log(dist + 1e-4)

        if reloss == 'CR':
            contrastive_loss = ConsLoss(img, re_img, label)
        elif reloss == 'L2':
            contrastive_loss = F.mse_loss(re_img, img)
        elif reloss == 'L1':
            contrastive_loss = l1_loss_fn(re_img, img)
        elif reloss == 'CR+L2':
            contrastive_loss = 0.5 * F.mse_loss(re_img, img) + 0.5 * ConsLoss(img, re_img, label)

            
        if i < warmup_iters and reloss is not None:
            loss = contrastive_loss
        else:
            
            pred = lpips_clas(dist_norm)    # [B,2]
            ce_loss = CELoss(pred, label) 
            if reloss is not None:
                loss = ce_loss + contrastive_loss
            else:
                loss = ce_loss

        if not torch.isfinite(loss):
            print("Non-finite loss detected, skipping step")
            print(ce_loss, contrastive_loss)
            opt.zero_grad() 
            continue
    
            # loss_sum += loss.item() * img.size(0) 
            # _, pred_top1 = pred.max(1)
            # correct_sum += (pred_top1 == label).sum().item()

        loss_sum += loss.item() * img.size(0)

        if i >= warmup_iters:
            _, pred_top1 = pred.max(1)
            correct_sum += (pred_top1 == label).sum().item()
        else:
            pred_top1 = label 
            correct_sum += 0 

            
        n_samples   += img.size(0)
            
        current_epoch_loss = loss_sum / n_samples
        current_epoch_acc  = correct_sum / n_samples
            
        loss = loss / accumu_steps 
    
        # scaler.scale(loss).backward() 
        loss.backward()
    
        if (i + 1) % accumu_steps == 0: 
            if i < warmup_iters:
                ce_loss = 0.0
            else:
                ce_loss = ce_loss.item()

            # scaler.step(opt) 
            opt.step()
            opt.zero_grad() 
            # scaler.update() 
            batch_bar.set_postfix(dict(loss=f'{current_epoch_loss:.5f}', ce_loss = f'{ce_loss:.5f}', cons_loss = f'{contrastive_loss.item():.5f}', acc=f'{current_epoch_acc:.5f}'))
            batch_bar.refresh() 
        
        # if i ==3000:
        # return losses.avg, t op.avg, center_buffer
        
        losses.update(loss.item(), img.size(0)) 
        
        # acc = accuracy(pred, label, topk=(1))
        # top.update(acc[0], img.size(0))        

        if i >= warmup_iters:
            acc = accuracy(pred, label, topk=(1))
            top.update(acc[0], img.size(0))
        else:
            acc = [torch.tensor(0.0, device=label.device).view(1)]
            top.update(acc[0], 0)  # 不更新
        

        swanlab.log({"loss":current_epoch_loss, "acc":current_epoch_acc})

    if (i + 1) % accumu_steps != 0:
        # scaler.step(opt)
        opt.step()
        # scaler.update()
        opt.zero_grad()

    return losses.avg, top.avg, center_buffer

@torch.no_grad()
def eval(lpips_clas, CELoss, ConsLoss, loader, inversion_module, center_buffer, subset = 'celeba', reloss = 'CR', num_finetune_steps=15):
    lpips_clas.eval()

    losses = AverageMeter()
    top    = AverageMeter()

    correct_dict = defaultdict(int)
    total_dict   = defaultdict(int)

    all_preds  = []
    all_labels = []
    turn = 0

    for img, label, orig_name in tqdm(loader, dynamic_ncols=True):
        img    = img.cuda()
        label  = label.cuda()
        img    = img.to(dtype=next(inversion_module.pipe.vae.encoder.parameters()).dtype)

        # with autocast():
        latents, re_img = inversion_module.process_image(img, label, mode='eval', num_finetune_steps=num_finetune_steps)

        

        img_vis = (img[:4] + 1) / 2
        re_img_vis = (re_img[:4] + 1) / 2
                
        merged_vis = global_min_max_norm([img_vis, re_img_vis])
        
        os.makedirs(f'save/{subset}', exist_ok=True) 
        save_image(merged_vis, f'save/{subset}/debug_step_{turn}_{label[0]}{label[1]}{label[2]}{label[3]}.png')
        turn += 1

        # binary classification
        dist      = inversion_module.lpips_loss_fn(img, re_img).mean([1, 2, 3])
        dist_norm = torch.log(dist + 1e-4)
        logits    = lpips_clas(dist_norm)                 # [B,2]
        pred      = logits.argmax(dim=1)                  # [B]

        # loss function
        if reloss == 'CR':
            contrastive_loss = ConsLoss(img, re_img, label)
        elif reloss == 'L2':
            contrastive_loss = F.mse_loss(re_img, img)
        elif reloss == 'L1':
            contrastive_loss = l1_loss_fn(re_img, img)
        elif reloss == 'CR+L2':
            contrastive_loss = 0.5 * F.mse_loss(re_img, img) + 0.5 * ConsLoss(img, re_img, label)
            
        loss = CELoss(logits, label) + contrastive_loss
        # loss  = CELoss(logits, label) + ConsLoss(img, re_img, label)

        losses.update(loss.item(), img.size(0))
        acc_batch = (pred == label).float().mean()
        top.update(acc_batch, img.size(0))

        # AP
        all_preds.append(logits.detach().cpu())
        all_labels.append(label.detach().cpu())

        for p, gt, cls in zip(pred.cpu(), label.cpu(), orig_name):
            total_dict[cls]   += 1
            correct_dict[cls] += int(p == gt)

    all_preds   = torch.cat(all_preds)            # [N,2]
    all_labels  = torch.cat(all_labels)           # [N]
    probs_fake  = torch.softmax(all_preds, dim=1)[:, 1].numpy()
    ap_all      = average_precision_score(all_labels.numpy(), probs_fake)

    per_cls_acc = {}
    print("\nPer‑class accuracy:")
    for cls in sorted(total_dict.keys()):
        acc_cls = correct_dict[cls] / total_dict[cls]
        per_cls_acc[cls] = acc_cls
        print(f"  {cls:15s}: {acc_cls:.4f}")

    print(f"\nAverage‑Precision (overall): {ap_all:.4f}")
    print(f"Top‑1 accuracy (overall)    : {top.avg:.4f}")

    return losses.avg, top.avg, ap_all, per_cls_acc

def main(root, reloss = 'CR', if_perb = True, if_fine = True, num_finetune_steps = 15, enc = True, unet = True, dec = True):
    # ... (rest of your main block remains the same)
    eval_freq = 1
    save_freq = 1

    strtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_floder = 'checkpoints/{}'.format(strtime)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train_set = ImageFolder(
        root=root,
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]))
    eval_set_lsun = ImageFolder(
        root='../DiffusionForensics/images/test/lsun_bedroom',
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]))
    eval_set_lsun = RemapDataset_lsun(eval_set_lsun)
    
    eval_set_imagenet = ImageFolder(
        root='../DiffusionForensics/images/test/imagenet',
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]))
    eval_set_imagenet = RemapDataset_imagenet(eval_set_imagenet)

    eval_set_celebahq = ImageFolder(
        root='../DiffusionForensics/images/test/celebahq',
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]))
    eval_set_celebahq = RemapDataset_celebahq(eval_set_celebahq)
    

    train_loader = DataLoader(train_set, batch_size =8, num_workers = 20, drop_last = True, shuffle = True, pin_memory=True)
    eval_loader_lsun = DataLoader(eval_set_lsun, batch_size = 8, num_workers = 20, drop_last = True, shuffle = False)
    eval_loader_imagenet = DataLoader(eval_set_imagenet, batch_size = 8, num_workers = 20, drop_last = True, shuffle = False)
    eval_loader_celebahq = DataLoader(eval_set_celebahq, batch_size = 8, num_workers = 20, drop_last = True, shuffle = False)
    
    lpips_clas = LPIPSClassifierMLP().cuda()
    # linear = LinearClassifier().cuda() 
    random_init(lpips_clas)
    CELoss = torch.nn.CrossEntropyLoss().cuda()
    ConsLoss = ContrastiveReconstructionLoss(base_loss='mse', margin=0.3, delta = 0.3)

    
    pipe = StableDiffusionImageVariationPipeline.from_pretrained("stable-diffusion-image-variations", use_safetensors=False).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True) 
    pipe.image_processor.do_rescale = False
    if hasattr(pipe, 'feature_extractor'):
        pipe.feature_extractor.do_rescale = False
    '''
    pipe.unet.float()               
    pipe.image_encoder.float()     
    '''
    # pipe.vae.decoder.float()     
    # pipe.vae.post_quant_conv.float() 

    center_buffer = ClassCenterBuffer(latent_dim=4 * 16 * 16, num_classes=2, device=device)
    lpips_loss_fn.cuda().eval() 

    # inversion_module = DiffusionInversionModule(pipe, lpips_loss_fn, class_center_buffer=center_buffer, device=device)
    inversion_module = DiffusionReconstructionModule(pipe, lpips_loss_fn, class_center_buffer=center_buffer, device=device)

    
    group_cls = {"params": lpips_clas.parameters(),
                "lr": 0.01, "weight_decay": 1e-4}
    if enc:
        group_enc = {"params": inversion_module.pipe.image_encoder.parameters(),
                    "lr": 5e-6, "weight_decay": 0}
    if unet:
        group_unet = {"params": inversion_module.pipe.unet.parameters(),
                    "lr": 5e-6, "weight_decay": 0}
    if dec:
        group_dec = {"params": inversion_module.pipe.vae.decoder.parameters(),
                    "lr": 1e-5, "weight_decay": 0}
    if if_fine:
        fine_list = [group_cls]
        if enc:
            fine_list.append(group_enc)
        if unet:
            fine_list.append(group_unet)
        if dec:
            fine_list.append(group_dec)
        optimizer = torch.optim.AdamW(fine_list)
    else:
        optimizer = torch.optim.AdamW([group_cls])

    for e in range(1):
        print(f"Starting Epoch {e}")
        freeze_cls = True
        warmup_iters = 200
        if e > 0:
            freeze_cls = False
            warmup_iters = 0
        loss, acc, center_buffer = train(lpips_clas, CELoss, ConsLoss, optimizer, train_loader, inversion_module, center_buffer, freeze_cls = freeze_cls, warmup_iters = warmup_iters, if_fine = if_fine, reloss = reloss, num_finetune_steps = num_finetune_steps)
        
        if e % eval_freq == 0:
            eval_loss1, eval_acc1, ap1, per_cls_acc1 = eval(lpips_clas, CELoss, ConsLoss, eval_loader_celebahq, inversion_module, center_buffer, subset = 'celeba', reloss = reloss, num_finetune_steps = num_finetune_steps)
            swanlab.log({"celebahq_eval_loss": eval_loss1, "celebahq_eval_acc": eval_acc1.item(), "celebahq_ap": ap1})
            
            eval_loss2, eval_acc2, ap2, per_cls_acc2 = eval(lpips_clas, CELoss, ConsLoss, eval_loader_imagenet, inversion_module, center_buffer, subset = 'imagenet', reloss = reloss, num_finetune_steps = num_finetune_steps)
            swanlab.log({"imagenet_eval_loss": eval_loss2, "imagenet_eval_acc": eval_acc2.item(), "imagenet_ap": ap2})

            eval_loss3, eval_acc3, ap3, per_cls_acc3 = eval(lpips_clas, CELoss, ConsLoss, eval_loader_lsun, inversion_module, center_buffer, subset = 'lsun', reloss = reloss, num_finetune_steps = num_finetune_steps)
            swanlab.log({"lsun_eval_loss": eval_loss3, "lsun_eval_acc": eval_acc3.item(), "lsun_ap": ap3})
            
            print(f"Epoch {e} Eval Loss: {eval_loss1:.5f}, Eval Acc: {eval_acc1.item():.5f}, {per_cls_acc1}")
            print(f"Epoch {e} Eval Loss: {eval_loss2:.5f}, Eval Acc: {eval_acc2.item():.5f}, {per_cls_acc2}")
            print(f"Epoch {e} Eval Loss: {eval_loss3:.5f}, Eval Acc: {eval_acc3.item():.5f}, {per_cls_acc3}")
            
            log_str = (
                f"Epoch {e} | training set {root}\n"
                f"celebahq | Eval Acc: {eval_acc1.item():.5f} | AP: {ap1} | {per_cls_acc1}\n"
                f"imagenet | Eval Acc: {eval_acc2.item():.5f} | AP: {ap2} | {per_cls_acc2}\n"
                f"lsun | Eval Acc: {eval_acc3.item():.5f} | AP: {ap3} | {per_cls_acc3}\n"
            )
            log_file = os.path.join("checkpoints", f"log.txt")
            write_log(log_file, log_str)
        
        if e % save_freq == 0:
            if not os.path.isdir(save_floder):
                os.makedirs(save_floder)
                        
            torch.save(lpips_clas.state_dict(), os.path.join(save_floder, f'lpips_clas_{e}.pth'))
            # torch.save(linear.state_dict(), os.path.join(save_floder, f'linear_{e}.pth'))

            center_save = os.path.join(save_floder, 'center.pth')
            torch.save(center_buffer, center_save)

            inversion_save = os.path.join(save_floder, 'inversion.pth')
            torch.save(inversion_module.pipe.unet.state_dict(), inversion_save)
        
    # loss, top1, ap = eval(lpips_clas, CELoss, ConsLoss, eval_loader_imagenet, inversion_module, center_buffer)
    # print(f"Final Test Acc: {top1}")

if __name__ == '__main__':
    
    # log
    log_file = os.path.join("checkpoints", f"log.txt")
    os.makedirs("checkpoints", exist_ok=True)


    # main(root='../DiffusionForensics/images/train/imagenet', num_finetune_steps = 15, enc = False, dec = False)
    # main(root='../DiffusionForensics/images/train/imagenet', num_finetune_steps = 15, unet = False, dec = False)
    # main(root='../DiffusionForensics/images/train/imagenet', num_finetune_steps = 15, enc = False, unet = False)
    # main(root='../DiffusionForensics/images/train/imagenet', num_finetune_steps = 15, enc = False)
    # main(root='../DiffusionForensics/images/train/imagenet', num_finetune_steps = 15, unet = False)
    # main(root='../DiffusionForensics/images/train/imagenet', num_finetune_steps = 15, dec = False)

    # main(root='../DiffusionForensics/images/train/lsun_bedroom', num_finetune_steps = 25)
    
    # main(root='../DiffusionForensics/images/train/lsun_bedroom', num_finetune_steps = 15)
    
    # main(root='../DiffusionForensics/images/train/imagenet', num_finetune_steps = 1)

    
    main(root='../DiffusionForensics/images/train/celebahq', num_finetune_steps = 5)
    main(root='../DiffusionForensics/images/train/celebahq', num_finetune_steps = 10)
    main(root='../DiffusionForensics/images/train/celebahq', num_finetune_steps = 15)
    main(root='../DiffusionForensics/images/train/celebahq', num_finetune_steps = 20)
    main(root='../DiffusionForensics/images/train/celebahq', num_finetune_steps = 25)
    main(root='../DiffusionForensics/images/train/celebahq', num_finetune_steps = 30)                                                                        
    main(root='../DiffusionForensics/images/train/celebahq', num_finetune_steps = 50)
    
    # main(root='../DiffusionForensics/images/train/imagenet', reloss = 'CR+L2', if_perb = False)
    # main(root='../DiffusionForensics/images/train/imagenet', reloss = 'L2', if_perb = False)
    # main(root='../DiffusionForensics/images/train/imagenet', reloss = 'L1', if_perb = False)
    # main(root='../DiffusionForensics/images/train/imagenet', if_perb = False)
    # main(root='../DiffusionForensics/images/train/imagenet', if_perb = False, if_fine = False)
    
    
    # main(root='../DiffusionForensics/images/train/lsun_bedroom', num_finetune_steps = 5)
    # main(root='../DiffusionForensics/images/train/lsun_bedroom', num_finetune_steps = 10)
    # main(root='../DiffusionForensics/images/train/lsun_bedroom', num_finetune_steps = 25)

    

    