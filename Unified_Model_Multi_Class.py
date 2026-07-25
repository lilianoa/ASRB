import sys
import torch
import torch.nn as nn
import random
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_sg
from utils_logging import get_logger

from dataset import MVTecDataset, RealIADDataset, VisADataset, BTADDataset
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset

from models import vit_encoder
from models.uad import ASRB
from models.vision_transformer import Mlp, Block, Queries_Block, Attention, Block_ReLU, FeatureJitter

warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    # Fixing the Random Seed
    setup_seed(1)

    # Data Preparation
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    if args.dataset == 'MVTec-AD':
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):
            train_path = os.path.join(args.data_path, item, 'train')
            test_path = os.path.join(args.data_path, item)

            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(sample[0], i) for sample in train_data.samples]
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)
        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif args.dataset == 'VisA':
        train_data_list = []
        train_path = args.data_path
        for i, item in enumerate(args.item_list):
            train_data = VisADataset(root=train_path, transform=data_transform, gt_transform=gt_transform, phase="train", class_names=item)
            train_data_list.append(train_data)
        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_data_list = []
        test_path = args.data_path
        for i, item in enumerate(args.item_list):
            test_data = VisADataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test", class_names=item)
            test_data_list.append(test_data)
    elif args.dataset == 'Real-IAD':
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):
            train_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                        gt_transform=gt_transform,
                                        phase='train')
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            test_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                       gt_transform=gt_transform,
                                       phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)

        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                       drop_last=True)
    elif args.dataset == 'BTAD':
        train_data_list = []
        train_path = args.data_path
        for i, item in enumerate(args.item_list):
            train_data = BTADDataset(root=train_path, transform=data_transform, gt_transform=gt_transform, phase="train", class_names=item)
            train_data_list.append(train_data)
        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_data_list = []
        test_path = args.data_path
        for i, item in enumerate(args.item_list):
            test_data = BTADDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test", class_names=item)
            test_data_list.append(test_data)

    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(args.encoder)

    if 'small' in args.encoder or 'vits' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder or 'vitb' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder or 'vitl' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    # Model Preparation
    Q_Decoder = []

    # Queries
    queries = nn.ParameterList(
                    [nn.Parameter(torch.randn(args.queries_num, embed_dim))
                     for _ in range(1)])

    # Queries bottleneck
    Q_Former = nn.ModuleList([Queries_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
                              for _ in range(1)])

    # Noise
    # noise_adaptor = FeatureJitter(base_std=0.05, norm_power=1, num_channels==embed_dim)
    # noise_adaptor = Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)

    # Decoder
    for layer in range(len(target_layers)):
        blk = Block_ReLU(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-8))
        Q_Decoder.append(blk)
    Q_Decoder = nn.ModuleList(Q_Decoder)

    model = ASRB(encoder=encoder,
                 bottleneck=Q_Former,
                 decoder=Q_Decoder,
                 target_layers=target_layers,
                 remove_class_token=True,
                 fuse_layer=fuse_layer,
                 queries=queries)
    model = model.to(device)

    if args.phase == 'train':
        # Model Initialization
        trainable = nn.ModuleList([Q_Former, Q_Decoder, queries])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=args.total_iters, warmup_iters=100)
        print_fn('train image number:{}'.format(len(train_data)))

        # Train
        it = 0
        for epoch in range(int(np.ceil(args.total_iters / len(train_dataloader)))):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                en, de = model(img)
                loss = global_cosine_hm_sg(en, de, alpha=3.0)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()

                it += 1
                if it % 100 == 0:
                    print_fn('iter [{}/{}], loss:{:.4f}'.format(it, args.total_iters, np.mean(loss_list)))
                    loss_list = []

                if it % 5000 == 0 and it != args.total_iters:
                    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, f'model_{it}.pth'))

                if it == args.total_iters:
                    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, f'model_{it}.pth'))
                    auroc_sp_list, ap_sp_list, auroc_px_list, f1_px_list, aupro_px_list = [], [], [], [], []

                    for item, test_data in zip(args.item_list, test_data_list):
                        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                                      num_workers=4)
                        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.005, resize_mask=args.crop_size, save_img=args.save_img, save_path=args.save_path)
                        auroc_sp, ap_sp, auroc_px, f1_px, aupro_px = results
                        auroc_sp_list.append(auroc_sp)
                        ap_sp_list.append(ap_sp)
                        auroc_px_list.append(auroc_px)
                        f1_px_list.append(f1_px)
                        aupro_px_list.append(aupro_px)
                        print_fn(
                            '{}: I-AUROC:{:.4f}, I-AP:{:.4f}, P-AUROC:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                                item, auroc_sp, ap_sp, auroc_px, f1_px, aupro_px))

                    print_fn('Mean: I-AUROC:{:.4f}, I-AP:{:.4f}, P-AUROC:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            np.mean(auroc_sp_list), np.mean(ap_sp_list),
                            np.mean(auroc_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                    model.train()
                    break

    elif args.phase == 'test':
        # Test
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, 'model_10000.pth')), strict=True)
        model.eval()

        auroc_sp_list, ap_sp_list, auroc_px_list, f1_px_list, aupro_px_list = [], [], [], [], []

        for item, test_data in zip(args.item_list, test_data_list):
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=4)
            results = evaluation_batch(model, test_dataloader, device, max_ratio=0.005, resize_mask=args.crop_size, save_img=args.save_img, save_path=args.save_path)
            auroc_sp, ap_sp, auroc_px, f1_px, aupro_px = results
            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            auroc_px_list.append(auroc_px)
            f1_px_list.append(f1_px)
            aupro_px_list.append(aupro_px)
            print_fn(
                '{}: I-AUROC:{:.4f}, I-AP:{:.4f}, P-AUROC:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item, auroc_sp, ap_sp, auroc_px, f1_px, aupro_px))
        print_fn('Mean: I-AUROC:{:.4f}, I-AP:{:.4f}, P-AUROC:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            np.mean(auroc_sp_list), np.mean(ap_sp_list),
            np.mean(auroc_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD')  # 'MVTec-AD' or 'VisA' or 'Real-IAD' or 'BTAD'
    parser.add_argument('--data_path', type=str, default=r'./data/mvtec')  # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--save_name', type=str, default='Multi-Class')
    parser.add_argument('--save_img', type=bool, default=False, help="if save anomaly maps, True")
    parser.add_argument('--save_path', type=str, default='./save_imgs/mvtec', help="anomaly maps results path")

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--queries_num', type=int, default=8)

    # training info
    parser.add_argument('--total_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')

    args = parser.parse_args()
    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_num={args.queries_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # category info
    if args.dataset == 'MVTec-AD':
        args.item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    elif args.dataset == 'VisA':
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset == 'Real-IAD':
        args.item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    elif args.dataset == 'BTAD':
        args.item_list = ['01', '02', '03']
    main(args)
