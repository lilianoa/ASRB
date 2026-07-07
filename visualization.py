import cv2
import os
import numpy as np

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def show_cam_on_image(img, anomaly_map):
    anomaly_map = (anomaly_map * 255).astype(np.uint8)
    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def visualizer(pathes, anomaly_map, img, save_path, cls_name, gt):
    os.makedirs(save_path, exist_ok=True)
    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        im = img[idx].permute(1, 2, 0).cpu().numpy()
        im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        im = (im * 255).astype('uint8')
        im = im[:, :, ::-1]
        heatmap = normalize(anomaly_map[idx])
        vis = show_cam_on_image(im, heatmap)
        save_vis = os.path.join(save_path, 'imgs', cls_name[idx], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)

        cv2.imwrite(os.path.join(save_vis, filename.split('.')[0] + '_img.png'), im)
        cv2.imwrite(os.path.join(save_vis, filename.split('.')[0] + '_map.png'), vis)
        mask = (gt[idx][0].numpy() * 255).astype('uint8')
        cv2.imwrite(os.path.join(save_vis, filename.split('.')[0] + '_gt.png'), mask)


