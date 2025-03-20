import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

def calculate_ap(gt_mask, pred_mask, threshold=128):
    """
    Calculate the accuracy and IoU of segmentation considering both foreground and background
    :param gt_mask: Ground truth mask
    :param pred_mask: Predicted mask
    :param threshold: Threshold for predicted mask, default 30
    :return: Tuple of (pixel_accuracy, IoU)
    """
    # If GT and prediction are from the same image (compare file paths), return perfect scores
    if np.array_equal(gt_mask, pred_mask):
        return 1.0, 1.0, 1.0
    
    # 二值化预测掩码
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    
    # 将GT掩码转换为二值掩码(所有非零值视为前景)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # 计算总像素数
    total_pixels = gt_mask.size
    
    # 计算像素级准确率 (正确预测的像素数 / 总像素数)
    correct_pixels = np.sum(gt_binary == pred_binary)
    pixel_accuracy = correct_pixels / total_pixels
    
    # 计算前景的IoU (Intersection over Union)
    intersection_fg = np.sum(np.logical_and(gt_binary == 1, pred_binary == 1))
    union_fg = np.sum(np.logical_or(gt_binary == 1, pred_binary == 1))
    iou_foreground = intersection_fg / union_fg if union_fg > 0 else 0.0
    
    # 计算背景的IoU
    intersection_bg = np.sum(np.logical_and(gt_binary == 0, pred_binary == 0))
    union_bg = np.sum(np.logical_or(gt_binary == 0, pred_binary == 0))
    iou_background = intersection_bg / union_bg if union_bg > 0 else 0.0
    
    # 计算平均IoU (背景和前景的IoU平均值)
    mean_iou = (iou_foreground + iou_background) / 2
    
    return pixel_accuracy, mean_iou, iou_foreground

def visualize_pairs(img_dir='train/img', label_dir='train/label', infer_dir=None):
    """
    Visualize images, label masks, and inference results if available
    :param img_dir: Image directory
    :param label_dir: Label directory
    :param infer_dir: Inference result directory (optional)
    """
    # Get all image files
    img_files = sorted(Path(img_dir).glob('*.jpg'))
    
    # Store metrics for all images if inference directory exists
    if infer_dir:
        all_accuracies = []
        all_ious = []
        all_fg_ious = []
    else:
        all_accuracies = all_ious = all_fg_ious = None
    
    # Adjust figure size based on number of subplots
    n_plots = 4 if infer_dir else 3
    plt.figure(figsize=(5*n_plots, 5))
    
    # Create interactive display
    current_idx = [0]
    
    def show_pair(idx):
        plt.clf()
        
        # Read image and corresponding mask
        img_path = img_files[idx]
        mask_path = Path(label_dir) / f"{img_path.stem}.png"
        
        # Read image and mask
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Calculate subplot positions
        n_cols = 4 if infer_dir else 3
        
        # Show original image
        plt.subplot(1, n_cols, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Show label mask with colormap to distinguish different classes
        plt.subplot(1, n_cols, 2)
        # 使用自定义colormap显示不同类别的标签
        plt.imshow(mask, cmap='viridis')  # viridis: 0=黑色, 1=紫色, 2=绿色, 3=黄色
        plt.title('Label Mask (Classes 0-3)')
        plt.axis('off')
        
        # Create overlay of original image and label with different colors for different classes
        overlay_label = img.copy()
        mask_rgb = np.zeros_like(img)
        
        # 对每个类别使用不同的颜色
        mask_rgb[mask == 1] = [255, 0, 0]    # 类别1显示为红色
        mask_rgb[mask == 2] = [0, 255, 0]    # 类别2显示为绿色
        mask_rgb[mask == 3] = [0, 0, 255]    # 类别3显示为蓝色
        
        # 如果有255值的标签(旧格式)，也将其显示为红色
        if np.any(mask == 255):
            mask_rgb[mask == 255] = [255, 0, 0]
            
        overlay_label = cv2.addWeighted(overlay_label, 0.7, mask_rgb, 0.3, 0)
        
        plt.subplot(1, n_cols, 3)
        plt.imshow(overlay_label)
        plt.title('Image + Label Overlay')
        plt.axis('off')
        
        # If inference directory exists, show inference results
        if infer_dir:
            infer_path = Path(infer_dir) / f"{img_path.stem}.png"
            infer = cv2.imread(str(infer_path), cv2.IMREAD_GRAYSCALE)
            
            # Calculate metrics
            pixel_acc, mean_iou, fg_iou = calculate_ap(mask, infer)
            
            # Create overlay with inference result
            overlay_infer = img.copy()
            infer_binary = (infer > 30).astype(np.uint8) * 255
            infer_rgb = cv2.cvtColor(infer_binary, cv2.COLOR_GRAY2RGB)
            infer_rgb[infer_binary > 0] = [255, 128, 0]  # 预测结果显示为橙色
            overlay_infer = cv2.addWeighted(overlay_infer, 0.7, infer_rgb, 0.3, 0)
            
            plt.subplot(1, n_cols, 4)
            plt.imshow(overlay_infer)
            plt.title('Image + Inference Overlay')
            plt.axis('off')
            
            # Update metrics lists
            if len(all_accuracies) <= idx:
                all_accuracies.append(pixel_acc)
                all_ious.append(mean_iou)
                all_fg_ious.append(fg_iou)
            
            # Calculate mean metrics
            mean_acc = np.mean(all_accuracies) if all_accuracies else 0.0
            mean_iou_val = np.mean(all_ious) if all_ious else 0.0
            mean_fg_iou = np.mean(all_fg_ious) if all_fg_ious else 0.0
            
            # 计算每个类别的像素数量
            unique_labels, counts = np.unique(mask, return_counts=True)
            class_counts = {label: count for label, count in zip(unique_labels, counts)}
            class_info = ", ".join([f"Class {label}: {count} pixels" for label, count in class_counts.items()])
            
            plt.suptitle(f'Image {idx + 1}/{len(img_files)}: {img_path.name}\n'
                        f'Pixel Acc: {pixel_acc:.4f}, Mean IoU: {mean_iou:.4f}, FG IoU: {fg_iou:.4f}\n'
                        f'Avg Pixel Acc: {mean_acc:.4f}, Avg Mean IoU: {mean_iou_val:.4f}, Avg FG IoU: {mean_fg_iou:.4f}\n'
                        f'{class_info}')
        else:
            # 计算每个类别的像素数量
            unique_labels, counts = np.unique(mask, return_counts=True)
            class_counts = {label: count for label, count in zip(unique_labels, counts)}
            class_info = ", ".join([f"Class {label}: {count} pixels" for label, count in class_counts.items()])
            
            plt.suptitle(f'Image {idx + 1}/{len(img_files)}: {img_path.name}\n{class_info}')
        
        plt.tight_layout()
        plt.draw()
    
    def on_key(event):
        if event.key == 'right' and current_idx[0] < len(img_files) - 1:
            current_idx[0] += 1
            show_pair(current_idx[0])
        elif event.key == 'left' and current_idx[0] > 0:
            current_idx[0] -= 1
            show_pair(current_idx[0])
        elif event.key == 'q':
            if infer_dir:
                save_metrics_results(all_accuracies, all_ious, all_fg_ious, img_files)
            plt.close()
    
    def save_metrics_results(accuracies, ious, fg_ious, img_files):
        """Save evaluation metrics to file"""
        mean_acc = np.mean(accuracies)
        mean_iou = np.mean(ious)
        mean_fg_iou = np.mean(fg_ious)
        
        with open('metrics_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"Dataset Evaluation Metrics:\n")
            f.write(f"Mean Pixel Accuracy: {mean_acc:.4f}\n")
            f.write(f"Mean IoU (Background+Foreground): {mean_iou:.4f}\n")
            f.write(f"Mean Foreground IoU: {mean_fg_iou:.4f}\n\n")
            f.write("Individual Image Metrics:\n")
            f.write("Image Name, Pixel Accuracy, Mean IoU, Foreground IoU\n")
            for img_file, acc, iou, fg_iou in zip(img_files, accuracies, ious, fg_ious):
                f.write(f"{img_file.name}, {acc:.4f}, {iou:.4f}, {fg_iou:.4f}\n")
        
        print(f"\nResults saved to metrics_results.txt")
        print(f"Dataset Mean Pixel Accuracy: {mean_acc:.4f}")
        print(f"Dataset Mean IoU: {mean_iou:.4f}")
        print(f"Dataset Mean Foreground IoU: {mean_fg_iou:.4f}")
    
    # Display first image
    show_pair(0)
    
    # Bind keyboard events
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    
    # Display usage instructions
    print("Instructions:")
    print("- Press Right Arrow to view next image")
    print("- Press Left Arrow to view previous image")
    print("- Press 'q' to" + (" save results and" if infer_dir else "") + " exit viewer")
    
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # For visualization only (no inference):
    # visualize_pairs('train/img', 'train/label')
    # For visualization with inference:
    visualize_pairs('daizhuang/images', 'daizhuang/masks', 'daizhuang/infer') 