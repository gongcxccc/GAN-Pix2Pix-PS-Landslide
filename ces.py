import argparse
# 定义测试参数
parser = argparse.ArgumentParser()
# parser = argparse.ArgumentParser(description="Example with argument function")  # 添加了描述信息
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batch of generated images")
parser.add_argument("--n_samples", type=int, default=10, help="number of images to generate per class")
parser.add_argument("--weights_path", type=str, default="checkpoints/epoch41_best_LossG_0.2647.pth", help="path to generator weights")
parser.add_argument("--output_dir", type=str, default="generated_images", help="directory to save generated images")
opt = parser.parse_args()
print(opt)
args = vars(opt)
for k, v in sorted(args.items()):
    # print('%s: %s' % (str(k), str(v)))
    print(f'{str(k)}: {str(v)}')