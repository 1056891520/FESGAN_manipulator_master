import argparse
import glob
import random
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import albumentations
import torch.nn.functional as F
from torchvision import utils as vutils
from torch.utils.data import Dataset, DataLoader
from FESGAN import fse_gan, D_z, D_img
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(0)
random.seed(0)

cl_labels = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3,
    'Sadness': 4, 'Surprise': 5}

class ImagePaths(Dataset):
    def __init__(self, path, size=128):
        self.data_path = path
        self.size = size
        self.file_paths = glob.glob(self.data_path+'/*/*.jpeg')
        self.orig_labels = [cl_labels.get(image_path.split('\\')[-2]) for image_path in self.file_paths]
        self._length = len(self.file_paths)
        # self.desire_labels = np.random.randint(0, 6, (1, self._length))[0].astype(int)
        self.desire_labels = [cl_labels.get(image_path.split('\\')[-2]) for image_path in self.file_paths]
        random.shuffle(self.desire_labels)
        self.rescaler = albumentations.Resize(height=self.size, width=self.size)
        # # self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.norm = albumentations.Normalize(mean=0.5, std=0.5)
        self.preprocessor = albumentations.Compose([self.rescaler, self.norm])

    # def get_labels(self):
    #     return self.labels

    # def get_aufeat(self):
    #     return self.au_feat

    def __len__(self):
        return len(self.file_paths)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        # image = self.preprocessor(image)
        # image = (image/255.).astype(np.float32)
        # image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def get_real(self, orig_label, i):
        index = np.where(np.array(self.orig_labels) == orig_label)[0]
        index_tmp = np.delete(np.array(index), np.where(index==i))
        ind = random.choice(index_tmp)
        return ind


    def get_des_label(self, des_label):
        des_label = F.one_hot(torch.tensor(des_label), num_classes=6)
        des_label = (2*des_label-1)*torch.randn(1)
        return des_label

    def get_label(self, label):
        label = F.one_hot(torch.tensor(label), num_classes=6)
        return label

    def __getitem__(self, i):
        example = self.preprocess_image(self.file_paths[i])
        orig_label = self.get_label(self.orig_labels[i])
        desr_label = self.get_des_label(self.desire_labels[i])
        real = self.preprocess_image(self.file_paths[self.get_real(self.orig_labels[i], i)])
        noise = torch.rand(64)*2-1

        return example, noise, real, orig_label, desr_label


def load_data(path, args):
    train_data = ImagePaths(path, size=128)
    # train_loader = DataLoader(train_data, batch_size=args.batch_size,
    #                           num_workers=2, pin_memory=True, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    return train_loader


class Test_FESGAN:
    def __init__(self, opt):
        self.fesgan = fse_gan(opt.inchannel, opt.latent_dim, opt.num_class).to(opt.device)
        self.fesgan.load_state_dict(torch.load(r'./checkpoints/fesgan_oulu_140.pth'))
        self.test(opt)

    def test(self, args):
        test_dataset = load_data(args.test_path, args)
        acc, nums = 0, 0
        test_accuracy = 0
        with tqdm(range(len(test_dataset))) as pbar:
            with torch.no_grad():
                for i, (images, noise, real, org_label, des_label) in enumerate(tqdm(test_dataset)):
                    imgs = images.to(device=args.device)
                    real = real.to(device=args.device)
                    # noise = noise.to(device=args.device)
                    noise = torch.randn((args.batch_size, 64)).to(device=args.device)
                    org_label = org_label.to(device=args.device)
                    des_label = des_label.to(device=args.device)
                    xrec, des_exp, fake_exp, latent_vector = self.fesgan(imgs, noise, org_label, des_label)

                    _, recg_loss, intra_loss, input_pred, real_pred, fake_pred = \
                        self.fesgan.classifer_loss(imgs, fake_exp, real, org_label, des_label)
                    nums += org_label.size()[0] + org_label.size()[0] + des_label.size()[0]
                    acc += sum(input_pred.max(axis=1)[1] == org_label.max(axis=1)[1]).detach().cpu().numpy() \
                        + sum(real_pred.max(axis=1)[1] == org_label.max(axis=1)[1]).detach().cpu().numpy() \
                        + sum(fake_pred.max(axis=1)[1] == des_label.max(axis=1)[1]).detach().cpu().numpy()
                    test_accuracy = acc / nums
                    real_fake_images = torch.cat((imgs[:4], des_exp[:4], fake_exp[:4], xrec[:4],))
                    vutils.save_image(real_fake_images[:4], args.save_path + str(i)+'.jpg', nrow=4, normalize=True)
                pbar.set_postfix(recg_loss=np.round(recg_loss.cpu().detach().numpy().item(), 8),
                                 test_accuracy=np.round(test_accuracy, 8))
                pbar.update(1)
        print('test model:recg_loss:{}, test_acc:{}'.format(recg_loss, test_accuracy))

if __name__ == '__main__':
    def get_parser():
        parser = argparse.ArgumentParser(description="Test_FESGAN")
        parser.add_argument('--test_path', type=str,
                            default=r'F:\FER_dataset_clearned\Oulu_CASIA_VIS_Strong\Frontal face\test',
                            help='Path to traindata')
        parser.add_argument('--inchannel', type=int, default=3, help='input channel')
        parser.add_argument('--latent_dim', type=int, default=70, help='latent vector dimmmension')
        parser.add_argument('--num_class', type=int, default=6, help='number classes')
        parser.add_argument('--save_path', type=str, default=r'./gent_imgs/', help='Path to traindata')
        parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train (default: 50)')
        parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU')
        parser.add_argument('--device', default='cuda', help='device to use for training / testing')
        return parser
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    Test_FESGAN(opt)