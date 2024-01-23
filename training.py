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
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(0)
random.seed(0)

# oulu
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


class Train_FESGAN:
    def __init__(self, opt):
        self.fesgan = fse_gan(opt.inchannel, opt.latent_dim, opt.num_class).to(opt.device)
        # self.D_z = D_z().to(opt.device)
        # self.D_img = D_img(opt.inchannel, opt.num_class).to(opt.device)
        self.opt_gen, self.opt_dis_img, self.opt_dis_z, self.opt_recg = self.configure_optimizers(opt)
        # self.train(opt)

    def configure_optimizers(self, opt):
        lr = opt.learning_rate
        opt_gen = torch.optim.Adam(list(self.fesgan.encoder.parameters())+
                                   list(self.fesgan.decoder.parameters()),
                                   lr=lr, betas=(0.5, 0.999))
        opt_dis_img = torch.optim.Adam(self.fesgan.dis_img.parameters(),
                                       lr=lr, betas=(0.5, 0.999))
        opt_dis_z = torch.optim.Adam(self.fesgan.dis_z.parameters(),
                                     lr=lr, betas=(0.5, 0.999))
        opt_recg = torch.optim.Adam(self.fesgan.classifier.parameters(),
                                    lr=lr, betas=(0.5, 0.999))
        return opt_gen, opt_dis_img, opt_dis_z, opt_recg

    def set_requires_grad(self, parameters, requires_grad=False):
        if not isinstance(parameters, list):
            parameters = [parameters]
        for param in parameters:
            if param is not None:
                param.requires_grad = requires_grad

    def train(self, args):
        train_dataset = load_data(args.train_path, args)
        log = {'loss_d_z': [], 'loss_d_img': [], 'loss_gen': [], 'loss_reg': [], 'train_acc': []}
        best_acc = 0.2
        train_accuracy = 0
        for epoch in range(args.epochs):
            loss_D_z = 0
            loss_D_img = 0
            loss_gen = 0
            loss_reg = 0
            acc, nums = 0, 0
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (images, noise, real, org_label, des_label) in enumerate(tqdm(train_dataset)):
                    imgs = images.to(device=args.device)
                    real = real.to(device=args.device)
                    noise = noise.to(device=args.device)
                    # noise = torch.randn((args.batch_size, 64)).to(device=args.device)
                    org_label = org_label.to(device=args.device)
                    des_label = des_label.to(device=args.device)
                    xrec, des_exp, fake_exp, latent_vector = self.fesgan(imgs, noise, org_label, des_label)

                    '''
                    优化 D_z
                    '''
                    d_z_dis_loss = self.fesgan.D_z_loss(noise, latent_vector)
                    self.set_requires_grad(self.fesgan.dis_z, True)
                    self.opt_dis_z.zero_grad()
                    # d_z_dis_loss.backward(retain_graph=True)
                    d_z_dis_loss.backward()
                    self.opt_dis_z.step()
                    loss_D_z += d_z_dis_loss.item()

                    '''
                    优化 D_img
                    '''
                    d_img_loss = self.fesgan.D_img_loss(imgs, des_exp, xrec, org_label)
                    self.set_requires_grad(self.fesgan.dis_img, True)
                    self.opt_dis_img.zero_grad()
                    # d_img_loss.backward(retain_graph=True)
                    d_img_loss.backward()
                    self.opt_dis_img.step()
                    loss_D_img += d_img_loss.item()
                    # pbar.set_postfix(d_z_dis_loss=np.round(d_z_dis_loss.cpu().detach().numpy().item(), 8),
                    #                  d_img_loss=np.round(d_img_loss.cpu().detach().numpy().item(), 8))
                    # pbar.update(1)

                    '''
                    优化 classifier
                    '''
                    if i % 4 == 0:
                        gen_des_exp_loss, gen_d_z_latent_loss, rec_loss, id_loss, des_adv_cla_loss = \
                            self.fesgan.Gen_loss(imgs, xrec, des_exp, latent_vector, des_label)
                        if epoch < args.p_epoch:
                            gen_loss = gen_des_exp_loss+gen_d_z_latent_loss+10*rec_loss+5*id_loss+des_adv_cla_loss

                            pbar.set_postfix(d_z_dis_loss=np.round(d_z_dis_loss.cpu().detach().numpy().item(), 8),
                                             d_img_loss=np.round(d_img_loss.cpu().detach().numpy().item(), 8),
                                             gen_loss=np.round(gen_loss.cpu().detach().numpy().item(), 8))
                            pbar.update(1)
                        else:
                            _, recg_loss, intra_loss, input_pred, real_pred, fake_pred = \
                                self.fesgan.classifer_loss(imgs, fake_exp, real, org_label, des_label)
                            classifer_loss = 0.001 * recg_loss + intra_loss

                            nums += org_label.size()[0] + org_label.size()[0] + des_label.size()[0]
                            acc += sum(input_pred.max(axis=1)[1] == org_label.max(axis=1)[1]).detach().cpu().numpy() \
                                + sum(real_pred.max(axis=1)[1] == org_label.max(axis=1)[1]).detach().cpu().numpy() \
                                + sum(fake_pred.max(axis=1)[1] == des_label.max(axis=1)[1]).detach().cpu().numpy()
                            train_accuracy = acc / nums
                            # print('{}/{}:train_acc:{}'.format(epoch + 1, args.epochs,  train_accuracy))
                            self.set_requires_grad(self.fesgan.classifier, True)
                            self.opt_recg.zero_grad()
                            classifer_loss.backward()
                            # classifer_loss.backward(retain_graph=True)
                            self.opt_recg.step()
                            loss_reg += classifer_loss.item()

                            '''
                            引入分类器的合成假图像获得损失更新生成器
                            '''
                            fake_exp_gray = fake_exp[:, 0, :, :] * 0.299 + fake_exp[:, 1, :, :] * 0.587 + fake_exp[:, 2, :, :] * 0.114
                            fake_mid_feature, fake_pred = self.fesgan.classifier(torch.unsqueeze(fake_exp_gray, dim=1))
                            fake_pred_loss = -torch.mean(torch.sum(des_label * torch.log(fake_pred), dim=1))
                            gen_loss = gen_des_exp_loss+gen_d_z_latent_loss+10*rec_loss+5*id_loss+des_adv_cla_loss+fake_pred_loss
                            pbar.set_postfix(d_z_dis_loss=np.round(d_z_dis_loss.cpu().detach().numpy().item(), 8),
                                             d_img_loss=np.round(d_img_loss.cpu().detach().numpy().item(), 8),
                                             gen_loss=np.round(gen_loss.cpu().detach().numpy().item(), 8),
                                             classifer_loss=np.round(classifer_loss.cpu().detach().numpy().item(), 8))
                            pbar.update(1)

                        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)  # 更改为梯度
                        self.opt_gen.zero_grad()
                        gen_loss.backward()
                        self.opt_gen.step()
                        loss_gen += gen_loss.item()

                print('{}/{}: loss_d_z:{}, loss_d_img:{}, loss_gen:{}, loss_reg:{}, train_acc:{}'.format(
                    epoch+1, args.epochs, loss_D_z/len(train_dataset), loss_D_img/len(train_dataset), loss_gen/len(train_dataset), loss_reg/len(train_dataset), train_accuracy))
                log.get('loss_d_z').append(loss_D_z/len(train_dataset))
                log.get('loss_d_img').append(loss_D_img/len(train_dataset))
                log.get('loss_gen').append(loss_gen/len(train_dataset))
                log.get('loss_reg').append(loss_reg / len(train_dataset))
                log.get('train_acc').append(train_accuracy)
            if epoch+1 > args.p_epoch and train_accuracy > best_acc:
                for i in glob.glob(os.path.join(r"./checkpoints/", '*.pth')):
                    os.remove(i)
                torch.save(self.fesgan.state_dict(), os.path.join(r"./checkpoints/", f"fesgan_oulu_{epoch+1}.pth"))
                best_acc = train_accuracy
        return log

def show_loss_acc(history, root_path, name1, name2, name3,name4, name5):
    loss_d_z = history.get('loss_d_z')
    loss_d_img = history.get('loss_d_img')
    loss_gen = history.get('loss_gen')
    loss_reg = history.get('loss_reg')
    train_acc = history.get('train_acc')

    # 绘图
    plt.figure(figsize=(8, 8))
    plt.rc('font', family='Times New Roman')
    plt.plot(loss_d_z, '-*', label='loss_d_z')
    plt.legend(loc='upper right')
    plt.ylabel('loss_d_z')
    plt.savefig(root_path+name1, dpi=300, bbox_inches='tight')

    plt.figure(figsize=(8, 8))
    plt.rc('font', family='Times New Roman')
    plt.plot(loss_d_img, '-*', label='loss_d_img')
    plt.legend(loc='upper right')
    plt.ylabel('loss_d_img')
    plt.savefig(root_path+name2, dpi=300, bbox_inches='tight')

    plt.figure(figsize=(8, 8))
    plt.rc('font', family='Times New Roman')
    plt.plot(loss_gen, '-*', label='loss_gen')
    plt.legend(loc='upper right')
    plt.ylabel('loss_gen')
    plt.savefig(root_path+name3, dpi=300, bbox_inches='tight')

    plt.figure(figsize=(8, 8))
    plt.rc('font', family='Times New Roman')
    plt.plot(loss_reg, '-*', label='loss_reg')
    plt.legend(loc='upper right')
    plt.ylabel('loss_reg')
    plt.savefig(root_path+name4, dpi=300, bbox_inches='tight')

    plt.figure(figsize=(8, 8))
    plt.rc('font', family='Times New Roman')
    plt.plot(train_acc, '-*', label='train_acc')
    plt.legend(loc='lower right')
    plt.ylabel('train_acc')
    plt.savefig(root_path+name5, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    def get_parser():
        parser = argparse.ArgumentParser(description="Train_FESGAN")
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
        parser.add_argument('--inchannel', type=int, default=3, help='input channel')
        parser.add_argument('--latent_dim', type=int, default=70, help='latent vector dimension')
        parser.add_argument('--p_epoch', type=int, default=80, help='mid_epoch')
        parser.add_argument('--num_class', type=int, default=6, help='number classes')
        parser.add_argument('--train_path', type=str, default= r'./Oulu_CASIA_VIS_Strong/train', help='Path to traindata')
        parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train (default: 200)')
        parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
        parser.add_argument('--device', default='cuda', help='device to use for training / testing')
        parser.add_argument('--start_epoch', type=int, default=0, help='(default: 0)')
        return parser
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    # log = Train_FESGAN(opt)
    log = Train_FESGAN(opt).train(opt)
    show_loss_acc(log, r'./history/', 'loss_d_z.jpg', 'loss_d_img.jpg', 'loss_gen.jpg', 'loss_reg.jpg', 'train_acc.jpg')
