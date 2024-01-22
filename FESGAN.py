import albumentations
import torch
import torch.nn as nn
from light_cnn import LightCNN_29Layers
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_channel, conv_filter):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channel, conv_filter, 4, 2)
        self.norm = nn.InstanceNorm2d(conv_filter)
        self.active = nn.LeakyReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.norm(x)
        x = self.active(x)
        return x

class tran_block(nn.Module):
    def __init__(self, in_channel, conv_filter):
        super(tran_block, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channel, conv_filter, 4, 2, 1)
        self.norm = nn.InstanceNorm2d(conv_filter)
        self.active = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.norm(x)
        x = self.active(x)
        return x

# class G_enc(nn.Module):
#     def __init__(self, in_channel, in_ch_mult=(3, 64, 128, 256, 512), out_ch_mult=(64, 128, 256, 512, 1024)):
#         super(G_enc, self).__init__()
#         self.inchannel = in_channel
#         self.in_ch = in_ch_mult
#         self.out_ch = out_ch_mult
#         self.enc = nn.ModuleList()
#         for i in range(len(self.in_ch)):
#             self.enc.append(conv_block(in_channel=self.in_ch[i], conv_filter=self.out_ch[i]))
#     def forward(self, input):
#         x = input
#         for i in range(len(self.in_ch)):
#             x = self.enc[i](x)
#         return x

# class G_enc(nn.Module):
#     def __init__(self, in_channel, in_ch_mult=(3, 64, 128, 256, 512), out_ch_mult=(64, 128, 256, 512, 1024)):
#         super(G_enc, self).__init__()
#         self.inchannel = in_channel
#         layer = [conv_block(3, 64),
#                  conv_block(64, 128),
#                  conv_block(128, 256),
#                  conv_block(256, 512),
#                  conv_block(512, 1024)]
#         self.enc = nn.Sequential(*layer)
#     def forward(self, input):
#         x = self.enc(input)
#         return x

'''
另一种定义方式
'''

class G_enc(nn.Module):
    def __init__(self, in_channel):
        super(G_enc, self).__init__()
        self.in_channel = in_channel
        self.block_0 = conv_block(self.in_channel, 64)
        self.block_1 = conv_block(64, 128)
        self.block_2 = conv_block(128, 256)
        self.block_3 = conv_block(256, 512)
        self.block_4 = conv_block(512, 1024)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 64)

    def forward(self, input):
        x = self.block_0(input)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = nn.Tanh()(x)
        return x

class G_dec(nn.Module):
    def __init__(self, in_channel, c):
        super(G_dec, self).__init__()
        self.in_channel = in_channel
        self.block_0 = tran_block(self.in_channel, 1024)
        self.block_1 = tran_block(1024, 512)
        self.block_2 = tran_block(512, 256)
        self.block_3 = tran_block(256, 128)
        self.block_4 = tran_block(128, 64)
        self.block_5 = tran_block(64, 32)
        self.top = nn.Sequential(*[nn.ConvTranspose2d(32, c, 4, 2, 1),
                                   nn.Tanh()])

    def forward(self, input):
        input = input.view(input.shape[0], input.shape[1], 1, 1)
        x = self.block_0(input)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.top(x)
        return x

class D_img(nn.Module):
    def __init__(self, in_channel, k):
        super(D_img, self).__init__()
        self.in_channel = in_channel
        self.layer = nn.Sequential(*[nn.Conv2d(self.in_channel, 64, 4, 2, 1),
                                     nn.InstanceNorm2d(64),
                                     # nn.LayerNorm([64, 64, 64]),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(64, 128, 4, 2, 1),
                                     nn.InstanceNorm2d(128),
                                     # nn.LayerNorm([128, 32, 32]),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(128, 256, 4, 2, 1),
                                     nn.InstanceNorm2d(256),
                                     # nn.LayerNorm([256, 16, 16]),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(256, 512, 4, 2, 1),
                                     nn.InstanceNorm2d(512),
                                     # nn.LayerNorm([512, 8, 8]),
                                     nn.LeakyReLU()])
        self.D_adv_img = nn.Conv2d(512, 1, 4, 2, 1)
        # self.D_adv_img = nn.Sequential(*[nn.Conv2d(512, 1024, 4, 2, 1),
        #                                  nn.Conv2d(1024, 1, 4, 1)])
        self.D_adv_cls = nn.Sequential(*[nn.Conv2d(512, 1024, 4, 2, 1),
                                         nn.Conv2d(1024, k, 4, 1)])
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 64)

    def forward(self, input):
        x = self.layer(input)
        adv_img = self.D_adv_img(x)
        # adv_img = adv_img.view(adv_img.shape[0], -1)
        adv_cla = self.D_adv_cls(x)
        return adv_img.squeeze(), torch.softmax(adv_cla.squeeze(), dim=1)
        # return adv_img.squeeze(), adv_cla.squeeze()

class D_z(nn.Module):
    def __init__(self):
        super(D_z, self).__init__()
        self.layer = nn.Sequential(*[nn.Linear(64, 64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 32),
                                     nn.LeakyReLU(),
                                     nn.Linear(32, 16),
                                     nn.LeakyReLU(),
                                     nn.Linear(16, 1),
                                     nn.Sigmoid()])
    def forward(self, input):
        x = self.layer(input)
        return x.squeeze()


class recog_r(nn.Module):
    def __init__(self, k):
        super(recog_r, self).__init__()
        self.light_cnn = LightCNN_29Layers().eval()
        self.light_cnn = torch.nn.DataParallel(self.light_cnn).cuda()
        self.light_cnn.load_state_dict(torch.load(r'./checkpoints/LightCNN_29Layers_checkpoint.pth.tar')['state_dict'])
        light_tmp = list(self.light_cnn.children())[0]
        self.feature = nn.Sequential(*[light_tmp.conv1,
                                       light_tmp.pool1,
                                       light_tmp.block1,
                                       light_tmp.group1,
                                       light_tmp.pool2,
                                       light_tmp.block2,
                                       light_tmp.group2,
                                       light_tmp.pool3])

        self.layer0 = nn.Sequential(*[nn.Conv2d(192, 256, 3, 2, 1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 512, 3, 2, 1)]).cuda()
        self.linear1 = nn.Sequential(*[nn.Linear(512, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, 512),
                                       nn.ReLU(),
                                       nn.Dropout(0.5)]).cuda()
        self.classifier = nn.Linear(512, k).cuda()

    def forward(self, input):
        with torch.no_grad():
            mid_feature = self.feature(input)
        x = self.layer0(mid_feature)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.classifier(x)
        return mid_feature, torch.softmax(x, dim=1)

class fse_gan(nn.Module):
    def __init__(self, inchannel, latent_dim, num_class):
        super(fse_gan, self).__init__()
        self.inchannel = inchannel
        self.latent_dim = latent_dim
        self.encoder = G_enc(self.inchannel)
        self.decoder = G_dec(in_channel=self.latent_dim, c=3)
        self.dis_img = D_img(in_channel=3, k=num_class)
        self.dis_z = D_z()
        self.classifier = recog_r(num_class)
        self.ou_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.light_cnn = LightCNN_29Layers().eval()
        self.light_cnn = torch.nn.DataParallel(self.light_cnn).cuda()
        self.light_cnn.load_state_dict(torch.load(r'./checkpoints/LightCNN_29Layers_checkpoint.pth.tar')['state_dict'])
        self.gray = albumentations.ToGray()

    def gradient_penalty(self, input_img, generate_img):
        # interpolate sample
        alpha = torch.rand(input_img.size(0), 1, 1, 1).to('cuda')
        inter_img = (alpha * input_img.data + (1 - alpha) * generate_img.data).requires_grad_(True)
        inter_img_prob, _ = self.dis_img(inter_img)

        # computer gradient penalty: x: inter_img, y: inter_img_prob
        # (L2_norm(dy/dx) - 1)**2
        dydx = torch.autograd.grad(outputs=inter_img_prob,
                                   inputs=inter_img,
                                   grad_outputs=torch.ones(inter_img_prob.size()).to('cuda'),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def forward(self, input, z, org_label, des_label):
        latent_vector = self.encoder(input)
        org_vector = torch.concat([latent_vector, org_label], dim=1)
        des_vector = torch.concat([latent_vector, des_label], dim=1)
        z_vector = torch.concat([z, org_label], dim=1)

        xrec = self.decoder(org_vector)
        des_exp = self.decoder(des_vector)
        fake_exp = self.decoder(z_vector)

        return xrec, des_exp, fake_exp, latent_vector

    def D_z_loss(self, z, latent_vector):
        d_z_dis_z = self.dis_z(z)
        d_z_dis_latent = self.dis_z(latent_vector.detach())
        d_z_dis_loss = -torch.mean(torch.log(d_z_dis_z))-torch.mean(torch.log(d_z_dis_latent))
        return d_z_dis_loss

    def D_img_loss(self, input, des_exp, xrec, org_label):
        input_adv_img, input_adv_cla = self.dis_img(input)
        # des_adv_img, des_adv_cla = self.dis_img(des_exp)
        des_adv_img, des_adv_cla = self.dis_img(des_exp.detach())
        D_img_adv_img_loss = -torch.mean(input_adv_img)+torch.mean(des_adv_img)
        D_img_adv_cla_real_loss = -torch.mean(org_label*torch.log(input_adv_cla))
        # D_img_adv_cla_real_loss = -torch.mean(torch.log(input_adv_cla))
        d_img_loss = D_img_adv_img_loss+D_img_adv_cla_real_loss
        loss_gp = self.gradient_penalty(input, xrec)
        return d_img_loss+loss_gp
        # return d_img_loss

    def Gen_loss(self, input, xrec, des_exp, latent, des_label):
        with torch.no_grad():
            input_gray = input[:, 0, :, :] * 0.299 + input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.114
            xrec_gray = xrec[:, 0, :, :] * 0.299 + xrec[:, 1, :, :] * 0.587 + xrec[:, 2, :, :] * 0.114
            input_id = self.light_cnn(torch.unsqueeze(input_gray, dim=1))[1]
            xrec_id = self.light_cnn(torch.unsqueeze(xrec_gray, dim=1))[1]
        id_loss = torch.mean(self.l1_loss(input_id, xrec_id))
        rec_loss = torch.mean(self.l1_loss(input, xrec))

        gen_d_z_latent_loss = -torch.mean(torch.log(self.dis_z(latent.detach())))

        des_adv_img, des_adv_cla = self.dis_img(des_exp.detach())
        gen_des_exp_loss = -torch.mean(des_adv_img)

        des_adv_cla_loss = -torch.mean(des_label*torch.log(des_adv_cla))
        # des_adv_cla_loss = -torch.mean(torch.log(des_adv_cla))

        return gen_des_exp_loss, gen_d_z_latent_loss, rec_loss, id_loss, des_adv_cla_loss

    def classifer_loss(self, input, fake_exp, real, org_label, des_label):
        des_label = F.one_hot(torch.argmax(des_label, dim=1), num_classes =6)
        input_gray = input[:, 0, :, :] * 0.299 + input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.114
        fake_exp_gray = fake_exp[:, 0, :, :] * 0.299 + fake_exp[:, 1, :, :] * 0.587 + fake_exp[:, 2, :, :] * 0.114
        real_gray = real[:, 0, :, :] * 0.299 + real[:, 1, :, :] * 0.587 + real[:, 2, :, :] * 0.114
        input_mid_feature, input_pred = self.classifier(torch.unsqueeze(input_gray, dim=1))
        fake_mid_feature, fake_pred = self.classifier(torch.unsqueeze(fake_exp_gray, dim=1))
        real_mid_feature, real_pred = self.classifier(torch.unsqueeze(real_gray, dim=1))

        fake_pred_loss = -torch.mean(torch.sum(des_label*torch.log(fake_pred), dim=1))

        recg_loss = torch.sum(-torch.mean(torch.sum(org_label*torch.log(input_pred), dim=1))+fake_pred_loss)

        dis_input_real = self.ou_loss(input_mid_feature, real_mid_feature)
        dis_input_fake = self.ou_loss(input_mid_feature.detach(), fake_mid_feature)     # RDBP
        intra_loss = dis_input_real+dis_input_fake

        return fake_pred_loss, recg_loss, intra_loss, input_pred, real_pred, fake_pred

# if __name__ == '__main__':
#     e = G_enc(in_channel=3)
#     d = G_dec(in_channel=64, c=3)
#     adv = D_img(in_channel=3, k=6)
#     z = D_z()
#     c = recog_r(k=6)
#     x = torch.randn((1, 3, 128, 128))
#     y = e(x)
#     yy = d(y)
#     a, b = adv(yy)
#     d = z(y)
#     yy = transforms.Grayscale()(yy)
#     f = c(yy.cuda())
#     print(y.shape, yy.shape)