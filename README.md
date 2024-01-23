# [Joint Deep Learning of Facial Expression Synthesis and Recognition](https://ieeexplore.ieee.org/document/8943107) 论文复现
Facial Expression Recognition
## CN：
1. Light_CNN 相关代码和权值：https://github.com/AlfredXiangWu/LightCNN
2. Note.txt写入了复现模型存在的一些问题；
3. requirements.txt为执行环境；
4. 首先执行training.py;然后执行test.py.
5. 一些结果如下，从左到右：$`x`$  ，   $`\widehat{x}`$   ，  $`x^{p,f}`$  ，  $`x^{rec}`$.
![33](https://github.com/1056891520/FESGAN_manipulator_master/assets/71159747/5a2f7053-6e18-4ebc-803f-3efb60123572)
![44](https://github.com/1056891520/FESGAN_manipulator_master/assets/71159747/33e7b84b-b779-4483-a77c-2418bc7e0305)
![43](https://github.com/1056891520/FESGAN_manipulator_master/assets/71159747/62ad60ac-9575-4111-88b1-6b8452b36781)
![34](https://github.com/1056891520/FESGAN_manipulator_master/assets/71159747/301dfbb8-a228-4076-bc07-55df6417b6c4)
6. 模型训练的收敛过程如下，分别是loss_d_z, loss_d_img, loss_reg(情绪分类器)，loss_gen, train_acc;
<img src="https://github.com/1056891520/FESGAN_manipulator_master/assets/71159747/33559122-682b-49a0-bf38-1b31989b5dc8" width="300px">
<img src="https://github.com/1056891520/FESGAN_manipulator_master/assets/71159747/ac0ae443-e1df-4e7e-b765-36843e899830" width="300px">
<img src="https://github.com/1056891520/FESGAN_manipulator_master/assets/71159747/648daf48-5986-4219-b2e7-8d134a32acf0" width="300px">
<img src="https://github.com/1056891520/FESGAN_manipulator_master/assets/71159747/95f6c097-9001-4a5a-b52a-7db7e16397fc" width="300px">
<img src="https://github.com/1056891520/FESGAN_manipulator_master/assets/71159747/07e32212-7f7f-4327-8afe-8e9f90fb8a2f" width="300px">
## 注:
1. 若原作者发现这个库，如有不妥请留言告知；
2. 若他人引用此代码，请标注出处；
3. 欢迎留言讨论交流，共同进步。
