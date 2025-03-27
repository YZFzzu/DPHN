import torch.nn.functional as F
import torch
from scipy import stats
import numpy as np
from dataloader2 import DataLoader
from RestoredNet3 import *
from CNNVIT_net8_up import *

import timm
from timm.models.vision_transformer import Block
from timm.models.resnet import BasicBlock,Bottleneck
from scipy.stats import spearmanr, pearsonr

class demoIQASolver(object):
    """training and testing"""
    def __init__(self, config, path, train_idx, test_idx):
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.remodel = RestoredNet3().cuda()
        self.remodel.train(True)
        self.l1_loss = torch.nn.L1Loss().cuda()
        self.lr1 = config.lr #2e-4
        self.lr2 =2e-4
        self.resnet50 = timm.create_model('resnet50', pretrained=True).cuda()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()
        self.init_saveoutput()
        self.transformer_block = ModifiedTransformerBlock(dim=768, heads=8, dim_head=96).cuda()
        self.predict=ImageQualityPredictor().cuda()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.remodel.parameters(), 'lr': self.lr1},
            {'params': self.transformer_block.parameters(), 'lr': self.lr2, 'weight_decay': 1e-05},
            {'params': self.predict.parameters(), 'lr': self.lr2, 'weight_decay': 1e-05}
            ])
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs,
        #                                                             eta_min=0)
        train_loader = DataLoader(config.dataset,
                                              path,
                                              train_idx,
                                              config.patch_size,
                                              config.train_patch_num,
                                              batch_size=config.batch_size,
                                              istrain=True)
        test_loader = DataLoader(config.dataset,
                                             path,
                                             test_idx,
                                             config.patch_size,
                                             config.test_patch_num,
                                             istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def init_saveoutput(self):
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.resnet50.modules():
            if isinstance(layer, Bottleneck):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)


    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0

        for t in range(self.epochs):
            self.remodel.train()
            self.predict.train()
            self.transformer_block.train()
            losses = []
            # save data for one epoch
            pred_epoch = []
            labels_epoch = []
            for img, label in self.train_data: #img(16,3,224,224) label(16,)
                img = torch.as_tensor(img.cuda())
                label = torch.as_tensor(label.cuda())
                self.optimizer.zero_grad()
                restored_img = self.remodel(img)
#初始化 SaveOutput 对象：在 ViT 模型的初始化阶段，为特定的 Block 层注册了 SaveOutput 钩子，这样每当这些层进行前向传播时，它们的输出都会被保存起来。
                _x = self.vit(img)#_x(16,1000)
                vit_dis_layer0,vit_dis_layer1,vit_dis_layer2,vit_dis_layer3,vit_dis_layer4= get_vit_feature_layer(self.save_output)#0-4 (16,196,768)
                vit_dis = get_vit_feature(self.save_output)
                self.save_output.outputs.clear()
                del _x

                _y = self.vit(restored_img)
                vit_ref = get_vit_feature(self.save_output)
                self.save_output.outputs.clear()
                del _y

                B, N, C = vit_ref.shape
                H, W = 14, 14
                vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)
                vit_ref = vit_ref.transpose(1, 2).view(B, C, H, W)#(16,768*5,14,14)


                _ = self.resnet50(img)#_(16,1000)
                cnn_dis= get_resnet_feature(self.save_output)# 0,1,2都是[B,256,56,56]（16，768，56，56）
                self.save_output.outputs.clear()
                #调整维度
                cnn_dis= unified_dimensions(cnn_dis)# (16,196,768)
                #融合
                fused_features_layer0 = self.transformer_block(vit_dis_layer0, extra_kv=cnn_dis)  # (16,196,768)
                fused_features_layer1 = self.transformer_block(vit_dis_layer1, extra_kv=cnn_dis)  # (16,196,768)
                fused_features_layer2 = self.transformer_block(vit_dis_layer2, extra_kv=cnn_dis)  # (16,196,768)
                fused_features_layer3 = self.transformer_block(vit_dis_layer3, extra_kv=cnn_dis)  # (16,196,768)
                fused_features_layer4 = self.transformer_block(vit_dis_layer4, extra_kv=cnn_dis)  # (16,196,768)

                fused_features = fused(fused_features_layer0,fused_features_layer1,fused_features_layer2,fused_features_layer3,fused_features_layer4)

                pred=self.predict(vit_dis,vit_ref,fused_features)

                loss = self.criterion(torch.squeeze(pred), label)
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

                # save results in one epoch
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = label.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)
                # compute correlation coefficient
            train_rho_srcc, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            train_rho_prcc, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

            train_ret_loss = np.mean(losses)
            print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(t + 1, train_ret_loss, train_rho_srcc, train_rho_prcc))
            # Update optimizer
            if (t + 1) % 10 == 0:
                lr1 = (self.lr1 * 9) / pow(10, ((t+1) // 10))#每10个epoch下降0.9
                lr2 =  (self.lr2 * 9) / pow(10, ((t+1) // 10))#每10个epoch下降0.9
                self.optimizer = torch.optim.Adam([
                {'params': self.remodel.parameters(), 'lr': lr1},
                {'params': self.transformer_block.parameters(), 'lr': lr2, 'weight_decay': 1e-05},
                {'params': self.predict.parameters(), 'lr': lr2, 'weight_decay': 1e-05}
            ])
            for i, param_group in enumerate(self.optimizer.param_groups):
                print(f"  The next epoch. Param Group {i} - LR: {param_group['lr']} - Weight Decay: {param_group['weight_decay']}")

            test_ret_loss, test_rho_srcc, test_rho_prcc = self.test(self.test_data)
            if test_rho_srcc > best_srcc:
                best_srcc = test_rho_srcc
                best_plcc = test_rho_prcc
                state_dicts = {
                    'remodel': self.remodel.state_dict(),
                    'fusion': self.transformer_block.state_dict(),
                    'predict': self.predict.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                }

                torch.save(state_dicts, "./model/without_{:.4f}_{:.4f}.pth".format(best_srcc,best_plcc))
                print("Save model success !!! SROCC: {:.4f} PLCC: {:.4f}".format(best_srcc, best_plcc))
        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))
        return best_srcc, best_plcc

    def test(self, data):
        losses = []
        """Testing"""
        with torch.no_grad():
            self.remodel.eval()
            self.predict.eval()
            self.transformer_block.eval()
            self.vit.eval()
            self.resnet50.eval()
            # save data for one epoch
            pred_epoch = []
            labels_epoch = []

            for img, label in data:
                img = torch.as_tensor(img.cuda())
                label = torch.as_tensor(label.cuda())
                restored_img = self.remodel(img)

                _x = self.vit(img)
                vit_dis_layer0,vit_dis_layer1,vit_dis_layer2,vit_dis_layer3,vit_dis_layer4 = get_vit_feature_layer(self.save_output)
                vit_dis = get_vit_feature(self.save_output)
                self.save_output.outputs.clear()
                del _x

                _y = self.vit(restored_img)
                vit_ref = get_vit_feature(self.save_output)
                self.save_output.outputs.clear()
                del _y

                B, N, C = vit_ref.shape
                H, W = 14, 14
                vit_ref = vit_ref.transpose(1, 2).view(B, C, H, W)
                vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)

                _ = self.resnet50(img)  # _(16,1000)
                cnn_dis= get_resnet_feature(self.save_output)  # (16,768,56,56)  # 0,1,2都是[B,768,56,56]
                self.save_output.outputs.clear()

                # 调整维度
                cnn_dis=unified_dimensions(cnn_dis)

                fused_features_layer0 = self.transformer_block(vit_dis_layer0, extra_kv=cnn_dis)  # (16,196,768)
                fused_features_layer1 = self.transformer_block(vit_dis_layer1, extra_kv=cnn_dis)  # (16,196,768)
                fused_features_layer2 = self.transformer_block(vit_dis_layer2, extra_kv=cnn_dis)  # (16,196,768)
                fused_features_layer3 = self.transformer_block(vit_dis_layer3, extra_kv=cnn_dis)  # (16,196,768)
                fused_features_layer4 = self.transformer_block(vit_dis_layer4, extra_kv=cnn_dis)  # (16,196,768)
                fused_features = fused(fused_features_layer0,fused_features_layer1,fused_features_layer2,fused_features_layer3,fused_features_layer4)

                pred = self.predict(vit_dis,vit_ref, fused_features)

                # compute loss
                loss = self.criterion(torch.squeeze(pred), label)
                loss_val = loss.item()
                losses.append(loss_val)

                # save results in one epoch
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = label.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)

            # compute correlation coefficient
            test_rho_srcc, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            test_rho_prcc, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            print('===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(np.mean(losses), test_rho_srcc,
                                                                                       test_rho_prcc))
            return np.mean(losses), test_rho_srcc, test_rho_prcc

