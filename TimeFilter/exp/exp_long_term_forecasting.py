from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.financial_selection import FinancialSelection
from utils.financial_report import FinancialReport, financial_metrics, metric_line
from data_provider.financial_registry import is_financial_dataset
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.masks = self._get_mask()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _get_mask(self):
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks
    
    def _get_mask_2(self):
        dtype = torch.float32
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len

        mask_base = torch.eye(L, device=self.device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        mask0 = torch.eye(L, device=self.device, dtype=dtype)
        mask0.view(self.args.c_out, N, self.args.c_out, N).diagonal(dim1=0, dim2=2).fill_(1)
        mask0 = mask0.unsqueeze(0).unsqueeze(0) - mask_base
        mask1 = torch.kron(torch.ones(self.args.c_out, self.args.c_out, device=self.device, dtype=dtype), 
                            torch.eye(N, device=self.device, dtype=dtype))
        mask1 = mask1.unsqueeze(0).unsqueeze(0) - mask_base
        mask2 = torch.ones((1, 1, L, L), device=self.device, dtype=dtype) - mask1 - mask0 - mask_base
        masks = torch.cat([mask0, mask1, mask2], dim=0)  # [3, 1, L, L]
        return masks

    def _financial_mask(self, batch_y_mark, reference):
        if self.args.data in ('NASDAQ', 'NYSE'):
            return batch_y_mark[:, -self.args.pred_len:, :].to(reference.device)
        return torch.ones_like(reference)

    def _prediction_loss(self, prediction, target, batch_y_mark, criterion):
        if self.args.data in ('NASDAQ', 'NYSE'):
            mask = self._financial_mask(batch_y_mark, prediction)
            return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1)
        return criterion(prediction, target)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        collect_financial = is_financial_dataset(self.args.data)
        financial_preds, financial_trues, financial_masks = [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                outputs, _ = self.model(batch_x, self.masks, is_training=False)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if collect_financial:
                    financial_preds.append(pred.numpy())
                    financial_trues.append(true.numpy())
                    financial_masks.append(self._financial_mask(batch_y_mark, true).numpy())

                loss = self._prediction_loss(pred, true, batch_y_mark, criterion)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        if collect_financial:
            self._last_financial_metrics = financial_metrics(
                np.concatenate(financial_preds, axis=0),
                np.concatenate(financial_trues, axis=0),
                np.concatenate(financial_masks, axis=0),
            )
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        validation_only = is_financial_dataset(self.args.data) and getattr(self.args, 'financial_validation_only', False)
        if not validation_only:
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        moe_aux_weight = getattr(self.args, 'moe_aux_weight', 0.05)

        if is_financial_dataset(self.args.data):
            self._financial_report = FinancialReport(setting, self.args, 'training')
            selection = FinancialSelection(path, self._financial_report, getattr(self.args, 'financial_selection', 'mse'))

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_mse, train_moe = [], []
            if is_financial_dataset(self.args.data) and self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs, moe_loss = self.model(batch_x, self.masks, is_training=True)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                prediction_loss = self._prediction_loss(outputs, batch_y, batch_y_mark, criterion)
                loss = prediction_loss if moe_aux_weight == 0 else prediction_loss + moe_aux_weight * moe_loss
                train_mse.append(prediction_loss.item())
                train_moe.append(float(moe_loss.detach()) if torch.is_tensor(moe_loss) else float(moe_loss))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if is_financial_dataset(self.args.data):
                val_financial = self._last_financial_metrics.copy()
            test_loss = float("nan") if validation_only else self.vali(test_data, test_loader, criterion)

            message = "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss)
            print(message + (' | Test skipped (validation-only)' if validation_only else f' Test Loss: {test_loss:.7f}'))
            if is_financial_dataset(self.args.data):
                test_financial = {} if validation_only else self._last_financial_metrics.copy()
                print(metric_line('Val', val_financial))
                if not validation_only:
                    print(metric_line('Test', test_financial))
                components = {'train_mse': float(np.mean(train_mse)), 'train_moe': float(np.mean(train_moe)),
                              'train_moe_weighted': float(moe_aux_weight * np.mean(train_moe)) if moe_aux_weight else 0.0,
                              'moe_aux_weight': moe_aux_weight,
                              'learning_rate': model_optim.param_groups[0]['lr']}
                components['peak_cuda_allocated_mb'] = (torch.cuda.max_memory_allocated(self.device) / 2**20
                                                       if self.device.type == 'cuda' else 0.0)
                print(f"Train components | MSE: {components['train_mse']:.8f} MoE: {components['train_moe']:.8f} weighted MoE: {components['train_moe_weighted']:.8f}")
                self._financial_report.epoch(
                    epoch + 1, train_loss, vali_loss, test_loss,
                    val_financial, test_financial, components,
                )
                selection.update(self.model, epoch + 1, val_financial)
            else:
                early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = selection.finish() if is_financial_dataset(self.args.data) else path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            checkpoint = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if is_financial_dataset(self.args.data):
                checkpoint = getattr(self.args, 'financial_checkpoint', None) or os.path.join(
                    getattr(self.args, 'financial_checkpoint_root', None) or self.args.checkpoints,
                    setting, 'checkpoint.pth')
                print(f'Checkpoint source: {checkpoint}')
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

        preds = []
        trues = []
        inputs = []
        financial_masks = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs, _ = self.model(batch_x, self.masks, is_training=False)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                if is_financial_dataset(self.args.data):
                    financial_masks.append(self._financial_mask(batch_y_mark, batch_y).cpu().numpy())
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]

                input_ = batch_x
                pred = outputs
                true = batch_y

                inputs.append(input_)
                preds.append(pred)
                trues.append(true)

        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'input.npy', inputs)
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        if is_financial_dataset(self.args.data):
            masks = np.concatenate(financial_masks, axis=0)
            metrics = financial_metrics(preds, trues, masks)
            print(metric_line('Final Test', metrics))
            if test or not hasattr(self, '_financial_report'):
                self._financial_report = FinancialReport(setting, self.args, 'evaluation_only')
            self._financial_report.final(metrics, preds, trues, test_data, masks)

        return
