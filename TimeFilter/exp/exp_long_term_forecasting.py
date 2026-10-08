from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.financial_selection import FinancialSelection, best_epoch_for_checkpoint
from utils.financial_progress import emit_progress
from utils.financial_gradient_diagnostic import record_gradient_diagnostic
from utils.financial_losses import stockmixer_rank_loss, stockmixer_validation_loss, daily_pearson_ic_loss
from utils.financial_report import FinancialReport, financial_metrics, metric_line
from utils.walkforward_selection import WalkforwardRankCheckpoint, epoch_correlations
from data_provider.financial_registry import is_financial_dataset
import torch
import torch.nn as nn
from torch import optim
import os
import json
import time
import warnings
import numpy as np
from torch.utils.data import default_collate

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
        if is_financial_dataset(self.args.data) and getattr(self.args, 'financial_optimizer', 'adam') == 'adamw':
            weight_decay = getattr(self.args, 'financial_weight_decay', 0.0)
            if weight_decay:
                decay, no_decay = [], []
                for name, parameter in self.model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    group = (no_decay if parameter.ndim < 2 or 'norm' in name.lower()
                             else decay)
                    group.append(parameter)
                parameters = [{'params': decay, 'weight_decay': weight_decay},
                              {'params': no_decay, 'weight_decay': 0.0}]
            else:
                parameters = self.model.parameters()
            model_optim = optim.AdamW(parameters, lr=self.args.learning_rate)
        else:
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

    def _training_loss_components(self, outputs, target, batch_y_mark, criterion, rank_weight):
        prediction_loss = self._prediction_loss(outputs, target, batch_y_mark, criterion)
        rank_loss = (stockmixer_rank_loss(outputs, target, self._financial_mask(batch_y_mark, outputs))
                     if rank_weight else prediction_loss.new_zeros(()))
        return prediction_loss, rank_loss

    def vali(self, vali_data, vali_loader, criterion, stockmixer_selection=False):
        total_loss = []
        batch_sizes = []
        collect_financial = is_financial_dataset(self.args.data)
        financial_preds, financial_trues, financial_masks = [], [], []
        selection_loss_sum = 0.0
        selection_days = 0
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
                    financial_mask = self._financial_mask(batch_y_mark, true)
                    financial_masks.append(financial_mask.numpy())
                    if stockmixer_selection:
                        daily_objective = stockmixer_validation_loss(
                            pred, true, financial_mask,
                            getattr(self.args, 'stockmixer_selection_rank_weight', 0.1))
                        selection_loss_sum += daily_objective.item() * pred.shape[0]
                        selection_days += pred.shape[0]

                loss = self._prediction_loss(pred, true, batch_y_mark, criterion)

                total_loss.append(loss)
                if collect_financial:
                    batch_sizes.append(pred.shape[0])
        total_loss = np.average(total_loss, weights=batch_sizes if collect_financial else None)
        if collect_financial:
            predictions = np.concatenate(financial_preds, axis=0)
            targets = np.concatenate(financial_trues, axis=0)
            masks = np.concatenate(financial_masks, axis=0)
            self._last_financial_metrics = financial_metrics(
                predictions, targets, masks,
            )
            if stockmixer_selection:
                self._last_financial_metrics['stockmixer_val_loss'] = selection_loss_sum / selection_days
                if getattr(self.args, 'financial_walkforward', False):
                    midpoint = len(predictions) // 2
                    for name, window in (('early', slice(None, midpoint)),
                                         ('late', slice(midpoint, None))):
                        part = financial_metrics(predictions[window], targets[window], masks[window])
                        self._last_financial_metrics[f'IC_{name}'] = part['IC']
                        self._last_financial_metrics[f'RankIC_{name}'] = part['RankIC']
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        financial = is_financial_dataset(self.args.data)
        validation_only = financial and getattr(self.args, 'financial_validation_only', False)
        test_each_epoch = not validation_only and (
            not financial or bool(getattr(self.args, 'financial_test_each_epoch', 1)))
        if test_each_epoch:
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
        rank_weight = getattr(self.args, 'rank_weight', 0.0) if is_financial_dataset(self.args.data) else 0.0
        grad_clip_norm = (getattr(self.args, 'financial_grad_clip_norm', 0.0) if financial else 0.0)
        finance = getattr(self.args, 'finance_adaptation', {})
        ic_weight = (finance.get('loss', {}).get('ic_weight', 0.0)
                     if finance.get('enabled', False) and is_financial_dataset(self.args.data) else 0.0)

        rank_candidate = None
        if is_financial_dataset(self.args.data):
            self._financial_report = FinancialReport(setting, self.args, 'training')
            selection = FinancialSelection(path, self._financial_report, getattr(self.args, 'financial_selection', 'mse'))
            rank_candidate = (WalkforwardRankCheckpoint(path, self._financial_report)
                              if getattr(self.args, 'financial_walkforward', False) else None)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        diagnostic_epochs = set(getattr(self.args, 'gradient_diagnostic_epochs', [])) if is_financial_dataset(self.args.data) else set()
        if diagnostic_epochs:
            batch_size = min(self.args.gradient_diagnostic_batch_size, len(train_data))
            if batch_size == 0:
                raise ValueError('Gradient diagnostic requires a nonempty training dataset')
            train_indices = np.linspace(0, len(train_data) - 1, batch_size, dtype=int).tolist()
            diagnostic_batch = default_collate([train_data[index] for index in train_indices])

            def run_diagnostic(epoch):
                def losses_for_batch(batch):
                    batch_x, batch_y, _, batch_y_mark = batch
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    outputs, moe_loss = self.model(batch_x, self.masks, is_training=True)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    target = batch_y[:, -self.args.pred_len:, f_dim:]
                    mse, rank = self._training_loss_components(
                        outputs, target, batch_y_mark, criterion, rank_weight)
                    components = {'MSE': (mse, 1.0), 'Rank': (rank, rank_weight),
                                  'IC': (daily_pearson_ic_loss(
                                      outputs, target, self._financial_mask(batch_y_mark, outputs)), ic_weight),
                                  'MoE': (moe_loss, moe_aux_weight)}
                    return components

                record_gradient_diagnostic(self.model, diagnostic_batch, train_indices,
                                           losses_for_batch, self._financial_report.path, epoch)

            if 0 in diagnostic_epochs:
                run_diagnostic(0)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_mse, train_rank, train_moe, train_ic = [], [], [], []
            clipped_steps = 0
            if is_financial_dataset(self.args.data):
                emit_progress(epoch + 1, self.args.train_epochs, 0, train_steps)
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
                prediction_loss, rank_loss = self._training_loss_components(
                    outputs, batch_y, batch_y_mark, criterion, rank_weight)
                loss = prediction_loss + rank_weight * rank_loss
                if ic_weight:
                    ic_loss = daily_pearson_ic_loss(
                        outputs, batch_y, self._financial_mask(batch_y_mark, outputs))
                    loss = loss + ic_weight * ic_loss
                    train_ic.append(ic_loss.item())
                if moe_aux_weight:
                    loss = loss + moe_aux_weight * moe_loss
                train_mse.append(prediction_loss.item())
                train_rank.append(rank_loss.item())
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
                    if grad_clip_norm:
                        scaler.unscale_(model_optim)
                        gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                        clipped_steps += int(gradient_norm > grad_clip_norm)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm:
                        gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                        clipped_steps += int(gradient_norm > grad_clip_norm)
                    model_optim.step()
                if is_financial_dataset(self.args.data):
                    emit_progress(epoch + 1, self.args.train_epochs, i + 1, train_steps)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(
                vali_data, vali_loader, criterion,
                stockmixer_selection=(getattr(self.args, 'financial_selection', None) == 'stockmixer_val_loss'))
            if is_financial_dataset(self.args.data):
                val_financial = self._last_financial_metrics.copy()
            # Optional StockMixer-style test reporting; checkpoint selection remains validation-only.
            test_loss = self.vali(test_data, test_loader, criterion) if test_each_epoch else float("nan")

            message = "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss)
            if is_financial_dataset(self.args.data) and 'stockmixer_val_loss' in val_financial:
                message += f" StockMixer Select Loss: {val_financial['stockmixer_val_loss']:.7f}"
            if validation_only:
                message += ' | Test skipped (validation-only)'
            elif test_each_epoch:
                message += f' Test Loss: {test_loss:.7f}'
            else:
                message += ' | Test deferred to final best.pth'
            print(message)
            if is_financial_dataset(self.args.data):
                test_financial = self._last_financial_metrics.copy() if test_each_epoch else {}
                print(metric_line('Val', val_financial))
                if test_each_epoch:
                    print(metric_line('Test', test_financial))
                components = {'train_mse': float(np.mean(train_mse)), 'train_moe': float(np.mean(train_moe)),
                              'train_moe_weighted': float(moe_aux_weight * np.mean(train_moe)) if moe_aux_weight else 0.0,
                              'train_rank': float(np.mean(train_rank)),
                              'train_rank_weighted': float(rank_weight * np.mean(train_rank)),
                              'rank_weight': rank_weight,
                              'moe_aux_weight': moe_aux_weight,
                              'learning_rate': model_optim.param_groups[0]['lr']}
                if grad_clip_norm:
                    components.update(grad_clip_norm=grad_clip_norm,
                                      grad_clip_fraction=clipped_steps / train_steps)
                if ic_weight:
                    components.update(train_ic=float(np.mean(train_ic)),
                                      train_ic_weighted=float(ic_weight * np.mean(train_ic)),
                                      ic_weight=ic_weight)
                components['peak_cuda_allocated_mb'] = (torch.cuda.max_memory_allocated(self.device) / 2**20
                                                       if self.device.type == 'cuda' else 0.0)
                print(f"Train components | MSE: {components['train_mse']:.8f} "
                      f"Rank: {components['train_rank']:.8f} weighted Rank: {components['train_rank_weighted']:.8f} "
                      f"MoE: {components['train_moe']:.8f} weighted MoE: {components['train_moe_weighted']:.8f}")
                if ic_weight:
                    print(f"Train IC loss: {components['train_ic']:.8f} "
                          f"weighted IC loss: {components['train_ic_weighted']:.8f}")
                self._financial_report.epoch(
                    epoch + 1, train_loss, vali_loss, test_loss,
                    val_financial, test_financial, components,
                )
                selection.update(self.model, epoch + 1, val_financial)
                if rank_candidate is not None:
                    rank_candidate.update(self.model, epoch + 1, val_financial)
            else:
                early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            if epoch + 1 in diagnostic_epochs:
                run_diagnostic(epoch + 1)

        best_model_path = selection.finish() if is_financial_dataset(self.args.data) else path + '/' + 'checkpoint.pth'
        if rank_candidate is not None:
            rank_candidate.finish(selection.best)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def walkforward_evaluate(self, setting):
        """Compare the two frozen validation choices on the later historical block."""
        from pathlib import Path
        import shutil

        report = self._financial_report
        folder = report.path
        shutil.copy2(folder / 'test_metrics.json', folder / 'future_metrics_A.json')
        shutil.copy2(folder / 'test_predictions.npz', folder / 'future_predictions_A.npz')

        checkpoint = Path(self.args.checkpoints) / setting / 'best_rankic.pth'
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        dataset, loader = self._get_data(flag='test')
        predictions, targets, masks = [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, _, batch_y_mark in loader:
                outputs, _ = self.model(batch_x.float().to(self.device), self.masks, is_training=False)
                pred = outputs[:, -self.args.pred_len:, :].detach().cpu()
                true = batch_y[:, -self.args.pred_len:, :].float()
                predictions.append(pred.numpy())
                targets.append(true.numpy())
                masks.append(self._financial_mask(batch_y_mark, true).numpy())
        pred = np.concatenate(predictions, axis=0)
        true = np.concatenate(targets, axis=0)
        mask = np.concatenate(masks, axis=0)
        metrics = financial_metrics(pred, true, mask)
        report.write_json('future_metrics_B.json', metrics)
        np.savez_compressed(folder / 'future_predictions_B.npz',
                            prediction=np.ascontiguousarray(pred[:, 0, :].T),
                            ground_truth=np.ascontiguousarray(true[:, 0, :].T),
                            mask=np.ascontiguousarray(mask[:, 0, :].T),
                            target_index=np.arange(dataset.target_start, dataset.target_end),
                            source_day_index=np.arange(dataset.target_start, dataset.target_end) + dataset.START_DAY,
                            lookback_length=np.asarray(dataset.seq_len), horizon=np.asarray(dataset.pred_len))
        report.write_json('selection_future_correlation.json', epoch_correlations(folder))
        print(metric_line('Future A (validation loss)',
                          json.loads((folder / 'future_metrics_A.json').read_text(encoding='utf-8'))))
        print(metric_line('Future B (validation RankIC)', metrics))

    def test(self, setting, test=0):
        checkpoint = None
        if test and is_financial_dataset(self.args.data):
            from pathlib import Path
            from utils.financial_checkpoint import resolve_financial_checkpoint
            checkpoint = resolve_financial_checkpoint(
                self.args, setting, Path(__file__).resolve().parents[1])
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            if checkpoint is None:
                checkpoint = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            else:
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
            best_checkpoint = (checkpoint if test else
                               os.path.join(self.args.checkpoints, setting, 'best.pth'))
            best_epoch = best_epoch_for_checkpoint(best_checkpoint)
            best_epoch_text = str(best_epoch) if best_epoch is not None else 'unknown (selection.json unavailable)'
            print(f'Final Test | {os.path.basename(best_checkpoint)} from epoch {best_epoch_text}')
            print(metric_line('Final Test', metrics))
            if test or not hasattr(self, '_financial_report'):
                self._financial_report = FinancialReport(setting, self.args, 'evaluation_only')
            self._financial_report.final(metrics, preds, trues, test_data, masks)

        return
