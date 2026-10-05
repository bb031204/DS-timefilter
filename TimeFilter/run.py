import argparse
import json
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
from utils.finance_adaptation_config import normalize_finance_adaptation

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='PatchTST')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='PatchTST',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    
    # TimeFilter
    parser.add_argument('--patch_len', type=int, default=16, help='length of patch')
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--moe_aux_weight', type=float, default=0.05,
                        help='MoE auxiliary loss weight for long-term forecasting; 0 disables the auxiliary objective')
    parser.add_argument('--rank_weight', type=float, default=0.0,
                        help='Financial cross-sectional ranking loss weight; 0 preserves original loss')
    parser.add_argument('--financial_optimizer', choices=['adam', 'adamw'], default='adam',
                        help='Financial optimizer; non-financial tasks keep the original Adam')
    parser.add_argument('--financial_weight_decay', type=float, default=0.0,
                        help='AdamW weight decay for financial training')
    parser.add_argument('--financial_grad_clip_norm', type=float, default=0.0,
                        help='Financial global gradient norm limit; 0 disables clipping')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='cosine', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

     # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    parser.add_argument('--financial_output_dir', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--financial_checkpoint_root', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--financial_config', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--financial_checkpoint', default=None)
    parser.add_argument('--financial_cpu', action='store_true')
    parser.add_argument('--financial_seed', type=int, default=2021)
    parser.add_argument('--financial_norm', type=int, choices=[0, 1], default=1,
                        help='Financial TimeFilter normalization: 1 original behavior, 0 bypass')
    parser.add_argument('--financial_input_features', choices=['returns', 'eod5'], default='returns',
                        help='SP500 input: daily returns or StockMixer five EOD features')
    parser.add_argument('--finance_adaptation', type=json.loads, default=None,
                        help='Finance-only modular settings, encoded as JSON by scripts/run_financial.py')
    parser.add_argument('--financial_selection', choices=['mse', 'RankIC', 'IC', 'stockmixer_val_loss'], default='mse')
    parser.add_argument('--financial_test_each_epoch', type=int, choices=[0, 1], default=1,
                        help='Evaluate financial test set after each epoch; 0 defers testing until final best.pth')
    parser.add_argument('--financial_walkforward', action='store_true',
                        help='Run the SP500 historical A/B checkpoint-selection audit')
    parser.add_argument('--financial_split', type=json.loads, default=None,
                        help='Historical SP500 split JSON: train_end, valid_end, future_end')
    parser.add_argument('--stockmixer_selection_rank_weight', type=float, default=0.1,
                        help='Rank weight in StockMixer-style validation loss; independent of training rank_weight')
    parser.add_argument('--gradient_diagnostic_epochs', type=int, nargs='*', default=[],
                        help='Financial gradient checks before training (0) and after selected epochs; empty disables')
    parser.add_argument('--gradient_diagnostic_batch_size', type=int, default=8,
                        help='Number of fixed training dates used by each financial gradient check')
    parser.add_argument('--financial_validation_only', action='store_true')
    parser.add_argument('--financial_force_rerun', action='store_true',
                        help='Repeat completed financial training; concurrent identical training remains blocked')
    args = parser.parse_args()
    try:
        args.finance_adaptation = normalize_finance_adaptation(args.finance_adaptation)
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.moe_aux_weight) or args.moe_aux_weight < 0:
        parser.error('--moe_aux_weight must be finite and non-negative')
    if not np.isfinite(args.rank_weight) or args.rank_weight < 0:
        parser.error('--rank_weight must be finite and non-negative')
    if not np.isfinite(args.financial_weight_decay) or args.financial_weight_decay < 0:
        parser.error('--financial_weight_decay must be finite and non-negative')
    if not np.isfinite(args.financial_grad_clip_norm) or args.financial_grad_clip_norm < 0:
        parser.error('--financial_grad_clip_norm must be finite and non-negative')
    if args.financial_optimizer == 'adam' and args.financial_weight_decay:
        parser.error('--financial_weight_decay requires --financial_optimizer adamw')
    if not np.isfinite(args.stockmixer_selection_rank_weight) or args.stockmixer_selection_rank_weight < 0:
        parser.error('--stockmixer_selection_rank_weight must be finite and non-negative')
    if (any(epoch < 0 for epoch in args.gradient_diagnostic_epochs)
            or args.gradient_diagnostic_epochs != sorted(set(args.gradient_diagnostic_epochs))):
        parser.error('--gradient_diagnostic_epochs must be sorted, distinct and non-negative')
    if args.gradient_diagnostic_batch_size <= 0:
        parser.error('--gradient_diagnostic_batch_size must be positive')
    from data_provider.financial_registry import is_financial_dataset, validate_files

    finance = args.finance_adaptation
    if finance['enabled'] and not is_financial_dataset(args.data):
        parser.error('--finance_adaptation only applies to financial datasets')
    if finance['enabled'] and finance['input_adapter']['enabled'] and args.financial_input_features != 'eod5':
        parser.error('Finance input adapter requires --financial_input_features eod5')
    if finance['enabled'] and finance['positional_encoding']['mode'] == 'patch_only' and args.pos != 1:
        parser.error('Patch-only positional encoding requires --pos 1')

    if args.financial_input_features == 'eod5' and args.data not in ('SP500', 'S&P500'):
        parser.error('--financial_input_features eod5 only applies to SP500')
    if args.financial_input_features == 'eod5' and args.financial_norm != 0:
        parser.error('--financial_input_features eod5 requires --financial_norm 0')
    if args.financial_walkforward:
        if args.data not in ('SP500', 'S&P500') or args.itr != 1:
            parser.error('--financial_walkforward requires one SP500 run')
        if args.financial_selection != 'stockmixer_val_loss':
            parser.error('--financial_walkforward requires StockMixer validation-loss selection for A')
        split = args.financial_split
        if (not isinstance(split, dict) or set(split) != {'train_end', 'valid_end', 'future_end'}
                or any(type(value) is not int for value in split.values())
                or not args.seq_len < split['train_end'] < split['valid_end'] < split['future_end'] <= 1259):
            parser.error('walk-forward split must satisfy seq_len < train_end < valid_end < future_end <= 1259')
    elif args.financial_split is not None:
        parser.error('--financial_split is reserved for --financial_walkforward')
    if args.financial_selection == 'stockmixer_val_loss' and args.data not in ('SP500', 'S&P500'):
        parser.error('--financial_selection stockmixer_val_loss only applies to SP500')
    if not is_financial_dataset(args.data) and (args.financial_norm != 1 or args.rank_weight != 0):
        parser.error('--financial_norm and --rank_weight only apply to financial datasets')
    if not is_financial_dataset(args.data) and args.gradient_diagnostic_epochs:
        parser.error('--gradient_diagnostic_epochs only applies to financial datasets')
    if is_financial_dataset(args.data):
        if not args.financial_output_dir:
            import sys
            from utils.financial_runtime import launch_financial
            raise SystemExit(launch_financial(args, sys.argv[1:]))
        if args.model != 'TimeFilter' or args.task_name != 'long_term_forecast':
            raise ValueError('Financial adapter requires TimeFilter and long_term_forecast')
        validate_files(args.root_path, args.data)
        if args.financial_cpu:
            args.use_gpu = False
        random.seed(args.financial_seed)
        np.random.seed(args.financial_seed)
        torch.manual_seed(args.financial_seed)
        torch.cuda.manual_seed_all(args.financial_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.financial_walkforward and args.is_training and args.financial_output_dir:
        from pathlib import Path
        import yaml
        Path(args.financial_output_dir, 'config.yaml').write_text(
            yaml.safe_dump(vars(args), allow_unicode=True, sort_keys=False), encoding='utf-8')

    print('Args in experiment:')
    print_args(args)
    if is_financial_dataset(args.data):
        print(f'Financial protocol | seed={args.financial_seed} patch_len={args.patch_len} '
              f'input={args.financial_input_features} norm={bool(args.financial_norm)} rank_weight={args.rank_weight} '
              f'alpha={args.alpha} moe_aux_weight={args.moe_aux_weight} '
              f'selection=validation:{args.financial_selection} '
              f'selection_rank_weight={args.stockmixer_selection_rank_weight} '
              f'optimizer={args.financial_optimizer} weight_decay={args.financial_weight_decay} '
              f'grad_clip_norm={args.financial_grad_clip_norm} '
              f'test_each_epoch={bool(args.financial_test_each_epoch)} '
              f'validation_only={args.financial_validation_only}')
        if finance['enabled']:
            print(f'Finance adaptation | {json.dumps(finance, ensure_ascii=False, sort_keys=True)}')

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if not (is_financial_dataset(args.data) and args.financial_validation_only):
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                if args.financial_walkforward:
                    exp.walkforward_evaluate(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
