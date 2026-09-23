from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, \
                                      Dataset_PEMS, Dataset_Climate
from data_provider.uea import collate_fn
from data_provider.stockmixer_sp500 import Dataset_SP500
from data_provider.stockmixer_exchange import Dataset_Exchange
from data_provider.financial_registry import is_financial_dataset
from torch.utils.data import DataLoader
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'Climate': Dataset_Climate,
    'SP500': Dataset_SP500,
    'S&P500': Dataset_SP500,
    'NASDAQ': Dataset_Exchange,
    'NYSE': Dataset_Exchange,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    if is_financial_dataset(args.data) and flag != 'train':
        shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader
