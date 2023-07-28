import pandas as pd
from argparse import ArgumentParser
from torch import cuda
from trainer import T5Trainer

parser = ArgumentParser()

parser.add_argument("--data-file", type=str, required=True)
parser.add_argument("--output-dir", type=str, default="./output/")

args = parser.parse_args()

df = pd.read_csv(args.data_file)
print(df.sample(10))

device = 'cuda' if cuda.is_available() else 'cpu'


model_params={
    "MODEL":"t5-base",             # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE":8,          # training batch size
    "VALID_BATCH_SIZE":8,          # validation batch size
    "TRAIN_EPOCHS":3,              # number of training epochs
    "VAL_EPOCHS":1,                # number of validation epochs
    "LEARNING_RATE":1e-4,          # learning rate
    "MAX_SOURCE_TEXT_LENGTH":64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH":32,   # max length of target text
    "SEED": 42                     # set seed for reproducibility

}

T5Trainer(dataframe=df, model_params=model_params, device=device, output_dir=args.output_dir)