import pandas as pd
from argparse import ArgumentParser
from torch import cuda
import torch
import numpy as np
from transformers import AutoModelWithLMHead, AutoTokenizer
from torch.utils.data import DataLoader
from data.dataset import ProductLabelsDataset
import os
from src.trainer import train, validate

parser = ArgumentParser()

parser.add_argument("--data-file", type=str, required=True)
parser.add_argument("--output-dir", type=str, default="./output/")
parser.add_argument("--model", type=str, default="t5-base")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--step_size", type=int, default=5)

parser.add_argument("--epochs", type=int, default=20)

args = parser.parse_args()

df = pd.read_csv(args.data_file)
print(df.sample(10))

def run(params):

    device = 'cuda' if cuda.is_available() else 'cpu'

    model_params={
        "MODEL": args.model,            
        "TRAIN_BATCH_SIZE":8,         
        "VALID_BATCH_SIZE":8,          
        "TRAIN_EPOCHS":args.epochs,             
        "VAL_EPOCHS":1,               
        "LEARNING_RATE":params["lr"],        
        "MAX_SOURCE_TEXT_LENGTH":64, 
        "MAX_TARGET_TEXT_LENGTH":32,  
        "SEED": args.seed,
        "FACTOR": params["factor"],
        "STEP_SIZE": params["step_size"]
    }


    torch.manual_seed(model_params["SEED"]) # pytorch random seed
    np.random.seed(model_params["SEED"]) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    print(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = AutoTokenizer.from_pretrained(model_params["MODEL"])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = AutoModelWithLMHead.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    print(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    print(df.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation. 
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state = model_params["SEED"])
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print(f"FULL Dataset: {df.shape}")
    print(f"TRAIN Dataset: {train_dataset.shape}")
    print(f"TEST Dataset: {val_dataset.shape}\n")


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = ProductLabelsDataset(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"])
    val_set = ProductLabelsDataset(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"])


    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': model_params["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        'num_workers': 0
        }


    val_params = {
        'batch_size': model_params["VALID_BATCH_SIZE"],
        'shuffle': False,
        'num_workers': 0
        }


    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_params["LEARNING_RATE"])

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        gamma=model_params["FACTOR"],
        step_size = model_params["STEP_SIZE"]
    )
    # Training loop
    print(f'[Initiating Fine Tuning]...\n')

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer, scheduler)

    print(f"[Saving Model]...\n")
    #Saving the model after training
    path = os.path.join(args.output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


    # evaluating test dataset
    print(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv(os.path.join(args.output_dir,'predictions.csv'))

    print(f"[Validation Completed.]\n")
    print(f"""[Model] Model saved @ {os.path.join(args.output_dir, "model_files")}\n""")
    print(f"""[Validation] Generation on Validation data saved @ {os.path.join(args.output_dir,'predictions.csv')}\n""")


params = {
    "model_name": args.model,
    "lr": args.lr,
    "factor": args.factor,
    "step_size": args.step_size
}

run(params=params)