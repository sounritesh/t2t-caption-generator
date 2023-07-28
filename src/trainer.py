import torch
import numpy as np
from data.dataset import ProductLabelsDataset
import os
import pandas as pd

def train(epoch, tokenizer, model, device, loader, optimizer, scheduler):
    model.train()

    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        try:
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
        except Exception as e:
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                labels=lm_labels,
            )
        loss = outputs[0]

        if _ % 10 == 0:
            print("Epoch {}: {}".format(str(epoch), str(loss)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


def validate(epoch, tokenizer, model, device, loader, params):
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_new_tokens=params["max_length"], 
              num_beams=params["num_beams"],
              repetition_penalty=params["repetition_penalty"], 
              length_penalty=params["length_penalty"], 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%10==0:
              print("Completed: {}".format(_))

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals