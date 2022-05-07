import numpy as np
import pandas as pd
from utils import *
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
#%%
data_df = get_input_df()
data_df = data_df.sample(frac=0.001)
#%%
data_df.head()
#%%
data_df = data_df.iloc[:len(data_df)//10,:]
unsup_df = pd.read_csv("augmented_data.csv")
#%%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data_df.index.values,
                                                data_df.Target.values,
                                                test_size = 0.1,
                                                random_state=5,
                                                stratify = data_df.Target.values)
#%%

#%%
data_df.loc[X_train,'data_type'] = 'train'
data_df.loc[X_test,'data_type'] = 'test'
#%%
data_df.groupby(['data_type', 'Target']).count()
#%%

#%%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                         do_lower_case = True)
#%%
unsup_df.Input = unsup_df.Input.apply(lambda x: str(x))
unsup_df.Augment = unsup_df.Augment.apply(lambda x: str(x))
#%%
encoder_train = tokenizer.batch_encode_plus(data_df[data_df["data_type"]=='train'].Input.values,
                                           add_special_tokens = True,
                                            return_attention_masks = True,
                                           pad_to_max_length = True,
                                           max_length = 256,
                                           return_tensors = 'pt')

encoder_unsup_orig = tokenizer.batch_encode_plus(unsup_df["Input"].values,
                                           add_special_tokens = True,
                                            return_attention_masks = True,
                                           pad_to_max_length = True,
                                           max_length = 256,
                                           return_tensors = 'pt')


encoder_unsup_aug = tokenizer.batch_encode_plus(unsup_df["Augment"].values,
                                           add_special_tokens = True,
                                            return_attention_masks = True,
                                           pad_to_max_length = True,
                                           max_length = 256,
                                           return_tensors = 'pt')

encoder_test = tokenizer.batch_encode_plus(data_df[data_df["data_type"]=='test'].Input.values,
                                           add_special_tokens = True,
                                            return_attention_masks = True,
                                           pad_to_max_length = True,
                                           max_length = 256,
                                           return_tensors = 'pt')

input_ids_train = encoder_train['input_ids']
attention_masks_train = encoder_train["attention_mask"]
labels_train = torch.tensor(data_df[data_df['data_type']=='train'].Target.values)

orig_input_ids_train = encoder_unsup_orig['input_ids']
orig_attention_masks_train = encoder_unsup_orig["attention_mask"]

aug_input_ids_train = encoder_unsup_aug['input_ids']
aug_attention_masks_train = encoder_unsup_aug["attention_mask"]

input_ids_test = encoder_test['input_ids']
attention_masks_test = encoder_test["attention_mask"]
labels_test = torch.tensor(data_df[data_df['data_type']=='test'].Target.values)
#%%

#%%

#%%
data_train = TensorDataset(input_ids_train,attention_masks_train,labels_train)
data_test = TensorDataset(input_ids_test,attention_masks_test,labels_test)
data_augment = TensorDataset(orig_input_ids_train,
                             orig_attention_masks_train,
                             aug_input_ids_train,
                             aug_attention_masks_train)
#%%
len(data_df.Target.unique())
#%%
len(data_train),len(data_test)#%%
#%%
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                     num_labels = len(data_df.Target.unique()),
                                     output_attentions = False,
                                     output_hidden_states =  False)
#%%


from torch.utils.data import RandomSampler,SequentialSampler,DataLoader

dataloader_train = DataLoader(
    data_train,
    sampler= RandomSampler(data_train),
    batch_size = 16

)


dataloader_test = DataLoader(
    data_test,
    sampler= RandomSampler(data_test),
    batch_size = 32

)

dataloader_augment = DataLoader(
    data_augment,
    sampler= RandomSampler(data_augment),
    batch_size = 16
)
#%%
from transformers import AdamW,get_linear_schedule_with_warmup
optimizer = AdamW(model.parameters(),lr = 1e-5,eps = 1e-8)

epochs  = 1000
scheduler = get_linear_schedule_with_warmup(
            optimizer,
    num_warmup_steps = 0,
   num_training_steps = len(dataloader_train)*epochs
)
#%%
from sklearn.metrics import f1_score

def f1_score_func(preds,labels):
    preds_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat,preds_flat,average = 'weighted')
#%%
def accuracy_per_class(preds,labels):
    dict_label = {'happy':1, 'sad':0}
    label_dict_reverse = {v:k for k,v in dict_label.items()}

    preds_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f"Class:{label_dict_reverse}")
        print(f"Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n")
#%%
import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Loading:{device}")
#%%
def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions,true_vals = [],[]

    for batch in tqdm(dataloader_val):
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':  batch[0],
                  'attention_mask':batch[1],
                  'labels': batch[2]
                 }
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total +=loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)


    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions,axis=0)
    true_vals = np.concatenate(true_vals,axis=0)
    return loss_val_avg,predictions,true_vals


#%%
from tqdm.notebook import tqdm
import wandb

import torch.nn as nn
import torch.nn.functional as F


wandb.init(project="sentiment-uda", entity="similarity-based-value-smoothing")
wandb.watch(model)

kl_loss = nn.KLDivLoss(reduction="batchmean")

for epoch in tqdm(range(1,epochs+1)):
    model.train()

    loss_train_total=0

    progress_bar = tqdm(dataloader_train,desc = "Epoch: {:1d}".format(epoch),leave = False,disable = False)
    augment_iter = iter(dataloader_augment)

    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)
        aug_batch = next(augment_iter)

        orig_inp, orig_mask, aug_inp, aug_mask = aug_batch


        inputs = {
            "input_ids":batch[0],
            "attention_mask":batch[1],
            "labels":batch[2]

        }
        outputs = model(**inputs)

        orig_inputs = {
            "input_ids":orig_inp,
            "attention_mask":orig_mask
        }

        aug_inputs = {
            "input_ids":aug_inp,
            "attention_mask":aug_mask
        }

        orig_logits = model(**orig_inputs).logits
        aug_logits = model(**aug_inputs).logits


        # input should be a distribution in the log space
        unsup_input = F.log_softmax(orig_logits)
        # Sample a batch of distributions. Usually this would come from the dataset
        unsup_target = F.softmax(aug_logits)
        kld_loss = kl_loss(unsup_input, unsup_target)

        sup_loss = outputs[0]
        loss = sup_loss + kld_loss
#         logits = outputs[1]
        loss_train_total +=loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(),1.0)

        optimizer.step()
        scheduler.step()

        wandb.log({'Unsupervised Loss':'{:.3f}'.format(kld_loss/len(batch))})
        wandb.log({'Supervised Loss':'{:.3f}'.format(sup_loss/len(batch))})
        progress_bar.set_postfix({'training_loss':'{:.3f}'.format(loss.item()/len(batch))})
        wandb.log({'training_loss':'{:.3f}'.format(loss.item()/len(batch))})
#     torch.save(model.state_dict(),f'/kaggle/output/BERT_ft_epoch{epoch}.model')To save the model after each epoch

    tqdm.write('\nEpoch {epoch}')

    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training Loss: {loss_train_avg}')
    wandb.log({'Training Loss' : loss_train_avg})
    val_loss,predictions,true_vals = evaluate(dataloader_test)
    test_score = f1_score_func(predictions,true_vals)
    tqdm.write(f'Val Loss:{val_loss}\n Test Score:{test_score}')
    wandb.log({'Val Loss': val_loss, 'Test Score': test_score})
