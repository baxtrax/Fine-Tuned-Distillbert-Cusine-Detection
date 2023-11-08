from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertModel
import wandb

classes = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian',
 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole',
 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']

ROOT_DIR = str(Path(__file__).resolve().parent) + '/whats-cooking/'
TRAIN = False
TEST = False

def main():
    # Setup wandb for experiment tracking
    wandb.login()
    wandb.init(project='whats-cooking')

    # Load the dataset
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    dataset   = CookingDatasetJSON(file_path=(ROOT_DIR + 'train/train.json'), 
                                   tokenizer=tokenizer, 
                                   max_length=100)

    # Create dataloaders
    train_loader, test_loader, valid_loader = create_dataloaders(dataset, 
                                                                 batch_size=(32,32,32))

    # Maybe use this in the future? Didnt seem to help much accruacy wise
    # DISTILBERT_DROPOUT = 0.2
    # DISTILBERT_ATT_DROPOUT = 0.21
 
    # # Configure DistilBERT's initialization
    # config = DistilBertConfig(dropout=DISTILBERT_DROPOUT, 
    #                         attention_dropout=DISTILBERT_ATT_DROPOUT, 
    #                         output_hidden_states=True)

    # Instantiate the model
    model = DistillBERTModified()

    # Train the model
    if TRAIN:
        model = finetune(epochs=10, 
                        train_loader=train_loader, 
                        valid_loader=valid_loader,
                        model=model, 
                        loss_fn=nn.CrossEntropyLoss(), 
                        optimizer=optim.Adam(model.parameters(), lr= 5e-5),
                        device='cpu' if not torch.cuda.is_available() else 'cuda')
    
    if TEST:
    # Load the bset model
        model.load_state_dict(torch.load('model.pth'))

        # Test the model
        test(model=model, 
            test_loader=test_loader, 
            device='cpu' if not torch.cuda.is_available() else 'cuda')

def finetune(epochs, train_loader, valid_loader, model, loss_fn, optimizer, device):
    """
    Finetunes the model on the training set and validates on the validation set.

    Params:
        epochs (int): Number of epochs to train for.
        train_loader (obj): Dataloader object containing the training set.
        valid_loader (obj): Dataloader object containing the validation set.
        model (obj): Model to train.
        loss_fn (obj): Loss function to use.
        optimizer (obj): Optimizer to use.
        device (str): Device to train on.

    Returns:
        model (obj): Trained model.
    """
    model.to(device)
    best_val_accuracy = 0.0
    
    # Train and validate the model each epoch
    for epoch in range(epochs):
        # Train the model
        train_loop = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
        train_loop.set_description(f'Epoch={epoch}/{epochs}, Training...')

        wandb.log({'epoch': epoch})  # Log the epoch number

        model.train()
        for batch, data in train_loop:
            ids     = data['ids'].to(device, dtype=torch.long)
            mask    = data['mask'].to(device, dtype=torch.long)
            targets = data['target'].to(device, dtype=torch.long)

            # Forward pass and backpropagation
            optimizer.zero_grad()
            
            output = model(ids=ids, mask=mask)
            loss = loss_fn(output, targets)
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, pred     = torch.max(output, dim=1)
            num_correct = torch.sum(pred == targets)
            num_samples = pred.size(0)
            accuracy    = float(num_correct)/float(num_samples)

            # Show progress while training
            train_loop.set_postfix(loss=loss.item(), acc=accuracy)
            wandb.log({'train_loss': loss.item(), 'train_accuracy': accuracy})

        
        # Check accuracy on validation set
        valid_loop = tqdm(enumerate(valid_loader), leave=False, total=len(valid_loader))
        valid_loop.set_description(f'Epoch={epoch}/{epochs}, Validating...')
        
        model.eval()
        for batch, data in valid_loop:
            ids     = data['ids'].to(device, dtype=torch.long)
            mask    = data['mask'].to(device, dtype=torch.long)
            targets = data['target'].to(device, dtype=torch.long)

            # Forward pass
            output = model(ids=ids, mask=mask)
            loss   = loss_fn(output, targets)
            
            # Calculate accuracy
            _, pred     = torch.max(output, dim=1)
            num_correct = torch.sum(pred == targets)
            num_samples = pred.size(0)
            accuracy    = float(num_correct)/float(num_samples)

            # Show progress while training
            valid_loop.set_postfix(loss=loss.item(), acc=accuracy)
            wandb.log({'valid_loss': loss.item(), 'valid_accuracy': accuracy})

            # Save the model if it is the best one yet
            if (accuracy > best_val_accuracy):
                best_val_accuracy = accuracy
                torch.save(model.state_dict(), 'model.pth')

    return model

def test(model, test_loader, device):
    """
    Tests the model on the test set.

    Params:
        model (obj): Model to test.
        test_loader (obj): Dataloader object containing the test set.
        device (str): Device to test on.
    """
    # Check accuracy on test set
    accuracies = []
    test_loop = tqdm(enumerate(test_loader), leave=False, total=len(test_loader))
    test_loop.set_description(f'Testing...')

    model.eval()
    for batch, data in test_loop:
        ids     = data['ids'].to(device, dtype=torch.long)
        mask    = data['mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        # Forward pass
        output = model(ids=ids, mask=mask)
        
        # Calculate accuracy
        _, pred     = torch.max(output, dim=1)
        num_correct = torch.sum(pred == targets)
        num_samples = pred.size(0)
        accuracy    = float(num_correct)/float(num_samples)

        # Show progress while training
        wandb.log({'test_accuracy': accuracy})
        test_loop.set_postfix(acc=accuracy)
        accuracies.append(accuracy)
    
    print(f'Average test accuracy: {np.mean(accuracies)}')

def create_dataloaders(dataset, train_size=0.7, valid_size=0.2, batch_size=(32,32,32)):
    """
    Splits the dataset into train, validation, and test sets.
    Test set is the remaining percentage of the dataset.

    Params:
        dataset (obj): Dataset object to split.
        train_size (float): Percentage of the dataset to use for training.
        valid_size (float): Percentage of the dataset to use for validation.

    Returns:
        train_dataset (obj): Dataset object containing the training set.
        valid_dataset (obj): Dataset object containing the validation set.
        test_dataset (obj): Dataset object containing the test set.
    """
    # Define split
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    valid_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - valid_size

    # Split the dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size[0])
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size[1])
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size[2])

    return train_dataloader, valid_dataloader, test_dataloader

class CookingDatasetJSON(Dataset):
    """
    Whats Cooking Contest Dataset.

    Params:
        file_path (str): Path to the JSON file containing the data.
        tokenizer (obj): Tokenizer object from the transformers library.
        max_length (int): Maximum length of the tokenized input sentence.
    """
    def __init__(self, file_path, tokenizer, max_length):
        super(CookingDatasetJSON, self).__init__()

        # Read in the JSON file
        df = pd.read_json(file_path).drop(columns=['id'])
        self.data_json = df[['ingredients', 'cuisine']] # Swap the columns so target on right

        # For each row in ingredients, join all the ingredients into one string, with a colon in between
        self.data_json['ingredients'] = self.data_json['ingredients'].apply(lambda x: ', '.join(x))

        # Convert all the cuisine names to their numerical class index, given the above 
        self.data_json['cuisine'] = self.data_json['cuisine'].apply(lambda x: classes.index(x))
                
        self.tokenizer=tokenizer
        self.target=self.data_json.iloc[:,1]
        self.max_length=max_length
        
    def __len__(self):
        return len(self.data_json)
    
    def __getitem__(self, index):
        text = self.data_json.iloc[index,0]
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            add_special_tokens=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target': torch.tensor(self.data_json.iloc[index, 1], dtype=torch.long)
        }

class DistillBERTModified(torch.nn.Module):
    def __init__(self, distill_bert_config=None):
        super().__init__()
        self.distill_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Apply config if provided
        if distill_bert_config is not None:
            self.distill_bert = self.distill_bert(distill_bert_config)

        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(768, len(classes))
    
    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output = self.out(output_1)
        return output

if __name__ == '__main__':
    main()