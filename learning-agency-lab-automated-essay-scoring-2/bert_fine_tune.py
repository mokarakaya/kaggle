import logging
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim

from util import get_device

device = get_device()
print(f"device: {device}")

is_on_kaggle = False

if is_on_kaggle:
    LOCAL_MODEL_PATH = '/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor (2)/local_bert_base_uncased_model'
    BATCH_SIZE = 16
    MAX_LEN = 256
    model_path = r"/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor/bert_regressor.pth"
    train_data_path  = r"/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
    test_data_path = r"/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    TOKENIZER_PATH = r"/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor (2)/local_bert_base_uncased_tokenizer"
    output_file_path = r"/kaggle/working/submission.csv"
else:
    LOCAL_MODEL_PATH = '../bert_base_uncased_model.pkl'
    BATCH_SIZE = 16
    MAX_LEN = 256
    model_path = r"data/bert_regressor.pth"
    train_data_path = "data/train.csv"
    test_data_path = r"data/test.csv"
    TOKENIZER_PATH = r"bert_base_uncased_tokenizer"
    output_file_path = r"../submission.csv"

class EssayDataset(Dataset):
    """ Custom Dataset class for essays """

    def __init__(self, tokenizer, essays, max_length, labels):
        self.tokenizer = tokenizer
        self.texts = essays
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }, self.labels[idx]


class BertRegressor(nn.Module):
    """ BERT Model for Regression Tasks """

    def __init__(self, pre_trained_model_name):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)  # Use 'out' to match the state dict

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.out(pooled_output)

model_name = "google-bert/bert-base-uncased"
model = BertRegressor(model_name).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)
df = pd.read_csv(train_data_path)
X = df[[ 'full_text']]
y = df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = EssayDataset(tokenizer, list(X_train["full_text"]), MAX_LEN, list(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

running_loss = 0.
last_loss = 0.

loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Here, we use enumerate(training_loader) instead of
# iter(training_loader) so that we can track the batch
# index and do some intra-epoch reporting
for i, data in tqdm(enumerate(train_dataloader)):
    # Every data instance is an input + label pair
    inputs, labels = data
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels.to(device)

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    outputs = model(**inputs)

    # Compute the loss and its gradients
    loss = loss_fn(outputs.reshape(-1), labels.type(torch.float32))
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    # Gather data and report
    running_loss += loss.item()
    if i % 1000 == 999:
        last_loss = running_loss / 1000  # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        running_loss = 0.

torch.save(model.state_dict(), LOCAL_MODEL_PATH)

# Parameters
# model = BertRegressor(LOCAL_MODEL_PATH)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()
#
# # Load Data
# test_df = pd.read_csv(test_data_path)
#
# # Data Processing
# tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
# test_dataset = EssayDataset(tokenizer, test_df['full_text'].tolist(), MAX_LEN)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
# # Prediction
# predictions = []
# with torch.no_grad():
#     for batch in test_dataloader:
#         output = model(**batch)
#         predictions.extend(output.flatten().tolist())
# test_df['score'] = predictions
# test_df = test_df.drop(columns=['full_text'])
# test_df['score'] = test_df['score'].round(0).astype(int)
# # Save Results
# test_df.to_csv(output_file_path, index=False)
#
#
# test_df['score'] = test_df['score'].round(0).astype(int)
# submission = test_df
