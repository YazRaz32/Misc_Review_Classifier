import pandas as pd
import numpy as np
from training import *
from testing import *
import os

# Save the model weights and configuration
def save_model(model, tokenizer, output_dir='sentiment_model'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)

# Load the model
def load_model(model_dir='sentiment_model'):
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    return model, tokenizer



def compute(data, model, tokenizer):
  tokenized = tokenizer(
            data,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenized = {k: v.to(device) for k, v in tokenized.items()}

  model.eval()

  with torch.no_grad():  # Disable gradient calculations during inference
      outputs = model(**tokenized)  
      logits = outputs.logits  
      predicted_class = torch.argmax(logits, dim=1) 

  print('Positive' if predicted_class.tolist()[0] else 'Negative')  # Print the predicted class index (0 or 1)




if __name__ == '__main__':
    
    df = pd.read_csv('imdb_dataset.csv', engine = 'python')

    # save_model(model, tokenizer, 'imdb_sentiment_model')

    model, tokenizer = load_model('Include/imdb_sentiment_model')

    # model, tokenizer, val_dataloader  = prepare_and_train(df[0:3000])

    # evaluate_model(model, val_dataloader, tokenizer)

    while True:
      compute(input(), model, tokenizer)

   
