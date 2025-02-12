from sklearn.model_selection import train_test_split
from data_preprocessing import *

class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            reviews.tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor([1 if label == 'positive' else 0 for label in labels])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def train_model(train_dataloader, model, epochs=6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

# Main execution
def prepare_and_train(df, test_size=0.2, batch_size=32):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review'].values, 
        df['sentiment'].values,
        test_size=test_size
    )
    

    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer)
    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    

    train_model(train_dataloader, model)
    
    return model, tokenizer, val_dataloader

