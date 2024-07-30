from datasets import Dataset
from datasets import ClassLabel
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import random
import argparse
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import os
import torch
import time
import os
from torch.utils.data import DataLoader, TensorDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#Function for loading model and optimizer
def load_model_and_optimizer(model_class, tokenizer_path, model_path, optimizer_class, optimizer_path, lr):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = model_class.from_pretrained(model_path, num_labels=4)
    optimizer = optimizer_class(model.parameters(), lr=lr)
    optimizer_state = torch.load(optimizer_path)
    optimizer.load_state_dict(optimizer_state)
    return model, tokenizer, optimizer

#Function for saving model and optimizer
def save_model_and_optimizer(model, tokenizer, optimizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

#Custom trainer class that allows for weighted loss functions
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert class weights to a tensor
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate loss
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Train an XLM-R model on a classification task.")
    parser.add_argument("--train_data", required=True, help="Path to the training dataset CSV file.")
    parser.add_argument("--test_data_en", required=True, help="Path to the EN test dataset CSV file.")
    parser.add_argument("--test_data_it", required=True, help="Path to the IT test dataset CSV file.")
    parser.add_argument("--test_data_slo", required=True, help="Path to the SLO test dataset CSV file.")
    parser.add_argument("--output_dir", help="Directory to save the trained model.", default="./model_output")
    parser.add_argument("--results_dir", help="Directory to save the results files.", default="./results")
    parser.add_argument("--batch_size", type=int, help="Batch size for training and evaluation.", default=8)
    parser.add_argument("--epochs", type=int, help="Number of training epochs.", default=3)

    args = parser.parse_args()
    
    input_dir = "/content/drive/MyDrive/multilingual-hate-speech/model_duplicate_all_NORMAL_1epoch/model"
    
    #Loading model, tokenizer, and optimizer
    model, tokenizer, optimizer = load_model_and_optimizer(
        AutoModelForSequenceClassification, input_dir, input_dir,
        torch.optim.AdamW, os.path.join(input_dir, "optimizer.pt"), 6e-6
    )

    training_args = TrainingArguments(output_dir=args.output_dir, evaluation_strategy='epoch',
                                      per_device_train_batch_size=args.batch_size,
                                      num_train_epochs=args.epochs, learning_rate=6e-6)
    
    #tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    #model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inializing optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    #optimizer.load_state_dict(torch.load(os.path.join('/content/drive/MyDrive/multilingual-hate-speech/model_duplicate_all_NORMAL_1epoch/model', "optimizer.pt")))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    df_train = pd.read_csv(args.train_data)
    df_hug = pd.DataFrame()
    df_hug['label'] = df_train['Type']
    df_hug['text'] = df_train['Text']
    dataset = Dataset.from_pandas(df_hug)
    orientation = ClassLabel(num_classes=4, names=["Appropriate", "Inappropriate", "Offensive", "Violent"])
    dataset = dataset.cast_column("label", orientation)

   # Splitting dataset into train and validation sets
    train_dataset, eval_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()

    #tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

'''
#Weighted trainer class
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        class_weights=[0.021, 0.371, 0.036, 0.572] 
    )
'''

#Regular trainer class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset
    )


    # Start timing
    start_time = time.time()

    trainer.train()

    # End timing
    end_time = time.time()
    print("Training time: " + str(end_time - start_time))

    # Saving model and tokenizer
    save_model_and_optimizer(model, tokenizer, optimizer, args.output_dir)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    df_test_en = pd.read_csv(args.test_data_en)
    df_test_it = pd.read_csv(args.test_data_it)
    df_test_slo = pd.read_csv(args.test_data_slo)

    # English test set

    df_hug_en = pd.DataFrame()
    df_hug_en['label'] = df_test_en['Type']
    df_hug_en['text'] = df_test_en['Text']

    text_list = df_hug_en['text'].tolist() 
    tokenized_test_set = tokenizer(text_list, padding="max_length",
                               truncation=True, return_tensors="pt")
    
    # Convert tokenized data into a TensorDataset
    test_dataset = TensorDataset(tokenized_test_set['input_ids'], tokenized_test_set['attention_mask'])

    # Create a DataLoader for handling batches
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Prediction in batches
    all_logits = []
    all_predictions = []
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            batch_input_ids, batch_attention_mask = batch[0].to(device), batch[1].to(device)
            outputs = model(batch_input_ids, batch_attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

            # Move logits and predictions back to CPU for storage/analysis
            all_logits.append(logits.cpu())
            all_predictions.append(predictions.cpu())

    # Concatenate all collected batches
    logits = torch.cat(all_logits, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
                               
    predictions = predictions.cpu().numpy()

    #print("Predictions:", predictions)

    cm = confusion_matrix(df_hug_en['label'], predictions)
    print(cm)
    print('Accuracy:' + str(accuracy_score(df_hug_en['label'], predictions)))
    print('Micro F1 score:' + str(f1_score(df_hug_en['label'], predictions, average='micro')))
    print('Macro F1 score:' + str(f1_score(df_hug_en['label'], predictions, average='macro')))
    f1_scores = f1_score(df_hug_en['label'], predictions, average=None)
    print("F1 '0 appropriate: '" + str(f1_scores[0]))
    print("F1 '1 inappropriate: '" + str(f1_scores[1]))
    print("F1 '2 offensive: '" + str(f1_scores[2]))
    print("F1 '3 violent: '" + str(f1_scores[3]))
    print('Recall violent class:' + str(recall_score(df_hug_en['label'], predictions, labels=[3], average=None)[0]))

    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(df_hug_en['label'], predictions),
        'micro_f1_score': f1_score(df_hug_en['label'], predictions, average='micro'),
        'macro_f1_score': f1_score(df_hug_en['label'], predictions, average='macro'),
        'f1_scores': {
            '0_appropriate': f1_scores[0],
            '1_inappropriate': f1_scores[1],
            '2_offensive': f1_scores[2],
            '3_violent': f1_scores[3]
        },
        'recall_3_violent': recall_score(df_hug_en['label'], predictions, labels=[3], average=None)[0]
    }

    # Save metrics to a pickle file in the specified directory
    metrics_file_path = os.path.join(args.results_dir, 'metrics_xlmr_en.pkl')
    with open(metrics_file_path, 'wb') as file:
        pickle.dump(metrics, file)

    # Save predictions to a pickle file in the specified directory
    metrics_file_path = os.path.join(args.results_dir, 'predictions_xlmr_en.pkl')
    with open(metrics_file_path, 'wb') as file:
        pickle.dump(predictions, file)

if __name__ == "__main__":
    main()
'''
# Italian test set

    df_hug_it = pd.DataFrame()
    df_hug_it['label'] = df_test_it['Tipo']
    df_hug_it['text'] = df_test_it['Testo']
    # test_dataset = Dataset.from_pandas(df_hug_en)
    # orientation = ClassLabel(num_classes=4, names=["Appropriate", "Inappropriate", "Offensive", "Violent"])
    # test_dataset = test_dataset.cast_column("label", orientation)

    text_list = df_hug_it['text'].tolist()  # Ensure it is a list of strings
    tokenized_test_set = tokenizer(text_list, padding="max_length",
                               truncation=True, return_tensors="pt")

    # Prepare dataset for the model
    input_ids = tokenized_test_set["input_ids"].to(device)
    attention_mask = tokenized_test_set["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get logits
    logits = outputs.logits

    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the most probable class for each input
    predictions = torch.argmax(probabilities, dim=-1)

    # Convert to numpy arrays for easier manipulation if needed
    probabilities = probabilities.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Now `probabilities` contains the probabilities of each class for each input
    # and `predictions` contains the index of the most probable class for each input
    print("Probabilities:", probabilities)
    print("Predictions:", predictions)

    cm = confusion_matrix(df_hug_it['label'], predictions)
    print(cm)
    print('Accuracy:' + str(accuracy_score(df_hug_it['label'], predictions)))
    print('Micro F1 score:' + str(f1_score(df_hug_it['label'], predictions, average='micro')))
    print('Macro F1 score:' + str(f1_score(df_hug_it['label'], predictions, average='macro')))
    f1_scores = f1_score(df_hug_it['label'], predictions, average=None)
    print("F1 '0 appropriate: '" + str(f1_scores[0]))
    print("F1 '1 inappropriate: '" + str(f1_scores[1]))
    print("F1 '2 offensive: '" + str(f1_scores[2]))
    print("F1 '3 violent: '" + str(f1_scores[3]))
    print('Recall violent class:' + str(recall_score(df_hug_it['label'], predictions, labels=[3], average=None)[0]))

    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(df_hug_it['label'], predictions),
        'micro_f1_score': f1_score(df_hug_it['label'], predictions, average='micro'),
        'macro_f1_score': f1_score(df_hug_it['label'], predictions, average='macro'),
        'f1_scores': {
            '0_appropriate': f1_scores[0],
            '1_inappropriate': f1_scores[1],
            '2_offensive': f1_scores[2],
            '3_violent': f1_scores[3]
        },
        'recall_3_violent': recall_score(df_hug_it['label'], predictions, labels=[3], average=None)[0]
    }

    # Save metrics to a pickle file in the specified directory
    metrics_file_path = os.path.join(args.results_dir, 'metrics_xlmr_it.pkl')
    with open(metrics_file_path, 'wb') as file:
        pickle.dump(metrics, file)

    # Save predictions to a pickle file in the specified directory
    metrics_file_path = os.path.join(args.results_dir, 'predictions_xlmr_it.pkl')
    with open(metrics_file_path, 'wb') as file:
        pickle.dump(predictions, file)

# Slovenian test set

    df_hug_slo = pd.DataFrame()
    df_hug_slo['label'] = df_test_slo['vrsta']
    df_hug_slo['text'] = df_test_slo['besedilo']
    # test_dataset = Dataset.from_pandas(df_hug_en)
    # orientation = ClassLabel(num_classes=4, names=["Appropriate", "Inappropriate", "Offensive", "Violent"])
    # test_dataset = test_dataset.cast_column("label", orientation)

    text_list = df_hug_slo['text'].tolist()  # Ensure it is a list of strings
    tokenized_test_set = tokenizer(text_list, padding="max_length",
                               truncation=True, return_tensors="pt")

    # Prepare dataset for the model
    input_ids = tokenized_test_set["input_ids"].to(device)
    attention_mask = tokenized_test_set["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get logits
    logits = outputs.logits

    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the most probable class for each input
    predictions = torch.argmax(probabilities, dim=-1)

    # Convert to numpy arrays for easier manipulation if needed
    probabilities = probabilities.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Now `probabilities` contains the probabilities of each class for each input
    # and `predictions` contains the index of the most probable class for each input
    print("Probabilities:", probabilities)
    print("Predictions:", predictions)

    cm = confusion_matrix(df_hug_slo['label'], predictions)
    print(cm)
    print('Accuracy:' + str(accuracy_score(df_hug_slo['label'], predictions)))
    print('Micro F1 score:' + str(f1_score(df_hug_slo['label'], predictions, average='micro')))
    print('Macro F1 score:' + str(f1_score(df_hug_slo['label'], predictions, average='macro')))
    f1_scores = f1_score(df_hug_slo['label'], predictions, average=None)
    print("F1 '0 appropriate: '" + str(f1_scores[0]))
    print("F1 '1 inappropriate: '" + str(f1_scores[1]))
    print("F1 '2 offensive: '" + str(f1_scores[2]))
    print("F1 '3 violent: '" + str(f1_scores[3]))
    print('Recall violent class:' + str(recall_score(df_hug_slo['label'], predictions, labels=[3], average=None)[0]))

    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(df_hug_slo['label'], predictions),
        'micro_f1_score': f1_score(df_hug_slo['label'], predictions, average='micro'),
        'macro_f1_score': f1_score(df_hug_slo['label'], predictions, average='macro'),
        'f1_scores': {
            '0_appropriate': f1_scores[0],
            '1_inappropriate': f1_scores[1],
            '2_offensive': f1_scores[2],
            '3_violent': f1_scores[3]
        },
        'recall_3_violent': recall_score(df_hug_slo['label'], predictions, labels=[3], average=None)[0]
    }

    # Save metrics to a pickle file in the specified directory
    metrics_file_path = os.path.join(args.results_dir, 'metrics_xlmr_slo.pkl')
    with open(metrics_file_path, 'wb') as file:
        pickle.dump(metrics, file)

    # Save predictions to a pickle file in the specified directory
    metrics_file_path = os.path.join(args.results_dir, 'predictions_xlmr_slo.pkl')
    with open(metrics_file_path, 'wb') as file:
        pickle.dump(predictions, file)

'''
