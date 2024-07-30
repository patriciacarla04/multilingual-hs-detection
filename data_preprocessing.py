
import pandas as pd
import re
import argparse


class PreprocessingPipeline:

    def __init__(self):
        self.label_transform = {
            "0": 0,
            "0. appropriate": 0, 
            "0. appropriato": 0, 
            "0 ni sporni govor": 0,
            "1. inappropriate": 1, 
            "1. inappropriato": 1, 
            "1 nespodobni govor": 1,
            "2. offensive": 2, 
            "2. offensivo": 2, 
            "2 žalitev": 2,
            "3. violent": 3, 
            "3. violento": 3, 
            "3 nasilje": 3
        }
    
    def str2num(self, label):
        if pd.isnull(label):
            return -1
        return self.label_transform.get(label, -1)
    
    def clean_text(self, text):
        text = text.lower()
        username_placeholder = "username{}"
        url_placeholder = "url{}"
        username_pattern = r'@\w+'
        url_pattern = r'https?://\S+'
        usernames = re.findall(username_pattern, text)
        urls = re.findall(url_pattern, text)
        for i, username in enumerate(set(usernames), start=1):
            text = text.replace(username, username_placeholder.format(i))
        for i, url in enumerate(set(urls), start=1):
            text = text.replace(url, url_placeholder.format(i))
        return text
    
    def remove_instances_without_data(self, df, subsets):
        begin = len(df)
        for sub in subsets:
            df = df.dropna(subset=[sub])
        print("Removed instances: " + str(begin - len(df)))
        return df
    
    def preprocess(self, data_path, file_name, subsets, sep=","):
        # Load the dataset
        df = pd.read_csv(data_path, sep=sep)
        
        # Remove instances without data
        df = self.remove_instances_without_data(df, subsets)
        
        # Apply transformations and cleaning
        df[subsets[1]] = df[subsets[1]].apply(self.str2num)
        df[subsets[0]] = df[subsets[0]].apply(self.clean_text)
        
        # Save the cleaned and preprocessed dataset
        df.to_csv(file_name, sep=sep, index=False)
        
        return df
    
def main(data_path, file_name, subsets, sep):
    preprocessor = PreprocessingPipeline()
    subsets_list = subsets.split(',')  # Assuming subsets are passed as a comma-separated string
    df_preprocessed = preprocessor.preprocess(data_path, file_name, subsets_list, sep)
    print(f"Preprocessing complete. Cleaned data saved to {file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data by removing instances without data, transforming labels, and cleaning text.')
    parser.add_argument('data_path', type=str, help='Path to the raw dataset file')
    parser.add_argument('file_name', type=str, help='File name for saving the cleaned and preprocessed dataset')
    parser.add_argument('subsets', type=str, help='Comma-separated list of column names to check for empty cells')
    parser.add_argument('--sep', type=str, default=',', help='Separator used in the dataset (default: ",")')
    
    args = parser.parse_args()
    
    main(args.data_path, args.file_name, args.subsets, args.sep)

#Example usage
#python data_preprocessing.py /Users/patriciagrigor/Desktop/MASTER/CEU_hate/multilingual-hate-speech/Data/IMSyPP_EN YouTube/IMSyPP_EN_YouTube_comments_evaluation_no_context.csv, /Users/patriciagrigor/Desktop/MASTER/CEU_hate/multilingual-hate-speech/EN_eval_clean.csv, "Text, Type"