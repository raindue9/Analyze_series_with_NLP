import spacy
import os
import sys
import pathlib
import pandas as pd
from utils import load_subtitles_dataset
from nltk import sent_tokenize
from ast import literal_eval

folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))


class NamedEntityRecognizer:
    def __init__(self):
        # spacy.require_gpu()
        # print("spaCy using GPU:", spacy.prefer_gpu())
        self.nlp_model = self.load_model()
        pass

    def load_model(self):
        nlp = spacy.load('en_core_web_trf')
        return nlp

    def get_ners_inference(self, script):
        script_sentences = sent_tokenize(script)

        ner_output = []

        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text
                    first_name = full_name.split(" ")[0]
                    first_name = first_name.strip()
                    ners.add(first_name)
            ner_output.append(ners)

        return ner_output

    def get_ners(self, dataset_path, save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df

        # load dataset
        df = load_subtitles_dataset(dataset_path)
        # 缺少GPU 则只设置10
        # df = df.head(10)
        # Run Inferences
        df['ners'] = df['script'].apply(self.get_ners_inference)

        if save_path is not None:
            df.to_csv(save_path, index=False)

        return df


