from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import re
import json
import math



class ChatBot:
    def __init__(self, name):
        self.name = name
        self.file_path = os.path.join(os.getcwd(), 'jeopardy.json')
        self.inp_tokens = []
        self.corpus_tokens = []
        self.model = None

        self.get_file(self.file_path)

    def __repr__(self):
        return "Hello! I'm " + self.name + ", a question answering bot who knows answers to all the questions from the 'Jeopardy!' game."

    def train_model(self):
        self.model = Doc2Vec(
            documents=self.corpus_tokens,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            epochs=15
        )

    def get_most_similar_doc(self):
        inferred_vector = self.model.infer_vector(self.inp_tokens)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=1)
        return sims[0]


    def greet(self):
        print('Ask me something!')
        self.process_input()
        print("Let's play!")
        doc_id, similarity_score = self.get_most_similar_doc()
        print(f"I know this question: its number is {doc_id}. I'm {math.floor(100 * similarity_score)}% sure of this.")
        with open(self.file_path, 'r') as f:
            data = json.load(f)
            print(f"{data[int(doc_id)]['answer']}")
        print('Do you want to ask me again? (yes/no)')
        if input() == 'no':
            print('It was nice to play with you! Goodbye!')
            exit()
        else:
            self.greet()


    def process_input(self):
        inq = input()
        self.inp_tokens = self.tokenize(inq)

    def delete_punct(self, sentence):
        sentence = [x for x in sentence if re.search(r'(\w|\d)+', x)]
        return sentence

    def tokenize(self, inp):
        tokens = self.delete_punct(word_tokenize(str(inp).lower()))
        return tokens

    def get_file(self, file_path):
        with open(file_path, 'r') as f:
            for id, line in enumerate(json.load(f)):
                tokens = self.tokenize(line['question'])
                self.corpus_tokens.append(TaggedDocument(words=tokens, tags=[id]))



chatbot = ChatBot('Bot')
chatbot.train_model()
print(chatbot)
chatbot.greet()

