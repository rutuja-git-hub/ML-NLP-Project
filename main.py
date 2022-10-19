from flask import Flask, render_template, request
import time
from sentence_transformers import util
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from sentence_transformers import SentenceTransformer
import torch
import output
save_path = 'embeddings.pt'

path = 'due_diligence.csv'
due_diligence = pd.read_csv(path)

file_name = []
request1 = []
department=[]
all_data=[]

with open(path, encoding='utf8') as fIn:
    for file_name_col, request_col, department_col in due_diligence.itertuples(index=False):
        file_name.append(str(file_name_col))
        request1.append(str(request_col))
        department.append(str(department_col))
        all_data.append(str(request_col))

app = Flask(__name__)
model = SentenceTransformer('output/training_stsbenchmark_-content-drive-MyDrive-output_from_faribas_notebook-train_mlm-output-sentence-transformers_bert-base-nli-mean-tokens-2022-07-12_08-45-38-2022-07-12_14-28-25')
corpus_embeddings = model.encode(all_data, show_progress_bar=True, convert_to_tensor=True)
torch.save(corpus_embeddings,save_path)
@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    inp_question = request.form['text']
    label=[]
    question=[]
    variable=[]
    file_used=[]
    depart=[]
    start_time = time.time()
    question_embedding = model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding,  torch.load(save_path))    
    end_time = time.time()
    hits = hits[0]
    print("Input question:", inp_question)
    print("Results (after {:.3f} seconds):".format(end_time-start_time))
    print("Number of possible duplicates:", len(hits)) # Print the number of possible duplicates

    # hits=[{'corpus_id': 5945, 'score': 0.6760969161987305}, {'corpus_id': 1909, 'score': 0.6732248663902283}, {'corpus_id': 1879, 'score': 0.6581871509552002}, {'corpus_id': 3491, 'score': 0.6230702996253967}, {'corpus_id': 5583, 'score': 0.6197046041488647}, {'corpus_id': 1272, 'score': 0.6162281632423401}, {'corpus_id': 6237, 'score': 0.5765202641487122}, {'corpus_id': 6156, 'score': 0.572158694267273}, {'corpus_id': 4980, 'score': 0.5668321847915649}, {'corpus_id': 2155, 'score': 0.5629119873046875}]
    for hit in hits[0:10]:
        # if hit['score'] > 0.89: 
        label.clear()
        question.clear()
        file_used.clear()
        depart.clear()

        score_1="{:.2f}".format(hit['score'])
        label.append(str(score_1))
        question.append(request1[hit['corpus_id']])
        file_used.append(file_name[hit['corpus_id']])
        depart.append(department[hit['corpus_id']])
        variable.append(label+question+file_used+depart)
    print(variable)
    return(render_template('index.html', variable=variable,input_question=inp_question))

if __name__ == "__main__":
    app.run(port='8088',threaded=False)