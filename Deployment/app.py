  
import config
import torch
import flask
import time
from flask import Flask
from flask import request, render_template
from werkzeug.utils import secure_filename
from model import SentimentClassifier
import functools
import Input as im
import torch.nn as nn
import joblib
import numpy as np
from flask import Flask,render_template,request,send_from_directory,send_file,make_response,redirect

app = Flask(__name__)

MODEL = None
DEVICE = "cpu"
# f = config.MODEL_PATH
# torch.load(f,map_location=torch.device('cpu'))
PREDICTION_DICT = dict()
memory = joblib.Memory("../input/", verbose=0)
ALLOWED_EXTENSIONS = set(['csv'])
cache = {}

def allowed_file(filename):
     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')


#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['Get','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template("error.html")
        file = request.files['file']

        # if user does not select file, browser also.
        # submit a empty part without filename
        if file.filename == '':
            return render_template("error.html")





        # Check whether the upoaded file is in allowed format.
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            final_df = im.ReadFile(file,filename)
            cache["foo"] = final_df
            return render_template("predictF.html",tables=[final_df.to_html(classes='data')], titles=final_df.columns.values,message=filename)
        else:
            return render_template("error.html")


@app.route('/predictText', methods=['Get','POST'])
def upload_Text():
    text = request.form['text']
    if(text==""):
        text = "Always keep smiling, please enter your review!"
    sentiment = im.ReadText(str(text))
    return render_template("predictT.html",message=sentiment,review=text)


@app.route('/download/<filename>',  methods=['Get','POST'])
def download_file(filename):
    response_file = cache["foo"]
    resp = make_response(response_file.to_csv())
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.route('/refresh', methods=['POST'])
def refresh():
    return render_template("index.html")   

def predict_from_cache(sentence):
    if sentence in PREDICTION_DICT:
        return PREDICTION_DICT[sentence]
    else:
        result = sentence_prediction(sentence)
        PREDICTION_DICT[sentence] = result
        return result


# if __name__ == "__main__":
#     app.run(debug=False,threaded=False)


@memory.cache
def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    # token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    # token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    # token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    # token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask)

    outputs = torch.softmax(outputs).cpu().detach().numpy()
    return outputs[0][0][0]


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence, model = MODEL)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        "neutral" : str(neutral_prediction),
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    MODEL = SentimentClassifier(3)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location = 'cpu'))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()

    # oaded_state = torch.load(model_path+seq_to_seq_test_model_fname,map_location='cuda:0'