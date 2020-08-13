import torch
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

MAX_LEN = 500
# EPOCHS = 10
BERT_PATH = 'bert-base-uncased'
# BERT_PATH = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
# BERT_PATH = (r"C:\Users\hanit\OneDrive\Desktop\Water_ROOT\input\uncased")
MODEL_PATH = (r"D:\Water_ROOT\input\best_model_state.bin")


TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['negative', 'neutral', 'positive']
Batch_size = 16

