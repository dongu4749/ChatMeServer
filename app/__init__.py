from flask import Flask, request, jsonify, render_template
from io import BytesIO
from PIL import Image
from datetime import datetime
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from konlpy.tag import Okt
from collections import Counter
import mysql.connector
import torch
import base64
import numpy as np
import io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")  
model_path = "C:/Users/dongu/Downloads/model_-epoch=20-train_loss=17.05.ckpt" 
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
gpt_checkpoint = torch.load(model_path, map_location=device)
# 현재 가중치 키를 새로운 가중치 키로 수정
fixed_state_dict = {}
for key in gpt_checkpoint["state_dict"].keys():
    new_key = key.replace("kogpt2.", "")
    fixed_state_dict[new_key] = gpt_checkpoint["state_dict"][key]

# 수정한 가중치 키로 모델의 가중치를 로드
model.load_state_dict(fixed_state_dict)
model.to(device)
model.eval()


bert_tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(bert_tokenizer.vocab_file, padding_token='[PAD]')

class BERTDataset(Dataset):#모델에 입력할 데이터를 처리
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

#cpu를 gpu를 사용하는 것으로 변경 가능.
device = torch.device('cpu')
#하이퍼파라미터
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 15
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5
tok = bert_tokenizer.tokenize

#클래스 수는 flask 서버에서 작동하실 때 반드시 같아야함.
#bert 모델을 이용한 분류기 모델 클래스
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


#BERT 모델 불러오기
bert_model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

best_acc=0.0
best_loss=99999999
epoch = 0

bert_model_path = "C:/Users/dongu/Downloads/saved_model_kobert.pt2"
checkpoint = torch.load(bert_model_path, map_location=torch.device('cpu'))
bert_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']


def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    bert_model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = bert_model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("0")#중립
            elif np.argmax(logits) == 1:
                test_eval.append("1")#분노
            elif np.argmax(logits) == 2:
                test_eval.append("2")#기쁨
            elif np.argmax(logits) == 3:
                test_eval.append("3")#불안
            elif np.argmax(logits) == 4:
                test_eval.append("4")#놀람
            elif np.argmax(logits) == 5:
                test_eval.append("5")#슬픔
            elif np.argmax(logits) == 6:
                test_eval.append("6")#혐오

        print(test_eval[0])
        return test_eval[0]
    
# # 모델 및 토크나이저 로드
# bert_tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
# bert_model = AutoModelForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=7)

# # 저장된 체크포인트
# ckpt_name = "C:/Users/dongu/Downloads/kobert_.pt2"
# # 체크포인트 로드
# checkpoint = torch.load(ckpt_name, map_location=torch.device('cpu'))
# bert_model.load_state_dict(checkpoint['model_state_dict'])
# bert_model.eval()


diary_summary_model = "jx7789/kobart_summary_v2"

gen_kwargs = {"length_penalty": 2.0, "num_beams":8, "max_length": 128}

pipe = pipeline("summarization", model=diary_summary_model)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    try:
        # 회원가입 정보 받아오기
        id = request.form['uid']
        
        conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="chatme",
        auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()
        

        # 새로운 사용자 추가
        sql = "INSERT INTO user (id) VALUES (%s)"
        val = (id,)
        cursor.execute(sql, val)
        conn.commit()

        # 응답 반환
        return "Success"

    except Exception as e:
        print(e)
        return "Error occurred", 500

    finally:
        # MySQL 연결 종료
        cursor.close()
        conn.close()


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()  # JSON 형식으로 전달된 데이터를 받습니다.
    user_id = data['user_id']  # 사용자 ID를 가져옵니다.
    message = data['message']  # 사용자의 입력을 가져옵니다.

    conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="chatme",
    auth_plugin='mysql_native_password'
    )
    cursor = conn.cursor()
    
    # 현재 시간 구하기
    current_user_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 메시지 저장
    sql = "INSERT INTO message (user_id, content, isUser, time) VALUES (%s, %s, %s, %s)"
    val = (user_id, message, True, current_user_time)
    cursor.execute(sql, val)
    conn.commit()
    
    # 모델을 사용하여 입력에 대한 답변을 생성합니다.
    response = generate_response(message)   

    # 현재 시간 구하기
    current_bot_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 챗봇 응답 저장
    sql = "INSERT INTO message (user_id, content, isUser, time) VALUES (%s, %s, %s, %s)"
    val = (user_id, response, False, current_bot_time)
    cursor.execute(sql, val)
    conn.commit()

    # 생성된 답변을 JSON 형식으로 반환합니다.
    return jsonify({'response': response})

@app.route('/chat_history', methods=['POST'])
def get_chat_history():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="chatme",
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()

        # 요청으로부터 사용자 ID 가져오기
        user_id = request.json.get('user_id')

        # 메시지 테이블로부터 해당 사용자의 채팅 내역 조회
        sql = "SELECT * FROM message WHERE user_id = %s"
        cursor.execute(sql, (user_id,))
        rows = cursor.fetchall()

        # 조회된 결과를 JSON 형식으로 변환하여 반환
        chat_history = []
        for row in rows:
            message = {
                'num': row[0],
                'user_id': row[1],
                'content': row[2],
                'isUser': row[3],
                'time': row[4].strftime('%Y-%m-%d %H:%M:%S')
            }
            if row[5] is not None:
                # 이미지가 있는 경우 이미지를 Base64 인코딩하여 추가
                image_data = row[5]
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                message['image'] = image_base64
            else:
                # 이미지가 없는 경우 텍스트 메시지를 추가
                message['content'] = row[2]

            chat_history.append(message)

        return jsonify({'chat_history': chat_history})

    except Exception as e:
        print(e)
        return "Error occurred", 500

    finally:
        # MySQL 연결 종료
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


@app.route('/get_photo', methods=['POST'])
def get_photo():
    try:
        user_id = request.json.get('user_id')
        year = request.json.get('year')
        month = request.json.get('month')
        dayOfMonth = request.json.get('dayOfMonth')

        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="chatme",
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()

        # 해당 날짜에 해당하는 이미지 조회
        sql = "SELECT * FROM message WHERE user_id = %s AND DATE(time) = %s"
        val = (user_id, f"{year}-{month:02d}-{dayOfMonth:02d}")
        cursor.execute(sql, val)
        result = cursor.fetchall()

        # 조회된 결과를 JSON 형식으로 변환하여 반환
        photo_history = []
        for row in result:
            message = {
                'num': row[0],
                'user_id': row[1],
                
                'isUser': row[3],
                'time': row[4].strftime('%Y-%m-%d %H:%M:%S')
            }
            if row[5] is not None:
                # 이미지가 있는 경우 이미지를 Base64 인코딩하여 추가
                image_data = row[5]
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                message['image'] = image_base64

            photo_history.append(message)

        return jsonify({'photo_history': photo_history})

    except Exception as e:
        print(e)
        return "Error occurred", 500

    finally:
        # MySQL 연결 종료
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()




@app.route('/photo', methods=['POST'])
def upload_photo():
    try:
        data = request.get_json()
        image_base64 = data['image']
        user_id = data['id']
        
        # 이미지 Base64 디코딩
        image_data = base64.b64decode(image_base64)
        
        # 이미지를 Pillow Image 객체로 열고, 크기 조정
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((640, 640))
        
        # 다시 이미지 데이터로 인코딩
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_data = buffered.getvalue()

        # 이미지 데이터를 데이터베이스에 저장
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="chatme",
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()
        current_user_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = "INSERT INTO message (user_id, isUser, time, image) VALUES (%s, %s, %s, %s)"
        val = (user_id, False, current_user_time, image_data)
        cursor.execute(sql, val)
        conn.commit()

        # 삽입된 사진의 ID 조회
        photo_id = cursor.lastrowid

        # 삽입된 사진 조회 및 반환
        sql = "SELECT image FROM message WHERE num = %s"
        val = (photo_id,)
        cursor.execute(sql, val)
        result = cursor.fetchone()

        # 이미지 데이터가 있으면 Base64 인코딩하여 반환
        if result is not None:
            photo_data = result[0]
            photo_base64 = base64.b64encode(photo_data).decode('utf-8')
            response = {
                'photo_id': photo_id,
                'image': photo_base64
            }
            return jsonify(response)
        else:
            return "사진을 찾을 수 없습니다."

    except Exception as e:
        print(e)
        return "Error occurred", 500

    finally:
        # MySQL 연결 종료
        cursor.close()
        conn.close()

@app.route('/emotion', methods=['POST'])
def get_emotion():
    try:
        # 클라이언 = request.json.get('year')
        user_id = request.json.get('user_id')
        year = request.json.get('year')
        month = request.json.get('month')
        dayOfMonth = request.json.get('dayOfMonth')
        isUser = True

        # 데이터베이스 연결 설정
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="chatme",
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()

         # 해당 날짜의 메시지를 선택합니다.
        sql = "SELECT content FROM message WHERE user_id = %s AND isUser = %s AND DATE(time) = %s"
        val = (user_id, isUser, f"{year}-{month:02d}-{dayOfMonth:02d}")
        cursor.execute(sql, val)
        result = cursor.fetchall()

        # 메시지를 순회하며 모델에 입력해 결과값을 가져옵니다.
        emotion_results = []
        for row in result:
            content = row[0]
            if content:  # 메시지가 비어 있지 않은 경우에만 전처리하고 모델에 입력합니다.
                # 여기를 수정했습니다. 리스트 대신 단일 텍스트를 전달합니다.
                emotion_idx = predict(content)
                emotion_results.append(emotion_idx)

        cursor.close()  
        conn.close()

        # 감정 분석 결과를 jsonify를 이용하여 json 형태로 반환
        return jsonify({'emotion_results': emotion_results})
    except Exception as e:
        # 오류 발생 시 오류 메시지 반환
        return jsonify({'error': str(e)})


@app.route('/diary', methods=['POST'])
def diary_summary():
    try:
        # 클라이언 = request.json.get('year')
        user_id = request.json.get('user_id')
        year = request.json.get('year')
        month = request.json.get('month')
        dayOfMonth = request.json.get('dayOfMonth')
        isUser = True

        # 데이터베이스 연결 설정
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="chatme",
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()

         # 해당 날짜의 메시지를 선택합니다.
        sql = "SELECT content FROM message WHERE user_id = %s AND isUser = %s AND DATE(time) = %s"
        val = (user_id, isUser, f"{year}-{month:02d}-{dayOfMonth:02d}")
        cursor.execute(sql, val)
        result = cursor.fetchall()

        # 각 content를 컴마로 연결하여 하나의 문자열로 만듭니다.
        joined_contents = ", ".join([row[0] for row in result])

        diary_Summary = diary_summary(joined_contents)

        cursor.close()  
        conn.close()

        # 감정 분석 결과를 jsonify를 이용하여 json 형태로 반환
        return jsonify({'diary_Summary': diary_Summary})
    except Exception as e:
        # 오류 발생 시 오류 메시지 반환
        return jsonify({'error': str(e)})
    
@app.route('/keyword', methods=['POST'])
def diary_keyword():
    try:
        # 클라이언 = request.json.get('year')
        user_id = request.json.get('user_id')
        year = request.json.get('year')
        month = request.json.get('month')
        dayOfMonth = request.json.get('dayOfMonth')
        isUser = True

        # 데이터베이스 연결 설정
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="chatme",
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor()

         # 해당 날짜의 메시지를 선택합니다.
        sql = "SELECT content FROM message WHERE user_id = %s AND isUser = %s AND DATE(time) = %s"
        val = (user_id, isUser, f"{year}-{month:02d}-{dayOfMonth:02d}")
        cursor.execute(sql, val)
        result = cursor.fetchall()

        joined_contents = ' '.join([row[0] for row in result])

        keywords = extract_keywords(joined_contents)
        # 키워드 값만 추출하여 리스트로 변환
        keyword_list = [keyword for keyword in keywords]

        cursor.close()
        conn.close()

        # 감정 분석 결과를 jsonify를 이용하여 json 형태로 반환
        return jsonify({'keyword': keyword_list})
    except Exception as e:
        # 오류 발생 시 오류 메시지 반환
        return jsonify({'error': str(e)})

    
def generate_response(input_text, max_length=20):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def predict_emotion(sentence):
    inputs = bert_tokenizer.encode_plus(sentence, add_special_tokens=True, padding='longest', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    return str(predicted_label)

def diary_summary(sentence):

    dialogue = sentence
    
    return pipe("[sep]".join(dialogue), **gen_kwargs)[0]["summary_text"]

def extract_keywords(text):
    # Initialize the morphological analyzer
    okt = Okt()

    # Extract nouns
    nouns = okt.nouns(text)

    # Calculate frequencies
    counter = Counter(nouns)

    # Select the top 5 keywords
    keywords = [noun for noun, freq in counter.most_common(3)]

    return keywords

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    