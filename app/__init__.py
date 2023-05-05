from flask import Flask, request, jsonify, render_template
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import mysql.connector
from datetime import datetime

app = Flask(__name__)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
model_path = "C:/Users/dongu/Downloads/kogpt2_model.pth"  
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


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
        database="chatme"
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
    database="chatme"
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
            database="chatme"
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
            chat_history.append(message)

        return jsonify({'chat_history': chat_history})

    except Exception as e:
        print(e)
        return "Error occurred", 500

    finally:
        # MySQL 연결 종료
        cursor.close()
        conn.close()

def generate_response(input_text, max_length=20):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
