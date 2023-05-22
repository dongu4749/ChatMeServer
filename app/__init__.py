from flask import Flask, request, jsonify, render_template
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import mysql.connector
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import io

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
        cursor.close()
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
            database="chatme"
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

            photo_history.append(message)

        return jsonify({'photo_history': photo_history})

    except Exception as e:
        print(e)
        return "Error occurred", 500

    finally:
        # MySQL 연결 종료
        cursor.close()
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
            database="chatme"
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
    
def generate_response(input_text, max_length=20):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
