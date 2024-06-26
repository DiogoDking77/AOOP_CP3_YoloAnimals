from flask import Flask, render_template, Response
import cv2
import subprocess
from ultralytics import YOLO
import threading

app = Flask(__name__)

# Função para obter o URL de streaming com yt-dlp
def get_youtube_stream_url(youtube_url):
    result = subprocess.run(['yt-dlp', '-g', youtube_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stream_url = result.stdout.strip()
    return stream_url

# Carregar o modelo treinado
model = YOLO('best.pt')  # Substitua pelo caminho para seu modelo treinado

# URL da transmissão ao vivo do YouTube
url = 'https://www.youtube.com/live/3bf1JDW_50k?si=a-W3gbM4-s0Q3CIm'  # Substitua pelo URL da transmissão ao vivo do YouTube
stream_url = get_youtube_stream_url(url)
cap = cv2.VideoCapture(stream_url)

def generate_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Fazer predições no frame
        results = model(frame)

        # Desenhar os resultados na imagem
        for result in results:
            annotated_frame = result.plot()

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

def run_app():
    app.run(host='0.0.0.0', port=5000)

# Iniciar o servidor Flask em uma thread separada
threading.Thread(target=run_app).start()
