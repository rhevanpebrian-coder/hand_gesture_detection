import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
import time

# 1. Inisialisasi Pygame untuk Suara
pygame.mixer.init()

audio_map = {
    "Selamat": "SELAMAT.mp3",
    "Berjuang": "BERJUANG.mp3",
    "Sukses": "SUKSESS.mp3"
}

# 2. Konfigurasi Resmi MediaPipe Tasks (Tanpa menggunakan mp.solutions)
model_path = 'hand_landmarker.task'
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

# 3. Inisialisasi Kamera dengan Backend V4L2 Linux
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

last_played = 0
AUDIO_COOLDOWN = 1.5

print("Program Hand Landmarker Tasks Berjalan... Tekan 'q' untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Ambil timestamp dalam milidetik (Wajib untuk mode VIDEO)
    timestamp = int(time.time() * 1000)
    
    # Konversi BGR ke RGB format MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Jalankan Deteksi
    detection_result = landmarker.detect_for_video(mp_image, timestamp)
    
    label = "Mencari isyarat..."

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Ambil koordinat landmark
            lm = hand_landmarks
            
            # Ambil status keterbukaan jari (Ujung vs Pangkal bawah jari)
            index_open = lm[8].y < lm[5].y
            middle_open = lm[12].y < lm[9].y
            ring_open = lm[16].y < lm[13].y
            pinky_open = lm[20].y < lm[17].y
            
            # Cek Jempol (Jarak horizontal dari pangkal telunjuk)
            thumb_open = abs(lm[4].x - lm[5].x) > 0.05 

            # --- LOGIKA COCOK ISYARAT (SUDAH DITUKAR) ---
            
            # 1. BERJUANG (Metal: Telunjuk & Kelingking tegak, Tengah & Manis ditekuk)
            if index_open and pinky_open and not middle_open and not ring_open:
                label = "Berjuang"
                
            # 2. SUKSES (Jempol: Hanya jempol terbuka, 4 jari lainnya ditekuk)
            elif thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
                label = "Sukses"
                
            # 3. SELAMAT (Mengepal: Semua 4 jari utama ditekuk rapat ke bawah)
            elif not index_open and not middle_open and not ring_open and not pinky_open:
                label = "Selamat"

            # Gambar titik koordinat di layar secara manual (Sebagai pengganti mp_drawing)
            for point in lm:
                cx, cy = int(point.x * 320), int(point.y * 240)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # 4. Logika Putar Audio
            if label in audio_map:
                cv2.putText(frame, f"{label} (100.00%)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                current_time = time.time()
                if (current_time - last_played) > AUDIO_COOLDOWN:
                    try:
                        pygame.mixer.music.load(audio_map[label])
                        pygame.mixer.music.play()
                        last_played = current_time
                    except Exception as e:
                        print(f"Gagal memutar audio: {e}")
            else:
                cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture - MediaPipe Task", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
landmarker.close()
