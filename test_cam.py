import cv2

# Membuka kamera utama menggunakan backend V4L2 Linux
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

print("Mencoba membuka kamera... Tekan 'q' pada jendela gambar untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil gambar dari kamera.")
        break
        
    # Menampilkan gambar mentah dari kamera tanpa pemrosesan AI
    cv2.imshow("Tes Kamera Mentah", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()