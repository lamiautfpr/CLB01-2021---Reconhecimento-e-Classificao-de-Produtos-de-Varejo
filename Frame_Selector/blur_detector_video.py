# imports
from imutils.video import VideoStream
from Frame_Selector.blur_detector import detect_blur_fft
import argparse
import imutils
import time
import cv2

# constrói o analisador e analisa os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--thresh", type=int, default=10,
	help="threshold for our blur detector to fire")
args = vars(ap.parse_args())

# inicia a webcam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop nos frames do video
while True:
	# redimensiona o frame para max 400px de largura
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	# converte o frame em tons de cinza e detecta se tem desfoque
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	(mean, blurry) = detect_blur_fft(gray, size=60,
		thresh=args["thresh"], vis=False)

# mostra se está borrado em cada frame
color = (0, 0, 255) if blurry else (0, 255, 0)
text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
text = text.format(mean)
cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, color, 2)

# mostra o frame de saída
cv2.imshow("Frame", frame)
key = cv2.waitKey(1) & 0xFF

# use 'q' para sair
if key == ord("q"):
	break

# limpa
cv2.destroyAllWindows()
vs.stop()