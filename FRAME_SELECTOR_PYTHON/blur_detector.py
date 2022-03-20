# imports
import matplotlib.pyplot as plt
import numpy as np

# pega as dimensões da img e deriva suas coordenadas
def detect_blur_fft(image, size=60, thresh=10, vis=False):
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))

# calcula a transformada de fourier
fft = np.fft.fft2(image)
fftShift = np.fft.fftshift(fft)

# if para verificar se está visualizando a saída
	if vis:
		# calcula a magnitude
		magnitude = 20 * np.log(np.abs(fftShift))
		# mostra a img original
		(fig, ax) = plt.subplots(1, 2, )
		ax[0].imshow(image, cmap="gray")
		ax[0].set_title("Input")
		ax[0].set_xticks([])
		ax[0].set_yticks([])
		# mostra a img de magnitude
		ax[1].imshow(magnitude, cmap="gray")
		ax[1].set_title("Magnitude Spectrum")
		ax[1].set_xticks([])
		ax[1].set_yticks([])
		
		plt.show()

# remove baixas frequências
# aplica a transformada de fourier inversa
fftShift[cY - size:cY + size, cX - size:cX + size] = 0
fftShift = np.fft.ifftshift(fftShift)
recon = np.fft.ifft2(fftShift)

# calula a magnitude dnv, mas com DC central zerado
# calcula a média da magnitude
magnitude = 20 * np.log(np.abs(recon))
mean = np.mean(magnitude)
# retorna se está borrado ou não
return (mean, mean <= thresh)