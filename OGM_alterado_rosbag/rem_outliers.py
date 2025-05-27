import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar o mapa gerado pelo matplotlib
img = cv2.imread('Figure_1.png', cv2.IMREAD_GRAYSCALE)

# Binarizar a imagem: tudo abaixo de 240 é "ocupado"
_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

# Remove pequenos objetos (outliers) com connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Criar imagem de saída (só mantém objetos com área mínima)
min_area = 100  # ajusta conforme necessário
# Etiquetar componentes conectadas
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
cleaned = np.zeros_like(binary)

for i in range(1, num_labels):  # ignorar label 0 (fundo)
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_area:
        cleaned[labels == i] = 255

# 4. Inverter de volta (obstáculos a preto se necessário)
final = cv2.bitwise_not(cleaned)

# 5. Guardar imagem final
cv2.imwrite("map_clean.png", final)

# 6. Mostrar antes e depois
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')
plt.grid(False) 

plt.subplot(1, 2, 2)
plt.imshow(final, cmap='gray')
plt.title("Sem Outliers")
plt.axis('off')
plt.grid(False) 

plt.tight_layout()
plt.show()
