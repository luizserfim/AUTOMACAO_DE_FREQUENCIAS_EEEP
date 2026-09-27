import cv2
import matplotlib.pyplot as plt
from src.leitor_imagem import carregar_imagem


caminhos = [
    "img/folha_teste_1 (1).jpg",
    "img/folha_teste_1 (2).jpg"
]

for caminho in caminhos:
    imagem = carregar_imagem(caminho)

    altura, largura, canais = imagem.shape

    print(f"\nImagem: {caminho}")
    print("Imagem carregada com sucesso!")
    print(f"Largura: {largura} pixels")
    print(f"Altura: {altura} pixels")
    print(f"Canais de cor: {canais}")

    # OpenCV trabalha com BGR.
    # Matplotlib espera RGB.
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imagem_rgb)
    plt.title(caminho)
    plt.axis("off")

plt.show()