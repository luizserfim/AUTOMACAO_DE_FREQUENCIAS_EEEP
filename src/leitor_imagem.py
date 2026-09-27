import cv2

def carregar_imagem(caminho):
    imagem = cv2.imread(caminho)
    if imagem is None:
        raise FileNotFoundError(f"Nao foi possivel carregar a imagem: {caminho}")
    return imagem