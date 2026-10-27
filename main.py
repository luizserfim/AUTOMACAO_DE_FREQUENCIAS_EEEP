"""Diagnóstico de grade e células do Projeto Frequência Inteligente.

Exibe apenas duas figuras por fotografia: grade e células pré-processadas.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.leitor_imagem import carregar_imagem
from src.processamento import binarizar_imagem, extrair_grade
from src.detector_grade import (
    detectar_contornos,
    selecionar_maior_contorno,
    detectar_vertices,
)
from src.segmentacao import (
    detectar_linhas_internas,
    localizar_linhas,
    localizar_linhas_por_faixas,
    reconstruir_linhas_horizontais,
    localizar_verticais_por_faixas,
    reconstruir_linhas_verticais,
)
from src.perspectiva import corrigir_perspectiva
from src.extracao_celulas import extrair_celulas
from src.classificador_celulas import preprocessar_celula, analisar_componentes

CAMINHOS = [
    "img/folha_teste_1 (1).jpg",
    "img/folha_teste_1 (2).jpg",
]
LINHA_ALUNO = 1
QUANTIDADE_AMOSTRAS = 6


def mostrar_grade(imagem_corrigida, linhas_horizontais, linhas_verticais, caminho):
    """Desenha a grade reconstruída em uma única figura."""
    imagem_grade = imagem_corrigida.copy()
    for pontos in linhas_horizontais:
        cv2.polylines(imagem_grade, [np.asarray(pontos, dtype=np.int32)],
                      False, (0, 255, 0), 2)
    for pontos in linhas_verticais:
        cv2.polylines(imagem_grade, [np.asarray(pontos, dtype=np.int32)],
                      False, (0, 0, 255), 2)

    fig, eixo = plt.subplots(figsize=(12, 7))
    eixo.imshow(cv2.cvtColor(imagem_grade, cv2.COLOR_BGR2RGB))
    eixo.set_title(f"GRADE COMPLETA RECONSTRUÍDA - {caminho}")
    eixo.axis("off")
    fig.tight_layout()


def mostrar_analise_celulas(celulas, caminho):
    """Mostra originais e binarizadas e imprime métricas no terminal."""
    celulas_linha = sorted(
        (c for c in celulas if c["linha"] == LINHA_ALUNO),
        key=lambda c: c["coluna"],
    )

    print("\nDIMENSÕES DAS PRIMEIRAS 12 CÉLULAS")
    for celula in celulas_linha[:12]:
        altura, largura = celula["imagem"].shape[:2]
        print(f"Coluna {celula['coluna']:02d}: {largura} x {altura} px")

    # Ainda é uma hipótese que a frequência comece na coluna 2.
    # Na primeira fotografia há indício de divisão indevida do campo de nomes.
    amostras = [c for c in celulas_linha if c["coluna"] >= 2]
    amostras = amostras[:QUANTIDADE_AMOSTRAS]
    if not amostras:
        print("Nenhuma célula disponível para análise.")
        return

    fig, eixos = plt.subplots(2, len(amostras), figsize=(14, 5), squeeze=False)
    for indice, celula in enumerate(amostras):
        original = celula["imagem"]
        processada = preprocessar_celula(original, margem=2)
        analise = analisar_componentes(processada, area_minima=3)

        print(f"\nLINHA {celula['linha']} | COLUNA {celula['coluna']}")
        print(f"Dimensões: {analise['largura_celula']} x {analise['altura_celula']}")
        print(f"Pixels brancos: {analise['pixels_brancos']}")
        print(f"Ocupação: {analise['percentual_ocupacao']:.2f}%")
        print(f"Componentes: {analise['quantidade_componentes']}")
        for numero, componente in enumerate(analise["componentes"], 1):
            print(
                f"  Componente {numero}: área={componente['area']} px², "
                f"largura={componente['largura']} px, "
                f"altura={componente['altura']} px"
            )

        eixos[0, indice].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        eixos[0, indice].set_title(f"Original | col. {celula['coluna']}")
        eixos[1, indice].imshow(processada, cmap="gray", vmin=0, vmax=255)
        eixos[1, indice].set_title("Pré-processada")
        eixos[0, indice].axis("off")
        eixos[1, indice].axis("off")

    fig.suptitle(f"ANÁLISE DAS CÉLULAS - {caminho}")
    fig.tight_layout()


def processar_fotografia(caminho):
    print("\n" + "=" * 55)
    print(f"PROCESSANDO: {caminho}")
    print("=" * 55)

    imagem = carregar_imagem(caminho)
    if imagem is None:
        print("Imagem não carregada; verifique o caminho.")
        return
    altura, largura = imagem.shape[:2]
    print(f"Imagem carregada: {largura} x {altura} px")

    imagem_binaria = binarizar_imagem(imagem)
    grade = extrair_grade(imagem_binaria)
    contornos = detectar_contornos(grade)
    maior_contorno = selecionar_maior_contorno(contornos)
    if maior_contorno is None:
        print("Nenhum contorno encontrado.")
        return

    aproximacao = detectar_vertices(maior_contorno)
    print(f"Contornos: {len(contornos)} | Vértices: {len(aproximacao)}")
    if len(aproximacao) != 4:
        print("Não foi possível corrigir a perspectiva: são necessários 4 vértices.")
        return

    imagem_corrigida = corrigir_perspectiva(imagem, aproximacao.reshape(4, 2))
    altura_corrigida, largura_corrigida = imagem_corrigida.shape[:2]
    horizontais, verticais, _ = detectar_linhas_internas(imagem_corrigida)
    pos_horizontais, pos_verticais = localizar_linhas(horizontais, verticais)
    print(f"Linhas globais: {len(pos_horizontais)} horizontais; "
          f"{len(pos_verticais)} verticais")

    faixas_h = localizar_linhas_por_faixas(
        horizontais, quantidade_faixas=8, percentual_minimo=0.45
    )
    linhas_h = reconstruir_linhas_horizontais(
        faixas_h, distancia_maxima=8, minimo_faixas=3,
        max_faixas_ausentes=1, largura_imagem=largura_corrigida,
        altura_imagem=altura_corrigida,
    )
    faixas_v = localizar_verticais_por_faixas(
        verticais, quantidade_faixas=8, percentual_minimo=0.45,
        distancia_agrupamento=3,
    )
    linhas_v = reconstruir_linhas_verticais(
        faixas_v, distancia_maxima=5, minimo_faixas=3,
        max_faixas_ausentes=1, altura_imagem=altura_corrigida,
        largura_imagem=largura_corrigida,
    )
    print(f"Linhas reconstruídas: {len(linhas_h)} horizontais; "
          f"{len(linhas_v)} verticais")

    print("\nPOSIÇÕES DAS LINHAS VERTICAIS RECONSTRUÍDAS")
    for indice, pontos in enumerate(linhas_v):
        coordenadas_x = [ponto[0] for ponto in pontos]
        if coordenadas_x:
            print(f"Linha vertical {indice:02d}: "
                  f"x médio = {np.mean(coordenadas_x):.2f} px")

    mostrar_grade(imagem_corrigida, linhas_h, linhas_v, caminho)

    celulas = extrair_celulas(
        imagem_corrigida, linhas_h, linhas_v, margem=2
    )
    print(f"\nCélulas extraídas: {len(celulas)}")
    mostrar_analise_celulas(celulas, caminho)
    print("Processamento concluído!")


if __name__ == "__main__":
    for caminho in CAMINHOS:
        processar_fotografia(caminho)
    print(f"\nTotal de figuras criadas: {len(plt.get_fignums())}")
    plt.show()
