
import cv2
import numpy as np


# ==================================================
# 1. DETECÇÃO DAS LINHAS INTERNAS
# ==================================================

def detectar_linhas_internas(imagem_corrigida):
    """
    Detecta linhas horizontais e verticais
    após a correção de perspectiva.

    Retorna:
        linhas_horizontais
        linhas_verticais
        grade_interna
    """

    # 1.1. Conversão para escala de cinza
    imagem_cinza = cv2.cvtColor(
        imagem_corrigida,
        cv2.COLOR_BGR2GRAY
    )

    # 1.2. Binarização adaptativa
    imagem_binaria = cv2.adaptiveThreshold(
        imagem_cinza,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    # 1.3. Dimensões da imagem
    altura, largura = imagem_binaria.shape

    # 1.4. Elementos estruturantes
    tamanho_horizontal = max(20, largura // 30)
    tamanho_vertical = max(20, altura // 30)

    kernel_horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (tamanho_horizontal, 1)
    )

    kernel_vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, tamanho_vertical)
    )

    # 1.5. Extração das linhas horizontais
    linhas_horizontais = cv2.morphologyEx(
        imagem_binaria,
        cv2.MORPH_OPEN,
        kernel_horizontal
    )

    # 1.6. Extração das linhas verticais
    linhas_verticais = cv2.morphologyEx(
        imagem_binaria,
        cv2.MORPH_OPEN,
        kernel_vertical
    )

    # 1.7. Kernels de fechamento morfológico
    kernel_fechamento_horizontal = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 1)
        )
    )

    kernel_fechamento_vertical = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 9)
        )
    )

    # 1.8. Conexão de falhas horizontais
    linhas_horizontais = cv2.morphologyEx(
        linhas_horizontais,
        cv2.MORPH_CLOSE,
        kernel_fechamento_horizontal
    )

    # 1.9. Conexão de falhas verticais
    linhas_verticais = cv2.morphologyEx(
        linhas_verticais,
        cv2.MORPH_CLOSE,
        kernel_fechamento_vertical
    )

    # 1.10. Combinação da grade
    grade_interna = cv2.bitwise_or(
        linhas_horizontais,
        linhas_verticais
    )

    return (
        linhas_horizontais,
        linhas_verticais,
        grade_interna
    )


# ==================================================
# 2. AGRUPAMENTO DE POSIÇÕES PRÓXIMAS
# ==================================================

def agrupar_posicoes(posicoes, distancia_maxima=5):
    """
    Agrupa coordenadas próximas e retorna
    o centro aproximado de cada grupo.
    """

    if len(posicoes) == 0:
        return []

    posicoes = sorted(int(p) for p in posicoes)

    grupos = [[posicoes[0]]]

    for posicao in posicoes[1:]:

        if posicao - grupos[-1][-1] <= distancia_maxima:
            grupos[-1].append(posicao)
        else:
            grupos.append([posicao])

    centros = [
        int(round(np.mean(grupo)))
        for grupo in grupos
    ]

    return centros


# ==================================================
# 3. LOCALIZAÇÃO GLOBAL DAS LINHAS
# ==================================================

def localizar_linhas(
    linhas_horizontais,
    linhas_verticais,
    percentual_minimo=0.35
):
    """
    Localiza linhas horizontais e verticais
    usando projeções globais.

    Retorna:
        posicoes_horizontais
        posicoes_verticais
    """

    altura, largura = linhas_horizontais.shape

    # 3.1. Projeção horizontal
    projecao_horizontal = np.count_nonzero(
        linhas_horizontais,
        axis=1
    )

    # 3.2. Projeção vertical
    projecao_vertical = np.count_nonzero(
        linhas_verticais,
        axis=0
    )

    # 3.3. Limites mínimos
    limite_horizontal = largura * percentual_minimo
    limite_vertical = altura * percentual_minimo

    # 3.4. Coordenadas candidatas
    candidatas_horizontais = np.where(
        projecao_horizontal >= limite_horizontal
    )[0]

    candidatas_verticais = np.where(
        projecao_vertical >= limite_vertical
    )[0]

    # 3.5. Agrupamento
    posicoes_horizontais = agrupar_posicoes(
        candidatas_horizontais
    )

    posicoes_verticais = agrupar_posicoes(
        candidatas_verticais
    )

    return (
        posicoes_horizontais,
        posicoes_verticais
    )


# ==================================================
# 4. DETECÇÃO HORIZONTAL POR FAIXAS
# ==================================================

def localizar_linhas_por_faixas(
    linhas_horizontais,
    quantidade_faixas=8,
    percentual_minimo=0.45,
    distancia_agrupamento=4
):
    """
    Divide a imagem em faixas verticais
    e localiza linhas horizontais em cada faixa.

    Retorna uma lista de dicionários.
    """

    altura, largura = linhas_horizontais.shape

    resultados = []

    # 4.1. Divisão em faixas
    limites = np.linspace(
        0,
        largura,
        quantidade_faixas + 1,
        dtype=int
    )

    for indice in range(quantidade_faixas):

        x_inicio = int(limites[indice])
        x_fim = int(limites[indice + 1])

        # 4.2. Recorte da faixa
        faixa = linhas_horizontais[
            :,
            x_inicio:x_fim
        ]

        largura_faixa = x_fim - x_inicio

        if largura_faixa == 0:
            continue

        # 4.3. Projeção horizontal local
        projecao = np.count_nonzero(
            faixa,
            axis=1
        )

        limite = largura_faixa * percentual_minimo

        candidatas = np.where(
            projecao >= limite
        )[0]

        # 4.4. Agrupamento
        posicoes = agrupar_posicoes(
            candidatas,
            distancia_maxima=distancia_agrupamento
        )

        # 4.5. Armazenamento
        resultados.append({
            "faixa": indice + 1,
            "x_inicio": x_inicio,
            "x_fim": x_fim,
            "posicoes": posicoes
        })

    return resultados


# ==================================================
# 5. RECONSTRUÇÃO DAS LINHAS HORIZONTAIS
# ==================================================

def reconstruir_linhas_horizontais(
    resultados_faixas,
    distancia_maxima=8,
    minimo_faixas=3,
    max_faixas_ausentes=1,
    largura_imagem=None,
    altura_imagem=None
):
    """
    Reconstrói linhas horizontais.

    Permite falhas entre faixas, considera
    inclinação e estende até as bordas.

    Retorna uma lista de linhas com pontos (x, y).
    """

    if not resultados_faixas:
        return []

    # 5.1. Determinação da largura
    if largura_imagem is None:
        largura_imagem = max(
            resultado["x_fim"]
            for resultado in resultados_faixas
        )

    linhas = []

    # 5.2. Percorrer as faixas
    for indice_faixa, resultado in enumerate(
        resultados_faixas
    ):

        x_inicio = resultado["x_inicio"]
        x_fim = resultado["x_fim"]

        x_centro = (x_inicio + x_fim) // 2

        posicoes = sorted(resultado["posicoes"])

        candidatos = []

        # 5.3. Previsão das posições
        for indice_linha, linha in enumerate(linhas):

            diferenca_faixas = (
                indice_faixa - linha["ultima_faixa"]
            )

            if diferenca_faixas < 1:
                continue

            if diferenca_faixas > max_faixas_ausentes + 1:
                continue

            pontos = linha["pontos"]

            ultimo_x, ultimo_y = pontos[-1]

            y_previsto = ultimo_y

            if len(pontos) >= 2:

                anterior_x, anterior_y = pontos[-2]

                delta_x = ultimo_x - anterior_x

                if delta_x != 0:

                    inclinacao = (
                        (ultimo_y - anterior_y) / delta_x
                    )

                    y_previsto = (
                        ultimo_y
                        + inclinacao * (x_centro - ultimo_x)
                    )

            for indice_posicao, y in enumerate(posicoes):

                distancia = abs(y - y_previsto)

                tolerancia = (
                    distancia_maxima * diferenca_faixas
                )

                if distancia <= tolerancia:

                    candidatos.append((
                        distancia,
                        indice_linha,
                        indice_posicao
                    ))

        # 5.4. Associação dos segmentos
        candidatos.sort(key=lambda item: item[0])

        linhas_utilizadas = set()
        posicoes_utilizadas = set()

        for _, indice_linha, indice_posicao in candidatos:

            if indice_linha in linhas_utilizadas:
                continue

            if indice_posicao in posicoes_utilizadas:
                continue

            y = int(posicoes[indice_posicao])

            linhas[indice_linha]["pontos"].append(
                (x_centro, y)
            )

            linhas[indice_linha]["ultima_faixa"] = (
                indice_faixa
            )

            linhas_utilizadas.add(indice_linha)
            posicoes_utilizadas.add(indice_posicao)

        # 5.5. Novas linhas
        for indice_posicao, y in enumerate(posicoes):

            if indice_posicao in posicoes_utilizadas:
                continue

            linhas.append({
                "pontos": [(x_centro, int(y))],
                "ultima_faixa": indice_faixa
            })

    # 5.6. Filtragem
    linhas_validas = [
        linha["pontos"]
        for linha in linhas
        if len(linha["pontos"]) >= minimo_faixas
    ]

    linhas_reconstruidas = []

    # 5.7. Extensão às bordas
    for pontos in linhas_validas:

        pontos = sorted(
            pontos,
            key=lambda ponto: ponto[0]
        )

        coordenadas_x = np.array(
            [p[0] for p in pontos],
            dtype=float
        )

        coordenadas_y = np.array(
            [p[1] for p in pontos],
            dtype=float
        )

        # Inclinação na extremidade esquerda
        delta_x_esquerda = (
            coordenadas_x[1] - coordenadas_x[0]
        )

        inclinacao_esquerda = (
            (coordenadas_y[1] - coordenadas_y[0])
            / delta_x_esquerda
        )

        y_esquerda = (
            coordenadas_y[0]
            - inclinacao_esquerda * coordenadas_x[0]
        )

        # Inclinação na extremidade direita
        delta_x_direita = (
            coordenadas_x[-1] - coordenadas_x[-2]
        )

        inclinacao_direita = (
            (coordenadas_y[-1] - coordenadas_y[-2])
            / delta_x_direita
        )

        x_direita = largura_imagem - 1

        y_direita = (
            coordenadas_y[-1]
            + inclinacao_direita
            * (x_direita - coordenadas_x[-1])
        )

        # Limitar à altura da imagem
        if altura_imagem is not None:

            y_esquerda = np.clip(
                y_esquerda,
                0,
                altura_imagem - 1
            )

            y_direita = np.clip(
                y_direita,
                0,
                altura_imagem - 1
            )

        linha_completa = [
            (0, int(round(y_esquerda)))
        ]

        linha_completa.extend(pontos)

        linha_completa.append(
            (x_direita, int(round(y_direita)))
        )

        linhas_reconstruidas.append(
            linha_completa
        )

    # 5.8. Ordenação de cima para baixo
    linhas_reconstruidas.sort(
        key=lambda pontos: np.mean(
            [y for x, y in pontos]
        )
    )

    return linhas_reconstruidas


# ==================================================
# 6. DETECÇÃO VERTICAL POR FAIXAS
# ==================================================

def localizar_verticais_por_faixas(
    linhas_verticais,
    quantidade_faixas=8,
    percentual_minimo=0.45,
    distancia_agrupamento=3
):
    """
    Divide a imagem em faixas horizontais
    e localiza as linhas verticais em cada faixa.

    Retorna uma lista com os limites das faixas
    e as posições das colunas.
    """

    altura, largura = linhas_verticais.shape

    resultados = []

    # 6.1. Divisão da altura em faixas
    limites = np.linspace(
        0,
        altura,
        quantidade_faixas + 1,
        dtype=int
    )

    for indice in range(quantidade_faixas):

        y_inicio = int(limites[indice])
        y_fim = int(limites[indice + 1])

        # 6.2. Recorte da faixa
        faixa = linhas_verticais[
            y_inicio:y_fim,
            :
        ]

        altura_faixa = y_fim - y_inicio

        if altura_faixa == 0:
            continue

        # 6.3. Projeção vertical local
        projecao = np.count_nonzero(
            faixa,
            axis=0
        )

        limite = altura_faixa * percentual_minimo

        candidatas = np.where(
            projecao >= limite
        )[0]

        # 6.4. Agrupamento
        posicoes = agrupar_posicoes(
            candidatas,
            distancia_maxima=distancia_agrupamento
        )

        # 6.5. Armazenamento
        resultados.append({
            "faixa": indice + 1,
            "y_inicio": y_inicio,
            "y_fim": y_fim,
            "posicoes": posicoes
        })

    return resultados


# ==================================================
# 7. RECONSTRUÇÃO DAS LINHAS VERTICAIS
# ==================================================

def reconstruir_linhas_verticais(
    resultados_faixas,
    distancia_maxima=5,
    minimo_faixas=3,
    max_faixas_ausentes=1,
    altura_imagem=None,
    largura_imagem=None
):
    """
    Reconstrói linhas verticais a partir
    das posições detectadas por faixas.

    Permite pequenas interrupções, estima
    inclinações e estende as linhas às bordas.

    Retorna uma lista de linhas com pontos (x, y).
    """

    if not resultados_faixas:
        return []

    # 7.1. Determinação da altura
    if altura_imagem is None:
        altura_imagem = max(
            resultado["y_fim"]
            for resultado in resultados_faixas
        )

    linhas = []

    # 7.2. Percorrer as faixas
    for indice_faixa, resultado in enumerate(
        resultados_faixas
    ):

        y_inicio = resultado["y_inicio"]
        y_fim = resultado["y_fim"]

        y_centro = (y_inicio + y_fim) // 2

        posicoes = sorted(resultado["posicoes"])

        candidatos = []

        # 7.3. Previsão da posição horizontal
        for indice_linha, linha in enumerate(linhas):

            diferenca_faixas = (
                indice_faixa - linha["ultima_faixa"]
            )

            if diferenca_faixas < 1:
                continue

            if diferenca_faixas > max_faixas_ausentes + 1:
                continue

            pontos = linha["pontos"]

            ultimo_x, ultimo_y = pontos[-1]

            x_previsto = ultimo_x

            if len(pontos) >= 2:

                anterior_x, anterior_y = pontos[-2]

                delta_y = ultimo_y - anterior_y

                if delta_y != 0:

                    inclinacao = (
                        (ultimo_x - anterior_x) / delta_y
                    )

                    x_previsto = (
                        ultimo_x
                        + inclinacao * (y_centro - ultimo_y)
                    )

            for indice_posicao, x in enumerate(posicoes):

                distancia = abs(x - x_previsto)

                tolerancia = (
                    distancia_maxima * diferenca_faixas
                )

                if distancia <= tolerancia:

                    candidatos.append((
                        distancia,
                        indice_linha,
                        indice_posicao
                    ))

        # 7.4. Associação dos segmentos
        candidatos.sort(key=lambda item: item[0])

        linhas_utilizadas = set()
        posicoes_utilizadas = set()

        for _, indice_linha, indice_posicao in candidatos:

            if indice_linha in linhas_utilizadas:
                continue

            if indice_posicao in posicoes_utilizadas:
                continue

            x = int(posicoes[indice_posicao])

            linhas[indice_linha]["pontos"].append(
                (x, y_centro)
            )

            linhas[indice_linha]["ultima_faixa"] = (
                indice_faixa
            )

            linhas_utilizadas.add(indice_linha)
            posicoes_utilizadas.add(indice_posicao)

        # 7.5. Criação de novas linhas
        for indice_posicao, x in enumerate(posicoes):

            if indice_posicao in posicoes_utilizadas:
                continue

            linhas.append({
                "pontos": [(int(x), y_centro)],
                "ultima_faixa": indice_faixa
            })

    # 7.6. Filtragem das linhas
    linhas_validas = [
        linha["pontos"]
        for linha in linhas
        if len(linha["pontos"]) >= max(2, minimo_faixas)
    ]

    linhas_reconstruidas = []

    # 7.7. Extensão ao topo e à base
    for pontos in linhas_validas:

        pontos = sorted(
            pontos,
            key=lambda ponto: ponto[1]
        )

        primeiro_x, primeiro_y = pontos[0]
        segundo_x, segundo_y = pontos[1]

        penultimo_x, penultimo_y = pontos[-2]
        ultimo_x, ultimo_y = pontos[-1]

        # Inclinação na extremidade superior
        delta_y_topo = segundo_y - primeiro_y

        if delta_y_topo == 0:
            continue

        inclinacao_topo = (
            (segundo_x - primeiro_x) / delta_y_topo
        )

        x_topo = (
            primeiro_x
            - inclinacao_topo * primeiro_y
        )

        # Inclinação na extremidade inferior
        delta_y_base = ultimo_y - penultimo_y

        if delta_y_base == 0:
            continue

        inclinacao_base = (
            (ultimo_x - penultimo_x) / delta_y_base
        )

        y_base = altura_imagem - 1

        x_base = (
            ultimo_x
            + inclinacao_base * (y_base - ultimo_y)
        )

        # Limita as posições à largura da imagem
        if largura_imagem is not None:

            x_topo = np.clip(
                x_topo,
                0,
                largura_imagem - 1
            )

            x_base = np.clip(
                x_base,
                0,
                largura_imagem - 1
            )

        linha_completa = [
            (int(round(x_topo)), 0)
        ]

        linha_completa.extend(pontos)

        linha_completa.append(
            (int(round(x_base)), y_base)
        )

        linhas_reconstruidas.append(
            linha_completa
        )

    # 7.8. Ordenação da esquerda para a direita
    linhas_reconstruidas.sort(
        key=lambda pontos: np.mean(
            [x for x, y in pontos]
        )
    )

    return linhas_reconstruidas
