# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 19:57:01 2024

@author: torug
"""

#%% Instalando os pacotes

# Instalando pacotes necessários para análise de dados e visualização
!pip install pandas
!pip install numpy
!pip install factor_analyzer
!pip install sympy
!pip install scipy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install pingouin
!pip install pyshp

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go
import shapefile as shp

#%% Importando o banco de dados

# Carregando o banco de dados a partir de um arquivo Excel
atlas = pd.read_excel('atlas_ambiental.xlsx')

# Renomeando a coluna 'mort_ext' para 'causasext'
atlas = atlas.rename(columns={'mort_ext': 'causasext'})

# Removendo colunas não necessárias para a análise de componentes principais
atlas_pca = atlas.drop(columns=['cód_ibge', 'distritos'])

# Descrevendo estatísticas básicas do conjunto de dados
describe = atlas_pca.describe()

# Calculando a matriz de correlação
corr = atlas_pca.corr()

# Calculando o teste de esfericidade de Bartlett
bartlett, p_value = calculate_bartlett_sphericity(atlas_pca)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

# Ajustando o modelo de análise fatorial com 9 fatores sem rotação
fa = FactorAnalyzer(n_factors=9, rotation=None, method='principal').fit(atlas_pca)

# Obtendo os autovalores
autovalores = fa.get_eigenvalues()[0]

print(autovalores)

# Ajustando o modelo de análise fatorial com 2 fatores sem rotação
fa = FactorAnalyzer(n_factors=2, rotation=None, method='principal').fit(atlas_pca)

# Obtendo a variância explicada pelos fatores
autovalores_fatores = fa.get_factor_variance()

# Criando uma tabela com os autovalores e variâncias
tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

# Obtendo as cargas fatoriais
cargas_fatoriais = fa.loadings_

# Criando uma tabela com as cargas fatoriais
tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = atlas_pca.columns

print(tabela_cargas)

# Transformando os dados originais nos novos fatores
fatores = pd.DataFrame(fa.transform(atlas_pca))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Concatenando os novos fatores ao dataframe original
atlas = pd.concat([atlas.reset_index(drop=True), fatores], axis=1)

# Adotando critério para o ranking somente com o primeiro fator
dados_mapa = atlas[['cód_ibge', 'distritos', 'Fator 1']].sort_values(by=['cód_ibge'], ascending=True).reset_index(drop=True)

#%% Lendo o shapefile

# Lendo o shapefile que contém os dados geográficos dos distritos
sf = shp.Reader("DEINFO_DISTRITO")

# Função para ler e transformar o shapefile em um DataFrame
def read_shapefile(sf):
    fields = [x[0] for x in sf.fields][1:]
    records = [y[:] for y in sf.records()]
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

# Convertendo o shapefile em um DataFrame
dist = read_shapefile(sf)

# Fonte: http://dados.prefeitura.sp.gov.br/dataset/distritos

# Convertendo a coluna 'COD_DIST' para numérica e ordenando
dist['COD_DIST'] = pd.to_numeric(dist['COD_DIST'])
dist = dist.sort_values(by=['COD_DIST'], ascending=True).reset_index(drop=True)

#%% Função para plotar o mapa dos distritos

def plot_map(df, x_lim=None, y_lim=None, figsize=(8,11)):
    plt.figure(figsize=figsize)
    id = 0
    for coordinates in df.coords:
        x = [i[0] for i in coordinates]
        y = [i[1] for i in coordinates]
        plt.plot(x, y, 'k')
        
        if (x_lim == None) & (y_lim == None):
            x0 = np.mean(x)
            y0 = np.mean(y)
            plt.text(x0, y0, id, fontsize=5)
        id = id + 1
    
    if (x_lim != None) & (y_lim != None):     
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        
    plt.axis('off')

# Plotando o mapa dos distritos
plot_map(dist)

#%% Função para calcular as cores com base nos valores do fator

def calc_color(data):
    new_data = pd.qcut(data, 6, labels=list(range(6)))
    paleta = sns.color_palette('YlOrBr', n_colors=6)
    color_sq = paleta.as_hex()
    color_ton = []
    for val in new_data:
        color_ton.append(color_sq[val])
    return color_ton

#%% Função para plotar o mapa com preenchimento de cores

def plot_map_fill_multiples_ids_3(df, title, distrito_id, color_ton, x_lim=None, y_lim=None, figsize=(8,11)):
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16)

    for coordinates in df.coords:
        x = [i[0] for i in coordinates]
        y = [i[1] for i in coordinates]
        plt.plot(x, y, 'k')
            
    for id in distrito_id:
        shape_ex = df.coords[id]
        x_lon = np.zeros((len(shape_ex), 1))
        y_lat = np.zeros((len(shape_ex), 1))
        for ip in range(len(shape_ex)):
            x_lon[ip] = shape_ex[ip][0]
            y_lat[ip] = shape_ex[ip][1]
        ax.fill(x_lon, y_lat, color_ton[distrito_id.index(id)])
        x0 = np.mean(x_lon)
        y0 = np.mean(y_lat)
        plt.text(x0, y0, id, fontsize=6)
    
    if (x_lim != None) & (y_lim != None):     
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    
    plt.axis('off')

#%% Plotando o mapa com as cores

# Definindo os distritos e os dados do fator 1
distritos = list(np.arange(96)) # id_distritos
data = list(dados_mapa['Fator 1']) # indicador socioeconômico (fator)
color_ton = calc_color(data) # tonalidade de cores

# Plotando o mapa com preenchimento de cores
plot_map_fill_multiples_ids_3(dist, 'Indicador Socioeconômico', distritos, color_ton)
