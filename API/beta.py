from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import yfinance as yf
import scipy.optimize as sco
from datetime import date, datetime, timedelta

app = Flask('__name__')

class PortfolioBeta:

    def __init__(self, qtde_anos, tickers, indicator, invested, proporcoes):
        self.qtde_anos = qtde_anos
        self.tickers = tickers
        self.indicator = indicator
        self.invested = invested
        self.proporcoes = proporcoes
        self.precos = pd.DataFrame()
        
        
    def get_yf_data(self, tickers):
        """
            Faz o download da fonte de dados do preço de fechamento junto ao yahoo finance.
            
            vars:
                - precos: Dataframe com todos os tickers e seus respectivos valores de fechamento passados
                - tickers: lista com nome de ativos para buscar preços
                
            return:
                - precos
        """
        
        today = date.today()
        date_today = today.strftime("%Y-%m-%d")
        inicio = datetime.now() + timedelta(days=(-365.25 * self.qtde_anos))
        date_start = inicio.strftime("%Y-%m-%d")
        precos = pd.DataFrame()
        novo_nome = []
                
        for ticker in tickers:
            try:
                ticker = yf.Ticker(ticker)
                precos[ticker] = ticker.history(start=date_start, end=date_today)["Close"]
            except Exception as e:
                print(f"Erro ao processar o ativo {ticker}: {e}")
                continue
            
        for i in range(len(tickers)):
            elemento = str(tickers[i][:tickers[i].find('.SA')])
            novo_nome.append(elemento)

        precos.columns = novo_nome
        return precos


    def calculate_portfolio_stats(self, precos):
        """
             Calcula oo retorno, volatilidade, risco e covariancia entre os ativo(s)
             
             pars:
             - precos: dataframe com os precos historicos de todos os ativos da carteira
             
             vars:
             - retornos: retorno atual
             - retorno_log: retornos acumulados durante o periodo apartir do primeiro dia listado
             
             return:
             - retorno_log
        """
        tik = self.tickers
        ind = self.indicator
        tikind = ind + tik
        precos = self.get_yf_data(tikind)
        
        retornos_log = np.log(precos/precos.shift(1)).dropna()
        
        return retornos_log


    def beta_ticker(self, returns_asset, returns_market):
        """
            Calcula o beta de um ativo em relação ao índice de mercado.

            Parâmetros:
            - returns_asset: retornos do ativo
            - returns_market: retornos do índice de mercado

            Retorna o valor do beta.
        """
        # Calcula a covariância entre os retornos da ação e do mercado
        covariance = np.cov(returns_asset, returns_market)[0, 1]
        # Calcula a variância dos retornos do mercado
        variance_market = np.var(returns_market)
        # Calcula o beta como a covariância dividida pela variância do mercado
        beta = covariance / variance_market

        return beta
    
    
    def calcular_betas(self):
        
        self.precos = self.get_yf_data(self.tickers)
        retornos_log = self.calculate_portfolio_stats(self.precos)
        
        # Seleciona os retornos das ações e do índice
        for i in range(len(self.indicator)):
            indice = str(self.tickers[i][:self.tickers[i].find('.SA')])
        returnos_bova11 = retornos_log[indice]

        # Dicionário para armazenar os betas de cada ativo
        betas = {}

        # Percorre as colunas do DataFrame para calcular o beta de cada ativo
        for col in retornos_log.columns:
            if col != indice:
                returnos_ativo = retornos_log[col]
                beta_ativo = self.beta_ticker(returnos_ativo, returnos_bova11)
                betas[col] = beta_ativo

        # Cria um novo DataFrame com os betas
        df_betas = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta'])
        return df_betas
    
    
    def calcular_beta_carteira(self, df, proporcoes=None):
        """
        Calcula o beta da carteira com base nos betas dos ativos e suas proporções.

        Parâmetros:
        - df: DataFrame contendo os betas dos ativos
        - proporcoes: lista opcional com as proporções de cada ativo na carteira

        Retorna o valor do beta da carteira.
        """

        betas = df['Beta'].tolist()
        
        proporcoes = list()
        if self.proporcoes is None or sum(self.proporcoes) == 0:
            for x in range(len(self.invested)):
                proporcoes.append(round((self.invested[x] / sum(self.invested)),2))
        else:
            # Verifica se a soma das proporções é igual a 1
            soma_proporcoes = sum(proporcoes)
            if soma_proporcoes != 1:
                print('Alerta: A soma das proporções DEVE SER igual a 1.')
                StopIteration()
                return None

        beta_carteira = sum(beta * proporcao for beta, proporcao in zip(betas, proporcoes))
        return betas, proporcoes, beta_carteira

     
    def run_beta_carteira(self):
        
        df_betas = self.calcular_betas()
        betas, proporcoes, beta_carteira = self.calcular_beta_carteira(df_betas, self.proporcoes)

        if beta_carteira is not None:
            beta_final = {
                'betas': betas,
                'proporcoes': proporcoes,
                'beta carteira': [beta_carteira]
            }
            return beta_final
        else:
            return {'404 - ERROR': 'Fail to process.'}

   
   
# Rota para calcular o beta do portfólio
@app.route('/beta_portfolio', methods=['POST'])
def beta_portfolio():
    data = request.get_json()
    tickers = data['tickers']
    invested = data['invested']
    qtde_anos = data['qtde_anos']
    indicator = data['indicator']
    proporcoes = data['proporcoes']
        
    beta_api = PortfolioBeta(qtde_anos, tickers, indicator, invested, proporcoes)
    PB = beta_api.run_beta_carteira()
    return jsonify(PB)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
