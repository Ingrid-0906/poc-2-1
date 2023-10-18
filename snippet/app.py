import pandas as pd
import numpy as np
import math
import time
import scipy.optimize as sco
import matplotlib.ticker as mtick
from datetime import date, datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.font_manager
font_list = matplotlib.font_manager.findSystemFonts()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class PortfolioOptimization:

    def __init__(self, i_performance, qtde_anos, tickers):
        self.ip = i_performance
        self.qtde_anos = qtde_anos
        self.tickers = tickers
        self.peso_otimo = float()
        
        
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
                
        for ticker in tickers:
            try:
                ticker = yf.Ticker(ticker)
                precos[ticker] = ticker.history(start=date_start, end=date_today)["Close"] 
            except Exception as e:
                print(f"Erro ao processar o ativo {ticker}: {e}")
                continue
            
        novo_nome = []
        for i in range(len(tickers)):
            elemento = str(tickers[i][:tickers[i].find('.SA')]) # elimina o '.SA'
            novo_nome.append(elemento)

        precos.columns = novo_nome
        
        return precos


    def calculate_portfolio_stats(self, precos):
        """
             Calcula oo retorno, volatilidade, risco e covariancia entre os ativo(s)
             
             vars:
             - retornos: retorno atual
             - rotulo: nome dos ativos
             - e_r: media dos retornos
             - vol: volatilidade dos retornos
             - mat_cov: matriz de covariancia dos ativos
             
             return:
             - retornos, rotulo, e_r, vol, mat_cov
        """
        retornos = precos.pct_change().dropna()
        rotulo = retornos.columns.to_list()
        e_r = retornos.mean()
        vol = retornos.std()
        mat_cov = retornos.cov()
        return retornos, rotulo, e_r, vol, mat_cov
    
    
    def perfomance_carteira(self, retornos_ip, retornos):
        """
            Compara os ativos um por um frente ao indice maior da bolsa a qual faz parte
            
            vars:
            - ip_cum: acumulado dos retornos do indice que indica performance
            - ativos_cum: acumulado do indice da carteira
            - dt_general: a união entre os dois
            
            return:
            - dt_general
        """
        ip_cum = (1 + retornos_ip).cumprod()
        ativos_cum = (1 + retornos).cumprod()
        dt_general = ip_cum.merge(ativos_cum,  how='inner', on='Date')
        dt_general.index = dt_general.index.strftime("%Y-%m-%d")
        return dt_general.to_json(orient='records')
        

    def generate_portfolios(self, n_portfolios):
        p_ret = []
        p_vol = []
        p_pesos = []

        _, _, e_r, _, mat_cov = self.calculate_portfolio_stats(self.precos)

        n_ativos = len(self.tickers)

        for _ in range(n_portfolios):
            pesos = np.random.random(n_ativos)
            pesos = pesos / np.sum(pesos)
            p_pesos.append(pesos)

            returns = np.dot(pesos, e_r)
            p_ret.append(returns)

            var = mat_cov.mul(pesos, axis=0).mul(pesos, axis=1).sum().sum()
            dp = np.sqrt(var)
            p_vol.append(dp)

        p_ret = np.array(p_ret)
        p_vol = np.array(p_vol)

        return p_ret, p_vol, p_pesos
    
    
    
    def TBL_All_Portfolios(self, dt):
        TBL_carteiras = {
            'setor': [],
            'ticker': [],
            'weight': [],
            'w_best': [],
            'return': [],
            'risk': []
        }
        
        dt['Return_Total'] = 0
        dt['Return_Total'] = dt.sum()
        
        for x in range(len(dt)):
            TBL_carteiras['ticker']  = dt.iloc[x]
            TBL_carteiras['weight'] = dt.iloc[0] / np.sum(dt['Return_Total'].iloc[x])
        
        


    def calculate_risk_free_rate(self, tx):
        rf = ((tx + 1) ** (1 / 252)) - 1
        print('Taxa livre de risco diária: ', '{:.2%}'.format(rf))
        return rf
    
    
    def optimize_portfolio(self, tpl1, e_r, mat_cov, n_ativos, rotulo):
        
        def port_vol(pesos): #Função de cálculo de risco
            return math.sqrt(np.dot(pesos,np.dot(mat_cov, pesos)))
        
        def port_ret(pesos): #Função de cálculo de retorno
            return np.sum(e_r*pesos)
        
        tpl2 = 1 - tpl1
        restri = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bnds = tuple((tpl1, tpl2) for _ in range(n_ativos))
        pesos_i = np.array(n_ativos * [1 / n_ativos])

        otim_menor_vol = sco.minimize(port_vol, pesos_i, method='SLSQP', bounds=bnds, constraints=restri)

        self.peso_otimo = otim_menor_vol['x'].round(2)
        ret_otimo = port_ret(otim_menor_vol['x'])
        vol_otimo = port_vol(otim_menor_vol['x'])

        print('Rótulos: ', rotulo, '\n',
              'Pesos  da carteira de menor risco:', self.peso_otimo, '\n',
              'Retorno da carteira de menor risco: ', '{:.2%}'.format(ret_otimo), '\n',
              'Risco da carteira de menor risco: ', '{:.2%}'.format(vol_otimo))
        
        return otim_menor_vol['x']
        
        
    def calculate_efficient_frontier(self, trets, e_r, mat_cov, tpl1, n_ativos):
        
        def port_vol(pesos): #Função de cálculo de risco
            return math.sqrt(np.dot(pesos,np.dot(mat_cov, pesos)))
        
        def port_ret(pesos): #Função de cálculo de retorno
            return np.sum(e_r*pesos)
        
        pesos_i=np.array(n_ativos*[1/n_ativos])
        tpl2 = 1-tpl1
        bnds=tuple((tpl1,tpl2) for _ in range(n_ativos)) # Gera uma tupla de (tpl1,tpl1) para cada ativo
        cons = [{'type': 'eq', 'fun': lambda x: port_ret(x) - tret}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        tvols = []

        for tret in trets:
            res = sco.minimize(port_vol, pesos_i, method='SLSQP', bounds=bnds, constraints=cons)
            tvols.append(res['fun'])

        tvols = np.array(tvols)
        return tvols


    def _3D_plot_efficient_frontier(self, mat_cov, e_r, p_ret, p_vol, otim_menor_vol):
        
        def port_vol(pesos): #Função de cálculo de risco
            return math.sqrt(np.dot(pesos,np.dot(mat_cov, pesos)))
        
        def port_ret(pesos): #Função de cálculo de retorno
            return np.sum(e_r*pesos)
        
        rf = self.calculate_risk_free_rate(tx)
        
        # Creating figure
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
 
        # Creating plot
        ax.scatter3D(p_vol, p_ret, (p_ret-rf)/p_vol, c=(p_ret-rf)/p_vol, cmap='RdYlGn')
        plt.title("Carteira de menor risco", fontsize=14)
        ax.set_xlabel('Risco', fontweight ='bold') 
        ax.set_ylabel('Retorno', fontweight ='bold') 
        ax.set_zlabel('Indice Sharpe', fontweight ='bold')
        ax.grid(b = True, color ='grey', linestyle ='-.', linewidth = 0.3, alpha = 0.2)
        ax.plot(port_vol(otim_menor_vol), port_ret(otim_menor_vol), (port_ret(otim_menor_vol)-rf)/port_vol(otim_menor_vol), 'bo', markersize=15)
        plt.show()
        
        
    def plot_efficient_frontier(self, mat_cov, e_r, p_ret, p_vol, tvols, trets, otim_menor_vol):
        def port_vol(pesos): #Função de cálculo de risco
            return math.sqrt(np.dot(pesos,np.dot(mat_cov, pesos)))
        
        def port_ret(pesos): #Função de cálculo de retorno
            return np.sum(e_r*pesos)
        
        rf = self.calculate_risk_free_rate(tx)
        
        plt.figure(figsize=(10,6))
        plt.scatter(p_vol,p_ret, c=(p_ret-rf)/p_vol, marker='.', cmap='RdYlGn')
        plt.plot(tvols,trets,'b', lw=2)
        plt.plot(port_vol(otim_menor_vol['x']), port_ret(otim_menor_vol['x']),'bo', markersize=15)
        plt.title('Carteiras com Fronteira Eficiente', fontsize=14)
        plt.xlabel('Risco', fontweight ='bold')
        plt.ylabel('Retorno', fontweight ='bold')
        plt.colorbar(label='Índice de Sharpe', fontweight ='bold')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.show()
        

    def best_alocation(label, weight):
        font = {'family' : 'LiberationSans', 'weight' : 'bold', 'size' : 16}
        #font = {'weight' : 'bold', 'size' : 16}
        
        def addlabels(label, weight):
            for i in range(len(label)):
                #plt.text(i,y[i],y[i], ha = 'center')
                plt.text(i, weight[i], '{:.2%}'.format(weight[i]), ha = 'center', **font )

        fig, ax =plt.subplots(figsize=(10,6))
        plt.bar(label, weight, color='maroon' )
        addlabels(label, weight)
        plt.title('Melhor Alocação e Menor Volatilidade', fontsize= 14)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        

    def run(self, tx, tpl1, n_portfolios):
        # definir quais ações são negociadas na bolsa
        
        n_ativos = len(self.tickers)
        self.precos = self.get_yf_data(self.tickers)
        self.ip = self.get_yf_data(self.ip)
        retornos, rotulo, e_r, _, mat_cov = self.calculate_portfolio_stats(self.precos)
        
        # Para gerar dado de desempenho
        retornos_ip, _, ip_e_r, _, _ = self.calculate_portfolio_stats(self.ip)
        
        p_ret, p_vol = self.generate_portfolios(n_portfolios)
        otim_menor_vol = self.optimize_portfolio(tx, tpl1, e_r, mat_cov, n_ativos, rotulo)
        trets = np.linspace(p_ret.min(), p_ret.max(), n_portfolios)
        tvols = self.calculate_efficient_frontier(trets, e_r, mat_cov, tpl1, n_ativos)
        
        
        # RETORNA COM UM JSON! DESEMPENHO CARTEIRA VS. INDICE GERAL
        performance = self.perfomance_carteira(retornos_ip, retornos)
        
        # RETORNA COM JSON TABELA COMPLETA DOS MELHORES 10 PORTFOLIOS


if __name__ == "__main__":
    qtde_anos = 1
    i_performance = ['^BVSP']
    tickers = ['EGIE3.SA', 'ITSA4.SA', 'VIVT3.SA', 'BBDC4.SA', 'SBSP3.SA','PETR4.SA', 'VALE3.SA']
    tx = 0.08
    tpl1 = 0.01
    n_portfolios = 1000

    portfolio_optimizer = PortfolioOptimization(i_performance, qtde_anos, tickers)
    portfolio_optimizer.run(tx, tpl1, n_portfolios)