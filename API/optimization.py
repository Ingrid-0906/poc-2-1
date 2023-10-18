from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import math
import yfinance as yf
import scipy.optimize as sco
from datetime import date, datetime, timedelta

app = Flask('__name__')

class PortfolioOptimization:

    def __init__(self, tx, tpl1, n_portfolios, qtde_anos, tickers, indicator, invested):
        self.tx = tx
        self.tpl1 = tpl1
        self.n_portfolios = n_portfolios
        self.qtde_anos = qtde_anos
        self.tickers = tickers
        self.indicator = indicator
        self.invested = invested
        self.precos = pd.DataFrame()
        self.peso_otimo = float()
        self.ret_otimo = float()
        self.vol_otimo = float()
        self.p_ret = float()
        self.p_vol = float()
        self.p_pesos = float()
        self.otim_menor_vol = float()
        
        
    def get_yf_data(self, tickers):
        """
            Faz o download da fonte de dados do preço de fechamento junto ao yahoo finance.
            
            pars:
             - tickers: nome dos ativos
            
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
              - precos: dataframe com todo o historico de precos da carteira
             
             vars:
              - retornos: retorno atual
              - rotulo: nome dos ativos
              - e_r: media dos retornos
              - vol: volatilidade dos retornos
              - mat_cov: matriz de covariancia dos ativos
             
             return:
             - retornos, rotulo, e_r, vol, mat_cov
        """
        retornos = precos.pct_change().dropna() # retirar registros com 'NaN'
        rotulo= retornos.columns.to_list()
        e_r=retornos.mean()
        vol=retornos.std()
        mat_cov=retornos.cov()
        return retornos, rotulo, e_r, vol, mat_cov
    

    def generate_portfolios(self):
        """
            Simulation that generates n_portifolios accordingly their individual but collective return and risk.
            
            vars:
             - p_ret: total return of a portfolio
             - p_vol: total volatility of a portfolio
             - p_pesos: individual weight of each stock
            
            return:
             - p_ret, p_vol, p_pesos
        """
        
        p_ret = []
        p_vol = []
        p_pesos = []

        _, _, e_r, _, mat_cov = self.calculate_portfolio_stats(self.precos)

        n_ativos = len(self.tickers)

        for _ in range(self.n_portfolios):
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
    
    
    def optimize_portfolio(self, e_r, mat_cov, n_ativos):
        """
            Cria a carteira otimizada escolhendo as posições que mantem ou supera em retorno e miniminiza o risco da carteira toda.
            
            pars:
             - e_r: media dos retornos
             - mat_cov: matriz de covariancia da carteira
             - n_ativos: # de ativos da carteira
            
            vars:
             - tpl2: máximo disponivel para ser alocado
             - restri: é do tipo igualdade que informa que a volatilidade deve ter como valor maximo 1
             - bnds: limites para cada ativo
             - pesos_i: pesos iguais para todos começarem
        """
        
        def port_vol(pesos): #Função de cálculo de risco
            return math.sqrt(np.dot(pesos,np.dot(mat_cov, pesos)))
        
        def port_ret(pesos): #Função de cálculo de retorno
            return np.sum(e_r*pesos)
        
        tpl2 = 1 - self.tpl1
        restri = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bnds = tuple((self.tpl1, tpl2) for _ in range(n_ativos))
        pesos_i = np.array(n_ativos * [1 / n_ativos])

        otim_menor_vol = sco.minimize(port_vol, pesos_i, method='SLSQP', bounds=bnds, constraints=restri)
        self.peso_otimo = otim_menor_vol['x']
        self.ret_otimo = port_ret(otim_menor_vol['x'])
        self.vol_otimo = port_vol(otim_menor_vol['x'])
        return otim_menor_vol['x']


    def run_profile_ativos(self):
        """
            Gera detalhes relacionados aos ativos.
            
            vars:
             - profile: dict with details about the stocks, it includes sector, actual weight, new weight, return and risk.
            
            return:
             - profile
        """
        
        novo_nome = list()
        _, _, e_r, vol, _ = self.calculate_portfolio_stats(self.precos)
        profile = {'sector': [], 'industry': [], 'ticker': [], 'peso_wallet': [], 'peso_new': self.peso_otimo, 'return': e_r, 'risk': vol}
        
        for ticker in self.tickers:
            try:
                ticker = yf.Ticker(ticker)
                profile['sector'].append(ticker.info['sector'])
                profile['industry'].append(ticker.info['industry'])
                profile['ticker'].append(ticker)
            except Exception as e:
                print(f"Erro ao processar o ativo {ticker}: {e}")
                continue
        
        for x in range(len(self.invested)):
            profile['peso_wallet'].append(round((self.invested[x] / sum(self.invested)),2))
                
        for i in range(len(self.tickers)):
            elemento = str(self.tickers[i][:self.tickers[i].find('.SA')])
            novo_nome.append(elemento)
        
        profile['ticker'] = novo_nome
        dt_profile = pd.DataFrame(data=profile)
        return dt_profile.to_dict(orient='records')
        

    def run_performance(self):
        """
            Calcula a performance da carteira frente ao seu principal indicador na bolsa de valores.
            
            vars:
             - indicador: nome dos tickers que faz parte da carteira 
             - precos: dataframe com todos os preços temporais dos ativos que compoem a carteira
             - ret_market: retorno dos ativos que faz parte da da carteira junto ao indicador ao longo do tempo selecionado
            
            return:
             - self.perfomance_carteira(): funcao
        """
        tik = self.tickers
        ind = self.indicator
        tikind = ind + tik
        precos = self.get_yf_data(tikind)
        retornos = precos.pct_change().dropna()
        
        ativos_cum = round((1 + retornos).cumprod(), 2)
    
        ativos_cum.insert(0, "Date", retornos.index.strftime("%Y-%m-%d"))
        ativos_cum.index = range(len(ativos_cum))
        return ativos_cum.to_dict(orient='records')
    
    
    def run_portfolios(self):
        """
            Create a dict with all n_portfolio with their respective weight, return and volatility.
            
            vars:
             - data_portfolios: dicionario com os principais dados dos tickers, como: nome, peso, retorno e volatilidade
            
           return:
             - data_portfolios
        """
        
        data_portfolios = {
            'ticker': self.tickers,
            'weight': self.p_pesos,
            'return': self.p_ret,
            'volatility': self.p_vol
        }
        
        return data_portfolios
        
    
    def run_plot_3D(self):
        """
             Cria os dados dos eixos para plotar os dados em 3d.
             
             vars:
             - e_r: media do retorno
             - mat_cov: matriz de covariancia
             
             return:
             - self._3D_plot_efficient_frontier(): funcao
        """
        
        _, _, e_r, _, mat_cov = self.calculate_portfolio_stats(self.precos)
        
        
        def port_vol(pesos): #Função de cálculo de risco
            return math.sqrt(np.dot(pesos,np.dot(mat_cov, pesos)))
        
        def port_ret(pesos): #Função de cálculo de retorno
            return np.sum(e_r*pesos)
        
        rf = ((self.tx + 1) ** (1 / 252)) - 1
        
        x = list(self.p_vol)
        y = list(self.p_ret)
        z = list((self.p_ret - rf) / self.p_vol)
        
        point_x = port_vol(self.otim_menor_vol)
        point_y = port_ret(self.otim_menor_vol)
        point_z = (port_ret(self.otim_menor_vol)-rf)/port_vol(self.otim_menor_vol)
        
        return {'x': x, 'y': y, 'z': z, 'point_x': point_x, 'point_y': point_y, 'point_z': point_z}
    
    
    def run_start_data(self):
        """
            Inicia a chamada coletando o preço historico dos ativos e gerando a otimizacao do portifolio.
            
            vars:
             - precos: dataframe com o preco historico
             - p_ret, p_vol, p_pesos: retorno, volatilidade e peso dos mil porfolios gerados
             - e_r, mat_cov: media dos retornos e matriz de covariancia da carteira
             - otim_menor_vol: valor x que representa o total otimizado
        """
        
        self.precos = self.get_yf_data(self.tickers)
        self.p_ret, self.p_vol, self.p_pesos = self.generate_portfolios()
        n_ativos = len(self.tickers)
        _, _, e_r, _, mat_cov = self.calculate_portfolio_stats(self.precos)
        self.otim_menor_vol = self.optimize_portfolio(e_r, mat_cov, n_ativos)
   
   
# Rota para otimizar o portfólio // deve ser o primeiro!
@app.route('/optimize_portfolio', methods=['POST'])
def optimize_portfolio():
    data = request.get_json()
    tx = data['tx']
    tickers = data['tickers']
    invested = data['invested']
    qtde_anos = data['qtde_anos']
    indicator = data['indicator']
    
    global portfolio_optimizer 
    portfolio_optimizer = PortfolioOptimization(tx, tpl1, n_portfolios, qtde_anos, tickers, indicator, invested)
    portfolio_optimizer.run_start_data()
    
    result = {
        "OptimalWeights": portfolio_optimizer.peso_otimo.tolist(),
        "OptimalReturn": portfolio_optimizer.ret_otimo,
        "OptimalVolatility": portfolio_optimizer.vol_otimo
    }
    return jsonify(result)

# Rota para obter detalhes sobre os ativos
@app.route('/profile_ativos', methods=['GET'])
def get_profile_ativos():
    profile = portfolio_optimizer.run_profile_ativos()
    return jsonify(profile)

# Rota para calcular o desempenho da carteira
@app.route('/performance', methods=['GET'])
def calculate_performance():
    performance = portfolio_optimizer.run_performance()
    return jsonify(performance)

# Rota para gerar um gráfico 3D da fronteira eficiente
@app.route('/plot_3d', methods=['GET'])
def generate_3d_plot():
    plot_data = portfolio_optimizer.run_plot_3D()
    return jsonify(plot_data)

@app.route('/portfolios', method=['GET'])
def get_portfolios():
    all_portfolios = portfolio_optimizer.run_portfolios()
    return jsonify(all_portfolios)


if __name__ == "__main__":
    # VARIAVEIS PADRÃO
    tpl1 = 0.01  # mínimo de 1%
    n_portfolios = 1000  # mínimo de 1 mil

    app.run(debug=True, port=8080)
