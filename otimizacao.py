import streamlit as st
import pandas as pd
import numpy as np
import math
import yfinance as yf
import scipy.optimize as sco
from datetime import date, datetime, timedelta
from plotly import graph_objs as go

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
        precos = pd.DataFrame()
        novo_nome = []
                
        for ticker in tickers:
            try:
                ticker = yf.Ticker(ticker)
                precos[ticker] = ticker.history(period=self.qtde_anos, interval='1d')["Close"]
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
        
        profile = {'sector': [], 
                   'industry': [], 
                   'ticker': [], 
                   'peso_wallet': [], 
                   'peso_new': self.peso_otimo, 
                   'return': e_r, 
                   'risk': vol}
        
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
        dt_profile = pd.DataFrame.from_dict(a, orient='index')
        dt_profile = dt_profile.transpose()
        return dt_profile
        

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
        tikind = tik._append(ind).reset_index()[0]
        precos = self.get_yf_data(tikind)
        retornos = precos.pct_change().dropna()
        ativos_cum = round((1 + retornos).cumprod(), 2)
        
        ativos_cum.insert(0, "Date", retornos.index.strftime("%Y-%m"))
        new_ativos_cum = ativos_cum.groupby(ativos_cum['Date']).agg(['mean'])
        new_ativos_cum.columns = new_ativos_cum.columns.droplevel(1)
        return new_ativos_cum
        
    
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
        dt_port = pd.DataFrame(data=data_portfolios)
        return dt_port
        
    
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
        
        x = list(self.p_vol*100)
        y = list(self.p_ret*100)
        z = list(((self.p_ret - rf) / self.p_vol)*100)
        
        point_x = port_vol(self.otim_menor_vol)*100
        point_y = port_ret(self.otim_menor_vol)*100
        point_z = (port_ret(self.otim_menor_vol)-rf)/port_vol(self.otim_menor_vol)*100
        
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
        result = {
            "OptimalWeights": self.peso_otimo.tolist(),
            "OptimalReturn": self.ret_otimo,
            "OptimalVolatility": self.vol_otimo
            }
        return result
   

# Funçoes para plotar os charts
def performe_chart(df):
    benchmark = portfolio_optimizer.indicator
    
    bench = str()
    for i in range(len(benchmark)):
        bench = str(benchmark[i][:benchmark[i].find('.SA')])    
           
    fig = go.Figure()
    for col in df.columns:
        if col == bench:
            fig.add_trace(go.Scatter(x=df.index, 
                                  y=df[col], 
                                  name=col,
                                  line = dict(color='white', 
                                              width=5, 
                                              dash='dot')))
        else:
            fig.add_trace(go.Scatter(x=df.index, 
                                     y=df[col], 
                                     name=col, 
                                     line = dict(width=1)))
        
    fig.layout.update(title_text='Carteira vs. Benchmark', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)


def profile_metrica(data):
    st.markdown("#### Retorno Carteira")
    div_1, div_2, div_3, div_4, div_5 = st.columns(5)
    
    try:
        div_1.metric(data['ticker'][0], f"{round(data['return'][0]*100,2)}%", f"{round(data['risk'][0]*100,2)}% risco", delta_color="off")
        div_2.metric(data['ticker'][1], f"{round(data['return'][1]*100,2)}%", f"{round(data['risk'][1]*100,2)}% risco", delta_color="off")
        div_3.metric(data['ticker'][2], f"{round(data['return'][2]*100,2)}%", f"{round(data['risk'][2]*100,2)}% risco", delta_color="off")
        div_4.metric(data['ticker'][3], f"{round(data['return'][3]*100,2)}%", f"{round(data['risk'][3]*100,2)}% risco", delta_color="off")
        div_5.metric(data['ticker'][4], f"{round(data['return'][4]*100,2)}%", f"{round(data['risk'][4]*100,2)}% risco", delta_color="off")
    except IndexError:
        pass
    

def run_3d_chart(data_3d):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=data_3d['x'],
        y=data_3d['y'],
        z=data_3d['z'],
        name='Carteiras',
        mode='markers',
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[data_3d['point_x']], 
        y=[data_3d['point_y']], 
        z=[data_3d['point_z']],
        name='Otimizado' ,
        mode='markers',
        marker=dict(size=12)
    ))

    # tight layout
    fig.update_layout(scene=dict(xaxis_title='Risco',
                                 yaxis_title='Retorno',
                                 zaxis_title='índice Sharpe'),
                      margin=dict(l=0, r=0, b=0))
    fig.update_traces(hovertemplate='Risco=%{x}<br>Retorno=%{y}<br>Sharpe=%{z}<extra></extra>', 
                      selector=dict(type='scatter3d'))
    st.plotly_chart(fig, use_container_width=True)
             

def best_weight(data):
    st.markdown("#### Re-Alocação Carteira")
    div_1, div_2, div_3, div_4, div_5 = st.columns(5)
    
    try:
        div_1.metric(data['ticker'][0], f"{round(data['peso_new'][0]*100,2)}%", f"{round((data['peso_wallet'][0]-data['peso_new'][0])*100,2)}%")
        div_2.metric(data['ticker'][1], f"{round(data['peso_new'][1]*100,2)}%", f"{round((data['peso_wallet'][1]-data['peso_new'][1])*100,2)}%")
        div_3.metric(data['ticker'][2], f"{round(data['peso_new'][2]*100,2)}%", f"{round((data['peso_wallet'][2]-data['peso_new'][2])*100,2)}%")
        div_4.metric(data['ticker'][3], f"{round(data['peso_new'][3]*100,2)}%", f"{round((data['peso_wallet'][3]-data['peso_new'][3])*100,2)}%")
        div_5.metric(data['ticker'][4], f"{round(data['peso_new'][4]*100,2)}%", f"{round((data['peso_wallet'][4]-data['peso_new'][4])*100,2)}%")
    except IndexError:
        pass
    
    st.markdown("<p>Valores <span style='color: green;'>positivos</span> indicam que precisa comprar, e <span style='color: red;'>negativos</span> que precisa vender</p>", unsafe_allow_html=True)
  

 
if __name__ == "__main__":
    # Setting: Data Start Here...
    st.set_page_config(
        page_title = 'WalletSafe Markowitz v.1.0',
        page_icon = '✅',
        layout = 'wide'
    )
    
    st.title('WalletSafe Markowitz v.1.0')
    
    st.markdown("### 1. Passo: Informe os ativos")
    
    ticker_1, ticker_2, ticker_3, ticker_4, ticker_5 = st.columns(5)
    inv_1, inv_2, inv_3, inv_4, inv_5 = st.columns(5)
    
    with ticker_1:
        st.text_input("Ativo.SA aqui 👇", key='ticker_1', value='EGIE3.SA')
    with ticker_2:
        st.text_input("Ativo.SA aqui 👇", key='ticker_2', value='ITSA4.SA')
    with ticker_3:
        st.text_input("Ativo.SA aqui 👇", key='ticker_3', value='VIVT3.SA')
    with ticker_4:
        st.text_input("Ativo.SA aqui 👇", key='ticker_4', value='BBDC4.SA')
    with ticker_5:
        st.text_input("Ativo.SA aqui 👇", key='ticker_5', value='PETR4.SA')
        
    with inv_1:
        st.number_input("Qual valor investido?", key='inv_1', value=150.40)
    with inv_2:
        st.number_input("Qual valor investido?", key='inv_2', value=350.40)
    with inv_3:
        st.number_input("Qual valor investido?", key='inv_3', value=340.40)
    with inv_4:
        st.number_input("Qual valor investido?", key='inv_4', value=350.20)
    with inv_5:
        st.number_input("Qual valor investido?", key='inv_5', value=450.35)
        
    st.markdown("### 2. Passo: Informe a taxa, tempo e índice")
    taxa, tempo, ind = st.columns(3)
    with taxa:
        st.number_input("Taxa Selic aqui 👇", key='taxa', value=0.09)
    with tempo:
        #st.number_input("Qual a janela de anos?", key='ano', value=1.0, step=1.0)
        st.selectbox('Qual janela de tempo?', ('1mo','3mo','6mo','1y','2y','5y'), key='tempo')
    with ind:
        st.text_input("Qual Benchmark.SA?", key='ind', value='BOVA11.SA')
    
    # Gerando os valores
    data_1 = {
        "tx": float(st.session_state['taxa']),
        'tpl1': 0.01,
        'n_portfolios': 1000,
        "qtde_anos": str(st.session_state['tempo']),
        "indicator": str(st.session_state['ind'])
    }
    
    tks = [str(st.session_state['ticker_1']), 
                    str(st.session_state['ticker_2']),
                    str(st.session_state['ticker_3']),
                    str(st.session_state['ticker_4']),
                    str(st.session_state['ticker_5'])],
    inv = [float(st.session_state['inv_1']),
                     float(st.session_state['inv_2']),
                     float(st.session_state['inv_3']),
                     float(st.session_state['inv_4']),
                     float(st.session_state['inv_5'])],
    
    new_tks = list()
    new_inv = list()
    for x in tks[0]:
        if x != '':
            new_tks.append(x)
        else:
            pass
    
    for y in inv[0]:
        if y != 0:
            new_inv.append(y)
        else:
            pass
    
    dt_1 = pd.DataFrame(data=[data_1])
    dt_2 = pd.DataFrame(data={
        'tickers':new_tks,
        'invested': new_inv
    })
    
    # Chamando a classe
    global portfolio_optimizer 
    portfolio_optimizer = PortfolioOptimization(
        dt_1['tx'].iloc[0], 
        dt_1['tpl1'].iloc[0],
        dt_1['n_portfolios'].iloc[0],
        dt_1['qtde_anos'].iloc[0],
        dt_2['tickers'],
        dt_1['indicator'],
        dt_2['invested']
    )

    
    start_data = dict()
    if st.button('Calcular Otimização', type='primary'):
        start_data = portfolio_optimizer.run_start_data()
            
        st.markdown("<hr/>",unsafe_allow_html=True)
        
        
        data_profile = portfolio_optimizer.run_profile_ativos()
        performance_lines = portfolio_optimizer.run_performance()
        data_3d = portfolio_optimizer.run_plot_3D()
        
        col1, col2= st.columns([1, 3])
        with col1:
            # Resultados: Retorno e Risco
            st.markdown("#### Sua Carteira Otimizada é")
            st.markdown(f"<h1 style='color: white;'>{round(start_data['OptimalReturn']*100, 2)}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='color: gray;'>{round(start_data['OptimalVolatility']*100, 2)}% volátil</h5>", unsafe_allow_html=True)
        with col2:
            # best weights
            best_weight(data_profile)
        
        # 3d plot - Carteira Otimizada
        run_3d_chart(data_3d)
        
        # Perfil de cada ticker
        profile_metrica(data_profile)
        
        # Performance: Benchmark
        performe_chart(performance_lines)
  
        st.markdown("##### Carteira Disponível")
        st.markdown("<h6>Nota. Filtrado por menor risco</h6>", unsafe_allow_html=True)
        # Tabela com outros possiveis portifolios
        p_vol, p_ret, p_sharpe = data_3d['x'], data_3d['y'], data_3d['z']
        p_pesos = portfolio_optimizer.p_pesos
        tickers = portfolio_optimizer.tickers
        new_table = pd.DataFrame(data=p_pesos, index=range(len(p_pesos)), columns=tickers)
        table_3d = pd.DataFrame(data=zip(p_vol, p_ret, p_sharpe), columns=['volátil', 'retorno', 'sharpe'])
        table_end = pd.concat([new_table, table_3d], axis=1)
        
        plot_dt = table_end.sort_values(by=['volátil'], ascending=True)
        st.dataframe(data=plot_dt, use_container_width=True)
        
