import requests
import pandas as pd
from datetime import datetime

def baixar_dados_yahoo(ticker, data_inicio, debug=False):
    """
    Baixa dados históricos do Yahoo Finance
    
    Args:
        ticker: Código do ativo (ex: 'BTC-USD', '^GSPC', 'AAPL')
        data_inicio: Data inicial no formato 'YYYY-MM' (ex: '2014-09')
        debug: Se True, mostra informações de debug
    
    Returns:
        DataFrame com colunas 'Data' e 'Preco'
    """
    # Converter data_inicio para timestamp
    try:
        ano, mes = map(int, data_inicio.split('-'))
        data_obj = datetime(ano, mes, 1)
        timestamp_inicio = int(data_obj.timestamp())
    except ValueError:
        print(f"Erro: Data inválida. Use o formato YYYY-MM (ex: 2014-09)")
        return None
    
    # Timestamp atual
    timestamp_fim = int(datetime.now().timestamp())
    
    # URL e parâmetros
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        'period1': timestamp_inicio,
        'period2': timestamp_fim,
        'interval': '1mo',
        'events': 'history',
        'includeAdjustedClose': 'true'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print(f"Baixando dados de {ticker} desde {data_inicio}...")
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        chart = data.get('chart', {}).get('result', [{}])[0]
        
        if not chart:
            print("Erro: Dados não encontrados")
            return None
        
        # Extrair timestamps e preços
        timestamps = chart.get('timestamp', [])
        quotes = chart.get('indicators', {}).get('quote', [{}])[0]
        adj_close = chart.get('indicators', {}).get('adjclose', [{}])[0]
        
        # Preferir Adjusted Close
        if adj_close and 'adjclose' in adj_close:
            precos = adj_close.get('adjclose', [])
        else:
            precos = quotes.get('close', [])
        
        if not timestamps or not precos:
            print("Erro: Dados incompletos")
            return None
        
        # Criar DataFrame com incremento sequencial de mês
        datas_sequenciais = []
        ano_atual, mes_atual = ano, mes
        
        for i in range(len(timestamps)):
            if debug and i < 10:
                dt = datetime.fromtimestamp(timestamps[i])
                print(f"Timestamp {i}: {dt.strftime('%Y-%m-%d')} -> {ano_atual:04d}-{mes_atual:02d} -> Preço: {precos[i]}")
            
            datas_sequenciais.append(f"{ano_atual:04d}-{mes_atual:02d}")
            
            # Incrementar mês
            mes_atual += 1
            if mes_atual > 12:
                mes_atual = 1
                ano_atual += 1
        
        df = pd.DataFrame({
            'Data': datas_sequenciais,
            'Preco': precos
        })
        
        # Remover apenas valores None/NaN nos preços
        df = df[df['Preco'].notna()].reset_index(drop=True)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None
    except Exception as e:
        print(f"Erro: {e}")
        return None


def salvar_csv(df, ticker, formato_simples=True):
    """
    Salva DataFrame em CSV
    
    Args:
        df: DataFrame com dados
        ticker: Nome do ativo
        formato_simples: Se True, salva apenas data,preco
    """
    nome_arquivo = f"{ticker.replace('^', '').replace('-', '_')}_mensal.csv"
    
    if formato_simples:
        # Formato: YYYY-MM,preco
        with open(nome_arquivo, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(f"{row['Data']},{row['Preco']}\n")
    else:
        # Formato padrão CSV com cabeçalho
        df.to_csv(nome_arquivo, index=False, encoding='utf-8')
    
    print(f"✓ Arquivo salvo: {nome_arquivo}")
    return nome_arquivo


def main():
    print("=" * 60)
    print("DOWNLOADER DE DADOS HISTÓRICOS - YAHOO FINANCE")
    print("=" * 60)
    
    # Entrada de dados
    ticker = input("\nCódigo do ativo (ex: BTC-USD, ^GSPC, AAPL): ").strip()
    data_inicio = input("Data inicial YYYY-MM (ex: 2014-09): ").strip()
    
    # Modo debug
    debug_mode = input("\nMostrar debug dos primeiros registros? (S/N): ").strip().upper() == 'S'
    
    # Baixar dados
    df = baixar_dados_yahoo(ticker, data_inicio, debug=debug_mode)
    
    if df is not None and not df.empty:
        print(f"\n✓ {len(df)} registros obtidos com sucesso!")
        
        # Mostrar amostra
        print("\nPrimeiros registros:")
        print(df.head(10).to_string(index=False))
        
        print("\nÚltimos registros:")
        print(df.tail(5).to_string(index=False))
        
        # Salvar
        salvar_csv(df, ticker, formato_simples=True)
        
    else:
        print("\n✗ Não foi possível obter os dados.")


if __name__ == "__main__":
    # Uso direto (descomente para testar)
    # df = baixar_dados_yahoo('BTC-USD', '2014-09')
    # df = baixar_dados_yahoo('^BVSP', '1994-01')
    # if df is not None:
    #     salvar_csv(df, 'BTC-USD')
    
    # Uso interativo
    main()