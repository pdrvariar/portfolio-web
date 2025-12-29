import requests
import pandas as pd
import json
from datetime import datetime

def get_sp500_via_api():
    """
    Obtém dados do S&P 500 usando API alternativa do Yahoo Finance
    """
    # Parâmetros para dados mensais desde 1994
    params = {
        'period1': 1410912000,  # 1º de janeiro de 1994
        'period2': int(datetime.now().timestamp()),  # Data atual
        'interval': '1mo',  # Mensal
        'events': 'history',
        'includeAdjustedClose': 'true'
    }
    
    url = "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print("Consultando API do Yahoo Finance...")
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extrair dados do JSON
        chart_data = data.get('chart', {}).get('result', [{}])[0]
        
        if not chart_data:
            print("Erro: Dados não encontrados na resposta da API")
            return None
        
        # Extrair timestamps e preços
        timestamps = chart_data.get('timestamp', [])
        quotes = chart_data.get('indicators', {}).get('quote', [{}])[0]
        adj_close = chart_data.get('indicators', {}).get('adjclose', [{}])[0]
        
        if not timestamps or not quotes:
            print("Erro: Dados incompletos na resposta")
            return None
        
        # Usar Adjusted Close se disponível, caso contrário usar Close
        if adj_close and 'adjclose' in adj_close:
            prices = adj_close.get('adjclose', [])
        else:
            prices = quotes.get('close', [])
        
        # Criar DataFrame
        dates = [datetime.fromtimestamp(ts).strftime('%Y-%m') for ts in timestamps]
        
        df = pd.DataFrame({
            'YearMonth': dates,
            'Adjusted close price': prices
        })
        
        # Remover valores nulos
        df = df.dropna()
        
        # Remover duplicatas (mantendo último de cada mês)
        df = df.drop_duplicates(subset='YearMonth', keep='last')
        
        # Ordenar por data
        df = df.sort_values('YearMonth').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Erro ao consultar API: {e}")
        return None

# Programa principal usando API
def main_api():
    print("="*60)
    print("BAIXADOR DE DADOS DO S&P 500 - YAHOO FINANCE API")
    print("="*60)
    
    # Instalar requests se necessário
    try:
        import requests
    except ImportError:
        print("Instalando requests...")
        import subprocess
        subprocess.call(["pip", "install", "requests"])
        import requests
    
    # Obter dados
    print("\nObtendo dados via API...")
    sp500_data = get_sp500_via_api()
    
    if sp500_data is not None and not sp500_data.empty:
        print(f"\n✓ Dados obtidos com sucesso!")
        print(f"\nTotal de registros: {len(sp500_data)}")
        
        # Salvar em CSV
        with open("SP500_API_mensal.csv", "w", encoding="utf-8") as f:
            f.write("YearMonth,Adjusted close price\n")
            f.write("SP500,COTACAO:USD\n")
            for _, row in sp500_data.iterrows():
                f.write(f"{row['YearMonth']},{row['Adjusted close price']}\n")
        
        print(f"\n✓ Arquivo salvo como 'SP500_API_mensal.csv'")
        print(f"\nAmostra dos dados:")
        print(sp500_data.head(10))
        
    else:
        print("\n✗ Não foi possível obter os dados.")

# Escolha qual método usar
if __name__ == "__main__":
    print("Escolha o método de obtenção de dados:")
    print("1. Web Scraping (direto do site)")
    print("2. API do Yahoo Finance")
    
    escolha = input("\nDigite 1 ou 2: ").strip()
    
    if escolha == "1":
        main()
    elif escolha == "2":
        main_api()
    else:
        print("Opção inválida. Executando método da API...")
        main_api()