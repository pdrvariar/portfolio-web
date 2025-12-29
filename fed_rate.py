import requests
import pandas as pd
from datetime import datetime

def baixar_fed_rate_mensal(data_inicio, data_fim=None, api_key='demo'):
    """
    Baixa a taxa Federal Funds Rate e calcula a taxa mensal efetiva
    
    Args:
        data_inicio: Data inicial no formato 'YYYY-MM' (ex: '1954-07')
        data_fim: Data final no formato 'YYYY-MM' (opcional, padrão: hoje)
        api_key: Chave da API FRED (use 'demo' para teste limitado)
    
    Returns:
        DataFrame com colunas 'Data' e 'FedRate' (taxa mensal % a.m.)
    """
    # Converter datas
    try:
        ano_ini, mes_ini = map(int, data_inicio.split('-'))
        data_ini_str = f"{ano_ini}-{mes_ini:02d}-01"
        
        hoje = datetime.now()
        
        if data_fim:
            ano_fim, mes_fim = map(int, data_fim.split('-'))
            
            # Validar se data final não é futura
            data_fim_obj = datetime(ano_fim, mes_fim, 1)
            if data_fim_obj > hoje:
                print(f"Aviso: Data final {data_fim} é futura. Usando data atual: {hoje.year}-{hoje.month:02d}")
                ano_fim, mes_fim = hoje.year, hoje.month
            
            data_fim_str = f"{ano_fim}-{mes_fim:02d}-28"
        else:
            data_fim_str = f"{hoje.year}-{hoje.month:02d}-{hoje.day:02d}"
    except ValueError:
        print("Erro: Data inválida. Use o formato YYYY-MM")
        return None
    
    # FRED API - Série DFF: Federal Funds Effective Rate (diária, % a.a.)
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': 'DFF',
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': data_ini_str,
        'observation_end': data_fim_str,
        'frequency': 'd'  # diária
    }
    
    print(f"Baixando Federal Funds Rate de {data_inicio} até {data_fim or 'hoje'}...")
    print(f"Série: DFF (Federal Funds Effective Rate - diária)")
    
    if api_key == 'demo':
        print("\n⚠️  Usando API key 'demo' (limitada)")
        print("Para uso intensivo, obtenha sua chave gratuita em:")
        print("https://fred.stlouisfed.org/docs/api/api_key.html\n")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'observations' not in data:
            print(f"Erro: {data.get('error_message', 'Dados não encontrados')}")
            return None
        
        observacoes = data['observations']
        
        if not observacoes:
            print("Erro: Nenhum dado retornado")
            return None
        
        # Criar DataFrame
        df = pd.DataFrame(observacoes)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filtrar valores válidos (às vezes vem '.')
        df = df[df['value'] != '.']
        df['value'] = df['value'].astype(float)
        
        # Agrupar por mês e calcular taxa mensal
        df['YearMonth'] = df['date'].dt.to_period('M')
        
        # Calcular taxa mensal composta
        resultado = []
        for periodo, grupo in df.groupby('YearMonth'):
            # Taxa diária anualizada para taxa diária efetiva
            # (1 + taxa_anual/100)^(1/365) - 1
            taxas_diarias = [(pow(1 + taxa/100, 1/365) - 1) for taxa in grupo['value']]
            
            # Acumular no mês
            fator_acumulado = 1.0
            for taxa_dia in taxas_diarias:
                fator_acumulado *= (1 + taxa_dia)
            
            taxa_mensal = (fator_acumulado - 1) * 100
            
            resultado.append({
                'Data': str(periodo),
                'FedRate': taxa_mensal
            })
        
        df_mensal = pd.DataFrame(resultado)
        
        return df_mensal
        
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None
    except Exception as e:
        print(f"Erro: {e}")
        return None


def salvar_csv(df, nome_arquivo='FED_rate_mensal.csv'):
    """
    Salva DataFrame em CSV no formato simples
    """
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['Data']},{row['FedRate']}\n")
    
    print(f"✓ Arquivo salvo: {nome_arquivo}")
    return nome_arquivo


def main():
    print("=" * 60)
    print("DOWNLOADER TAXA FED MENSAL - FRED API")
    print("=" * 60)
    print("\nSérie DFF: Federal Funds Effective Rate (% a.a.)")
    print("Calcula a taxa mensal efetiva")
    print("Dados disponíveis desde: 1954")
    
    # API Key
    print("\n" + "=" * 60)
    print("API KEY (opcional)")
    print("=" * 60)
    print("Para uso limitado, pode usar 'demo'")
    print("Para uso ilimitado, obtenha chave grátis em:")
    print("https://fred.stlouisfed.org/docs/api/api_key.html")
    
    api_key = input("\nAPI Key (Enter = demo): ").strip()
    if not api_key:
        api_key = 'demo'
    
    # Entrada de dados
    data_inicio = input("\nData inicial YYYY-MM (ex: 1994-01): ").strip()
    data_fim = input("Data final YYYY-MM (Enter = hoje): ").strip()
    
    if not data_fim:
        data_fim = None
    
    # Baixar dados
    df = baixar_fed_rate_mensal(data_inicio, data_fim, api_key)
    
    if df is not None and not df.empty:
        print(f"\n✓ {len(df)} registros obtidos com sucesso!")
        
        # Mostrar amostra
        print("\nPrimeiros registros (taxa mensal %):")
        print(df.head(10).to_string(index=False))
        
        print("\nÚltimos registros:")
        print(df.tail(5).to_string(index=False))
        
        # Salvar
        salvar_csv(df)
        
    else:
        print("\n✗ Não foi possível obter os dados.")


if __name__ == "__main__":
    # Uso direto (descomente para testar)
    # df = baixar_fed_rate_mensal('1994-01', api_key='sua_chave_aqui')
    # if df is not None:
    #     salvar_csv(df)
    
    # Uso interativo
    main()