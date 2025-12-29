import requests
import pandas as pd
from datetime import datetime

def baixar_selic_mensal(data_inicio, data_fim=None):
    """
    Baixa SELIC e calcula a taxa mensal efetiva
    
    Args:
        data_inicio: Data inicial no formato 'YYYY-MM' (ex: '1994-01')
        data_fim: Data final no formato 'YYYY-MM' (opcional, padrão: hoje)
    
    Returns:
        DataFrame com colunas 'Data' e 'Selic' (taxa mensal % a.m.)
    """
    # Converter datas
    try:
        ano_ini, mes_ini = map(int, data_inicio.split('-'))
        data_ini_str = f"01/{mes_ini:02d}/{ano_ini}"
        
        if data_fim:
            ano_fim, mes_fim = map(int, data_fim.split('-'))
            data_fim_str = f"31/{mes_fim:02d}/{ano_fim}"
        else:
            hoje = datetime.now()
            data_fim_str = f"31/{hoje.month:02d}/{hoje.year}"
    except ValueError:
        print("Erro: Data inválida. Use o formato YYYY-MM")
        return None
    
    # Série 11: SELIC diária (% a.a.)
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
    params = {
        'formato': 'json',
        'dataInicial': data_ini_str,
        'dataFinal': data_fim_str
    }
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0'
    }
    
    print(f"Baixando SELIC diária de {data_inicio} até {data_fim or 'hoje'}...")
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        dados = response.json()
        
        if not dados:
            print("Erro: Nenhum dado retornado")
            return None
        
        # Criar DataFrame
        df = pd.DataFrame(dados)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = df['valor'].astype(float)
        
        # Agrupar por mês e calcular taxa mensal
        df['YearMonth'] = df['data'].dt.to_period('M')
        
        # Calcular taxa mensal composta: (1 + taxa_diaria/100)^dias - 1
        resultado = []
        for periodo, grupo in df.groupby('YearMonth'):
            # Taxa diária anualizada para taxa diária efetiva
            # (1 + taxa_anual/100)^(1/252) - 1
            taxas_diarias = [(pow(1 + taxa/100, 1/252) - 1) for taxa in grupo['valor']]
            
            # Acumular no mês: (1 + r1) * (1 + r2) * ... - 1
            fator_acumulado = 1.0
            for taxa_dia in taxas_diarias:
                fator_acumulado *= (1 + taxa_dia)
            
            taxa_mensal = (fator_acumulado - 1) * 100
            
            resultado.append({
                'Data': str(periodo),
                'Selic': taxa_mensal
            })
        
        df_mensal = pd.DataFrame(resultado)
        
        return df_mensal
        
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None
    except Exception as e:
        print(f"Erro: {e}")
        return None


def salvar_csv(df, nome_arquivo='SELIC_mensal.csv'):
    """
    Salva DataFrame em CSV no formato simples
    """
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['Data']},{row['Selic']}\n")
    
    print(f"✓ Arquivo salvo: {nome_arquivo}")
    return nome_arquivo


def main():
    print("=" * 60)
    print("DOWNLOADER SELIC MENSAL - BANCO CENTRAL DO BRASIL")
    print("=" * 60)
    print("\nSérie 11: SELIC diária (% a.a.) - agregada mensalmente")
    print("Calcula a taxa mensal efetiva")
    print("Dados disponíveis desde: 1986")
    
    # Entrada de dados
    data_inicio = input("\nData inicial YYYY-MM (ex: 1994-01): ").strip()
    data_fim = input("Data final YYYY-MM (Enter = hoje): ").strip()
    
    if not data_fim:
        data_fim = None
    
    # Baixar dados
    df = baixar_selic_mensal(data_inicio, data_fim)
    
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
    # df = baixar_selic_mensal('1994-01')
    # if df is not None:
    #     salvar_csv(df)
    
    # Uso interativo
    main()