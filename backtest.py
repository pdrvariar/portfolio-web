import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import configparser
import os
import traceback
from flask import render_template_string

# Mapeamento para exibição na tela
TIPO_MAPA_TELA = {
    'COTACAO': 'Cotação',
    'CAMBIO': 'Câmbio',
    'TAXA_MENSAL': 'Taxa Mensal',
    'TAXA_ANUAL': 'Taxa Anual'
}

def parse_second_line_from_csv(file_path):
    """Extrai Nome Amigável, Tipo de Histórico e Moeda da segunda linha do CSV"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return file_path.replace('.csv', ''), 'COTACAO', 'BRL'  # fallback

        second_line = lines[1].strip()
        if ':' in second_line and ',' in second_line:
            try:
                nome_amigavel, tipo_moeda = second_line.split(',')
                tipo, moeda = tipo_moeda.split(':')
                return nome_amigavel.strip(), tipo.strip().upper(), moeda.strip().upper()
            except Exception:
                pass
        return file_path.replace('.csv', ''), 'COTACAO', 'BRL'  # fallback

def create_example_config():
    """Cria um arquivo de configuração exemplo se necessário"""
    example_path = 'configuration/config_example.ini'
    if not os.path.exists('configuration'):
        os.makedirs('configuration')
    
    if not os.path.exists(example_path):
        config = configparser.ConfigParser()
        
        config['PARAMETERS'] = {
            'initial_capital': '100000',
            'start_year_month': '2010-01',
            'rebalance_freq': 'M',
            'output_currency': 'BRL'
        }
        
        config['ALLOCATION'] = {
            'gspc': '0.3',
            'bvsp': '0.4',
            'btc_usd': '0.3'
        }
        
        config['DATA_COLUMNS'] = {
            'usd_column': 'Adjusted Close'
        }
        
        with open(example_path, 'w') as configfile:
            config.write(configfile)

def load_config(config_file='configuration/config.ini'):
    config = configparser.ConfigParser()
    
    # Se o arquivo padrão não existir, usar um exemplo
    if not os.path.exists(config_file):
        config_file = 'configuration/config_example.ini'
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Arquivo de configuração padrão não encontrado: {config_file}")
    
    config.read(config_file)

    output_currency = config['PARAMETERS'].get('output_currency', 'BRL').upper()
    initial_capital = float(config['PARAMETERS']['initial_capital'])
    start_year_month = config['PARAMETERS']['start_year_month']
    
    # CORREÇÃO: Tratar caso onde end_year_month está vazio
    end_year_month = config['PARAMETERS'].get('end_year_month', '')
    if end_year_month.strip() == '':
        end_year_month = None
    
    # Mapeamento de códigos para valores numéricos
    freq_map = {
        'N': 0, 'M': 1, 'B': 2, 'Q': 3, 'D': 4, 'S': 6, 'A': 12
    }
    
    rebalance_code = config['PARAMETERS']['rebalance_freq'].upper()
    rebalance_freq = freq_map.get(rebalance_code, 1)  # Default para mensal se não encontrado

    # Load allocation and validate sum
    allocation = {}
    for k, v in config['ALLOCATION'].items():
        try:
            allocation[k.lower()] = float(v)
        except ValueError:
            allocation[k.lower()] = float(v.replace(',', '.'))
    
    total_allocation = sum(allocation.values())
    if not np.isclose(total_allocation, 1.0, atol=0.00000001):
        raise ValueError(f"A soma da alocação deve ser 100%! Atual: {total_allocation*100:.8f}%")
    
    # Load performance factors
    performance_factors = {}
    if 'PERFORMANCE_FACTOR' in config:
        for k, v in config['PERFORMANCE_FACTOR'].items():
            try:
                performance_factors[k.lower()] = float(v)
            except ValueError:
                performance_factors[k.lower()] = float(v.replace(',', '.'))    

    return {
        'initial_capital': initial_capital,
        'start_year_month': start_year_month,
        'rebalance_freq': rebalance_freq,  # Valor numérico
        'allocation': allocation,
        'output_currency': output_currency,
        'performance_factors': performance_factors,
        'end_year_month': end_year_month
    }

def apply_performance_factor(df, asset, factor):
    """Aplica fator de performance à série histórica do ativo"""
    if factor == 1.0:
        return df
    
    # Calcular retornos percentuais
    returns = df[asset].pct_change().fillna(0)
    
    # Aplicar fator de performance
    adjusted_returns = returns * factor
    
    # Recalcular preços ajustados
    adjusted_prices = [df[asset].iloc[0]]
    for i in range(1, len(df)):
        adjusted_prices.append(adjusted_prices[-1] * (1 + adjusted_returns.iloc[i]))
    
    df[asset] = adjusted_prices
    return df

def convert_rate_to_price(df, rate_column, initial_value=100):
    df = df.sort_values('Ano_Mes')
    rates = df[rate_column] / 100.0
    cumulative_factors = (1 + rates).cumprod()
    df[rate_column] = initial_value * cumulative_factors
    return df

def load_prices_from_individual_csvs(ativos, folder='historical_price', performance_factors=None):
    df_merged = None
    metadata = {}  # Guarda nome_amigavel, tipo e moeda para cada ativo

    for ativo in ativos:
        file_path = os.path.join(folder, f"{ativo}.csv")
        if os.path.exists(file_path):
            nome_amigavel, tipo, moeda = parse_second_line_from_csv(file_path)
            df = pd.read_csv(file_path, skiprows=2)
            df.columns = ['Ano_Mes', ativo]
            df['Ano_Mes'] = pd.to_datetime(df['Ano_Mes'], format='%Y-%m')
            df[ativo] = pd.to_numeric(df[ativo], errors='coerce')

            if tipo == 'TAXA_MENSAL':
                df = convert_rate_to_price(df, ativo, initial_value=100)
            elif tipo == 'TAXA_ANUAL':
                df[ativo] = ((1 + df[ativo] / 100.0) ** (1/12) - 1) * 100
                df = convert_rate_to_price(df, ativo, initial_value=100)

            factor = performance_factors.get(ativo, 1.0) if performance_factors else 1.0
            df = apply_performance_factor(df, ativo, factor)

            metadata[ativo] = {
                'nome_amigavel': nome_amigavel,
                'tipo': tipo,
                'moeda': moeda,
                'arquivo': os.path.basename(file_path)
            }

            df_merged = df if df_merged is None else pd.merge(df_merged, df, on='Ano_Mes', how='outer')

    df_merged = df_merged.sort_values('Ano_Mes')
    for col in df_merged.columns:
        if col != 'Ano_Mes':
            df_merged[col] = df_merged[col].ffill().bfill()

    return df_merged.dropna().reset_index(drop=True), metadata

def plot_portfolio_results(df_result, initial_capital, allocation, rebalance_freq, output_currency, currency_mapping, config_basename='config'):
    # Configurações iniciais
    currency_symbol = '$' if output_currency == 'USD' else 'R$'
    final_value = df_result['Total_Acumulado'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    num_years = len(df_result['Ano_Mes'].dt.year.unique())
    annual_return = (((final_value / initial_capital) ** (1 / num_years)) - 1) * 100 if num_years > 0 else 0
    
    # Cálculo de métricas
    df_result['Retorno_Mensal'] = df_result['Total_Acumulado'].pct_change()
    vol_mensal = df_result['Retorno_Mensal'].std() * 100
    acumulado = df_result['Total_Acumulado']
    max_dd = ((acumulado / acumulado.cummax()) - 1).min() * 100
    sharpe_ratio = df_result['Retorno_Mensal'].mean() / df_result['Retorno_Mensal'].std() if df_result['Retorno_Mensal'].std() != 0 else 0
    
    # Texto de resumo
    allocation_str = ', '.join([f"{k.upper()}: {v*100:.1f}%" for k, v in allocation.items()])
    resumo_texto = (
        f"<b>Configuração:</b> {config_basename.upper()} | "
        f"<b>Capital Inicial:</b> {currency_symbol}{initial_capital:,.2f}<br>"
        f"<b>Valor Final:</b> {currency_symbol}{final_value:,.2f} ({total_return:.2f}% de retorno)<br>"
        f"<b>Frequência Rebalanceamento:</b> {rebalance_freq} {'mês' if rebalance_freq == 1 else 'meses'}<br>"
        f"<b>Retorno Médio Anual:</b> {annual_return:.2f}% | "
        f"<b>Volatilidade Mensal:</b> {vol_mensal:.2f}% | "
        f"<b>Máx. Drawdown:</b> {max_dd:.2f}% | "
        f"<b>Sharpe:</b> {sharpe_ratio:.2f}<br>"
        f"<b>Alocação:</b> {allocation_str}"
    )

    # Cálculos para gráficos anuais
    df_result['Ano'] = df_result['Ano_Mes'].dt.year
    first_months = df_result.groupby('Ano')['Ano_Mes'].first()
    last_months = df_result.groupby('Ano')['Ano_Mes'].last()
    
    annual_data = []
    anos = sorted(df_result['Ano'].unique())
    
    for ano in anos:
        start_date = first_months[ano]
        if ano + 1 in last_months:
            end_date = last_months[ano] if ano == anos[-1] else first_months[ano + 1]
        else:
            end_date = last_months[ano]
        
        start_row = df_result[df_result['Ano_Mes'] == start_date].iloc[0]
        end_row = df_result[df_result['Ano_Mes'] == end_date].iloc[0]
        
        # Calcular retorno total
        start_value = start_row['Total_Acumulado']
        end_value = end_row['Total_Acumulado']
        annual_return_total = (end_value - start_value) / start_value * 100
        
        # Calcular retorno por ativo (valor)
        ativos_returns = {}
        for col in df_result.columns:
            if col.startswith('Val_'):
                asset = col[4:]
                start_asset = start_row[col]
                end_asset = end_row[col]
                ativos_returns[asset] = (end_asset - start_asset) / start_asset * 100
        
        # Calcular variação de cotação por ativo COM CONVERSÃO DE MOEDA
        cotacao_returns = {}
        for col in df_result.columns:
            if col.startswith('Cotacao_'):
                asset = col[8:]
                asset_currency = currency_mapping.get(asset.lower(), 'brl')['moeda'].lower()
                
                # Obter preços originais
                start_price_orig = start_row[col]
                end_price_orig = end_row[col]
                
                # Obter taxas de câmbio
                start_usd_brl = start_row['USD_BRL']
                end_usd_brl = end_row['USD_BRL']
                
                # Converter preços para moeda de saída se necessário
                if asset_currency != output_currency.lower():
                    if output_currency == 'BRL' and asset_currency == 'usd':
                        start_price = start_price_orig * start_usd_brl
                        end_price = end_price_orig * end_usd_brl
                    elif output_currency == 'USD' and asset_currency == 'brl':
                        start_price = start_price_orig / start_usd_brl
                        end_price = end_price_orig / end_usd_brl
                    else:
                        start_price = start_price_orig
                        end_price = end_price_orig
                else:
                    start_price = start_price_orig
                    end_price = end_price_orig
                
                # Calcular variação
                if start_price > 0:
                    cotacao_returns[asset] = (end_price - start_price) / start_price * 100
                else:
                    cotacao_returns[asset] = 0
        
        annual_data.append({
            'Ano': ano,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Total_Return': annual_return_total,
            'Ativos_Returns': ativos_returns,
            'Cotacao_Returns': cotacao_returns
        })
    
    # Preparar dados para gráficos
    anos_list = [d['Ano'] for d in annual_data]
    returns_total = [d['Total_Return'] for d in annual_data]
    
    # Criar figura com subplots
    fig = make_subplots(
        rows=5, cols=2,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"colspan": 2, "type": "bar"}, None],
            [{"colspan": 2, "type": "bar"}, None],
            [{"colspan": 2, "type": "bar"}, None],
            [{"colspan": 2, "type": "table"}, None]  # Nova linha para a tabela
        ],
        subplot_titles=(
            "Evolução do Valor Total",
            "Retorno Acumulado",
            "Rentabilidade Anual Total",
            "Rentabilidade Anual por Ativo",
            "Variação Anual de Cotação por Ativo",
            ""
        ),
        vertical_spacing=0.05
    )

    # Gráfico 1: Valor total
    fig.add_trace(
        go.Scatter(
            x=df_result['Ano_Mes'],
            y=df_result['Total_Acumulado'],
            mode='lines',
            name='Valor Total',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Gráfico 2: Retorno acumulado
    retorno_acumulado = df_result['Total_Acumulado'] / initial_capital
    fig.add_trace(
        go.Scatter(
            x=df_result['Ano_Mes'],
            y=retorno_acumulado,
            mode='lines',
            name='Retorno Acumulado',
            line=dict(color='orange')
        ),
        row=1, col=2
    )

    # Gráfico 3: Rentabilidade Anual Total
    fig.add_trace(
        go.Bar(
            x=anos_list,
            y=returns_total,
            name='Retorno Anual',
            marker_color='green',
            text=[f"{v:.1f}%" for v in returns_total],
            textposition='auto'
        ),
        row=2, col=1
    )

    # Gráfico 4: Rentabilidade Anual por Ativo
    ativos_plot = [col[4:] for col in df_result.columns if col.startswith('Val_')]
    for idx, ativo in enumerate(ativos_plot):
        asset_returns = [d['Ativos_Returns'].get(ativo, 0) for d in annual_data]
        fig.add_trace(
            go.Bar(
                x=anos_list,
                y=asset_returns,
                name=ativo.upper(),
                text=[f"{v:.1f}%" for v in asset_returns],
                textposition='auto'
            ),
            row=3, col=1
        )

    # Gráfico 5: Variação Anual de Cotação por Ativo
    ativos_plot_cot = [col[8:] for col in df_result.columns if col.startswith('Cotacao_')]  # LINHA CORRIGIDA

    for idx, ativo in enumerate(ativos_plot_cot):
        price_returns = [d['Cotacao_Returns'].get(ativo, 0) for d in annual_data]
        fig.add_trace(
            go.Bar(
                x=anos_list,
                y=price_returns,
                name=ativo.upper(),
                text=[f"{v:.1f}%" for v in price_returns],
                textposition='auto'
            ),
            row=4, col=1
        )

    # Adicionar tabela de métricas como um subplot
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Métrica</b>', '<b>Valor</b>'],
                fill_color='#1f77b4',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[
                    [
                        'Configuração', 
                        'Frequência Rebalanceamento',
                        'Capital Inicial',
                        'Valor Final',
                        'Retorno Total',
                        'Retorno Médio Anual',
                        'Volatilidade Mensal',
                        'Máximo Drawdown',
                        'Índice de Sharpe',
                        'Alocação de Ativos'
                    ],
                    [
                        config_basename.upper(),
                        f"{rebalance_freq} {'mês' if rebalance_freq == 1 else 'meses'}",
                        f"{currency_symbol}{initial_capital:,.2f}",
                        f"{currency_symbol}{final_value:,.2f}",
                        f"{total_return:.2f}%",
                        f"{annual_return:.2f}%",
                        f"{vol_mensal:.2f}%",
                        f"{max_dd:.2f}%",
                        f"{sharpe_ratio:.2f}",
                        ", ".join([f"{k.upper()}: {v*100:.1f}%" for k, v in allocation.items()])
                    ]
                ],
                fill_color='#f9f9f9',
                align='left',
                font=dict(size=12)
            )
        ),
        row=5, col=1
    )

    # Layout e formatação
    fig.update_layout(
        annotations=[dict(
            text=resumo_texto,
            xref="paper", yref="paper",
            x=0.5, y=1.15,
            xanchor='center',
            yanchor='top',
            showarrow=False,
            align="left",
            font=dict(size=14))
        ],
        height=2200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        barmode='group'
    )
    
    # Configurações de eixo
    fig.update_yaxes(title_text=f"Valor ({currency_symbol})", row=1, col=1)
    fig.update_yaxes(title_text="Multiplicador", row=1, col=2)
    fig.update_yaxes(title_text="Retorno (%)", row=2, col=1)
    fig.update_yaxes(title_text="Retorno (%)", row=3, col=1)
    fig.update_yaxes(title_text="Variação (%)", row=4, col=1)
    
    # Formatar eixos como percentuais
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)
    
    # Salvar como HTML
    #html_file = os.path.join('spreadsheet', f'{config_basename}_{output_currency.lower()}.html')
    #fig.write_html(html_file)

    # Substituir a parte de salvar HTML por retornar o conteúdo
    html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
    return html_content    
    
    return html_file
    
def run_portfolio_simulation(config_basename='config'):
    try:
        config_data = load_config()
        initial_capital = config_data['initial_capital']
        start_year_month = config_data['start_year_month']
        rebalance_freq = config_data['rebalance_freq']
        allocation = config_data['allocation']
        output_currency = config_data['output_currency']
        performance_factors = config_data['performance_factors']
        end_year_month = config_data['end_year_month']
        
        out_folder = 'spreadsheet'
        os.makedirs(out_folder, exist_ok=True)

        ativos = [k.lower() for k in allocation.keys()]

        prices, metadata = load_prices_from_individual_csvs(
            ativos, 
            folder='historical_price',
            performance_factors=performance_factors
        )

        # Load USD/BRL
        usd_brl_path = os.path.join('historical_price', 'usd_brl.csv')
        usd_brl = pd.read_csv(usd_brl_path, skiprows=1)
        usd_brl.columns = ['Ano_Mes', 'USD_BRL']
        usd_brl['Ano_Mes'] = pd.to_datetime(usd_brl['Ano_Mes'], format='%Y-%m')
        usd_brl['USD_BRL'] = pd.to_numeric(usd_brl['USD_BRL'], errors='coerce')
        
        # Preencher valores ausentes e remover NaNs
        usd_brl['USD_BRL'] = usd_brl['USD_BRL'].ffill().bfill()
        usd_brl = usd_brl.dropna()

        # Find common dates
        common_dates = set(prices['Ano_Mes']) & set(usd_brl['Ano_Mes'])

        if not common_dates:
            raise ValueError("Não há nenhuma data comum entre todos os ativos e USD/BRL!")

        earliest_common_date = min(common_dates)
        start_date_ini = pd.to_datetime(start_year_month)
        start_date = max(start_date_ini, earliest_common_date)
        end_date_auto = prices['Ano_Mes'].max()
        if end_year_month:
            end_date_param = pd.to_datetime(end_year_month)
            end_date = min(end_date_param, end_date_auto)
        else:
            end_date = end_date_auto
        
        # Filtrar datas válidas
        dates = prices[
            (prices['Ano_Mes'] >= start_date) & 
            (prices['Ano_Mes'] <= end_date) &
            (prices['Ano_Mes'].isin(common_dates))
        ]['Ano_Mes'].unique()

        if len(dates) == 0:
            raise ValueError("Não há datas válidas para simulação após o filtro!")

        portfolio_qty = {a: 0.0 for a in ativos}
        results = []

        first_prices = prices[prices['Ano_Mes'] == start_date].iloc[0]
        usd_first = usd_brl[usd_brl['Ano_Mes'] == start_date]['USD_BRL'].values[0]
        total_alloc = initial_capital

        for ativo, pct in allocation.items():
            price = first_prices[ativo]
            ativo_currency = metadata[ativo]['moeda'].lower()
            if ativo_currency != output_currency.lower():
                if output_currency == 'BRL' and ativo_currency == 'usd':
                    price *= usd_first
                elif output_currency == 'USD' and ativo_currency == 'brl':
                    price /= usd_first
            portfolio_qty[ativo] = (pct * total_alloc) / price

        for idx, date in enumerate(sorted(dates)):
            prices_row = prices[prices['Ano_Mes'] == date].iloc[0]
            usd = usd_brl[usd_brl['Ano_Mes'] == date]['USD_BRL'].values[0]

            total_value = 0.0
            ativos_values = {}
            rebalance = 'Não'
            target_values = {}
            deltas = {}  # Armazena as variações de quantidade por ativo

            # Calcular valor atual de cada ativo
            for ativo in ativos:
                price = prices_row[ativo]
                ativo_currency = metadata[ativo]['moeda'].lower()
                price_base = price
                if ativo_currency != output_currency.lower():
                    if output_currency == 'BRL' and ativo_currency == 'usd':
                        price_base *= usd
                    elif output_currency == 'USD' and ativo_currency == 'brl':
                        price_base /= usd
                ativos_values[ativo] = portfolio_qty[ativo] * price_base
                total_value += ativos_values[ativo]

            # Verificar se é momento de rebalancear
            if rebalance_freq > 0 and idx % rebalance_freq == 0:
                rebalance = 'Sim'
                # Calcular valores alvo
                target_values = {a: allocation[a.lower()] * total_value for a in ativos}
                
                # Ajustar quantidades
                for ativo in ativos:
                    price = prices_row[ativo]
                    ativo_currency = metadata[ativo]['moeda'].lower()
                    price_base = price
                    if ativo_currency != output_currency.lower():
                        if output_currency == 'BRL' and ativo_currency == 'usd':
                            price_base *= usd
                        elif output_currency == 'USD' and ativo_currency == 'brl':
                            price_base /= usd
                    
                    current_value = ativos_values[ativo]
                    diff = target_values[ativo] - current_value
                    delta_qty = diff / price_base
                    
                    # Armazenar delta para registro
                    deltas[ativo] = delta_qty
                    
                    # Ajustar quantidade
                    portfolio_qty[ativo] += delta_qty

            # Construir linha de resultados com novas colunas
            row = {
                'Ano_Mes': date,
                'Total_Acumulado': total_value,
                'Rebalanceado': rebalance,
                'USD_BRL': usd,
                'Target_Total': total_value if rebalance == 'Sim' else None
            }

            # Adicionar dados por ativo
            for ativo in ativos:
                row[f'Cotacao_{ativo.upper()}'] = prices_row[ativo]
                row[f'Qtd_{ativo.upper()}'] = portfolio_qty[ativo]
                row[f'Val_{ativo.upper()}'] = ativos_values[ativo]
                
                # Adicionar detalhes de rebalanceamento quando aplicável
                if rebalance == 'Sim':
                    row[f'Target_{ativo.upper()}'] = target_values[ativo]
                    row[f'Pre_Rebalance_{ativo.upper()}'] = ativos_values[ativo]
                    row[f'Delta_{ativo.upper()}'] = deltas[ativo]
                else:
                    row[f'Target_{ativo.upper()}'] = None
                    row[f'Pre_Rebalance_{ativo.upper()}'] = None
                    row[f'Delta_{ativo.upper()}'] = 0.0

            results.append(row)

        df_result = pd.DataFrame(results)
        
        # Gerar HTML com tabela paginada
        html_table = generate_paginated_table(df_result, 25)
        
        # Gerar HTML com gráficos
        html_graph = plot_portfolio_results(df_result, initial_capital, allocation, 
                                           rebalance_freq, output_currency, 
                                           metadata, config_basename)
        
        return {
            'success': True,
            'html_table': html_table,  # Manter esta linha
            'html_graph': html_graph,
            'raw_data': df_result.to_dict()  # Adicionar dados brutos se necessário
        }
                
    except ValueError as e:
        return {
            'success': False,
            'error': f"ERRO DE CONFIGURAÇÃO: {str(e)}"
        }
    except Exception as e:
        traceback.print_exc()
        return {
            'success': False,
            'error': f"ERRO INESPERADO: {str(e)}"
        }
    

def generate_paginated_table(df, page_size=25):
    df = df.fillna('')
    table_html = df.to_html(
        classes='table table-striped table-bordered w-100',
        index=False,
        border=0,
        table_id="datatable"  # <-- importante para ativar DataTables
    )
    return f"""
    <div class="table-responsive" style="overflow-x: auto;">
        {table_html}
    </div>
    """

def get_available_assets_with_history():
    assets = {}
    folder = 'historical_price'
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                asset_name = file.replace('.csv', '')
                file_path = os.path.join(folder, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # Pega nome amigável, tipo e moeda da segunda linha
                    nome_amigavel, tipo_raw, moeda = parse_second_line_from_csv(file_path)
                    tipo_historico = TIPO_MAPA_TELA.get(tipo_raw.upper(), tipo_raw.title())

                    # Coleta datas do histórico (linhas de dados)
                    data_linhas = [l for l in lines[2:] if l.strip() and not l.startswith(',')]
                    primeira = data_linhas[0].split(',')[0].strip() if data_linhas else 'N/D'
                    ultima = data_linhas[-1].split(',')[0].strip() if data_linhas else 'N/D'

                    assets[asset_name] = {
                        'nome_amigavel': nome_amigavel,
                        'arquivo': file,
                        'start': primeira,
                        'end': ultima,
                        'tipo': tipo_historico,
                        'moeda': moeda.upper()
                    }
                except Exception as e:
                    assets[asset_name] = {
                        'nome_amigavel': asset_name,
                        'arquivo': file,
                        'start': 'Erro',
                        'end': 'Erro',
                        'tipo': 'Erro',
                        'moeda': 'BRL'
                    }
    return assets

def get_config_files():
    """Lista todos os arquivos de configuração existentes"""
    configs = []
    folder = 'configuration'
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith('.ini'):
                configs.append(file)
    return sorted(configs)

def save_config_web(config_name, params, allocation, factors):
    try:
        config = configparser.ConfigParser()
        
        config['PARAMETERS'] = {
            'initial_capital': str(params['initial_capital']),
            'start_year_month': params['start_year_month'],
            'rebalance_freq': params['rebalance_freq'],
            'output_currency': params['output_currency']
        }
        
        if params['end_year_month']:
            config['PARAMETERS']['end_year_month'] = params['end_year_month']
        
        config['ALLOCATION'] = allocation
        
        # Adicionar fatores de performance
        config['PERFORMANCE_FACTOR'] = factors
        
        # Salvar arquivos
        os.makedirs('configuration', exist_ok=True)
        
        personal_config_file = os.path.join('configuration', f"{config_name}.ini")
        with open(personal_config_file, 'w') as configfile:
            config.write(configfile)
        
        # Salvar também como config.ini (sobrescreve)
        default_config_file = os.path.join('configuration', 'config.ini')
        with open(default_config_file, 'w') as configfile:
            config.write(configfile)
        
        return {'success': True, 'message': f"Configuração salva em {personal_config_file} e também em {default_config_file}!"}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}