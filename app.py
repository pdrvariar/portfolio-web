# app.py
from flask import Flask, render_template, request, jsonify, send_file
import os
import threading
from backtest import run_portfolio_simulation, get_available_assets_with_history, load_config, save_config_web, get_config_files, create_example_config
import uuid
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'configuration'
app.secret_key = 'secret_key'  # Necessário para sessões

# Dicionário para armazenar resultados de simulação
results_storage = {}

# Criar arquivos exemplo se necessário
create_example_config()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_assets')
def get_assets():
    assets = get_available_assets_with_history()
    return jsonify(assets)

@app.route('/get_configs')
def get_configs():
    configs = get_config_files()
    return jsonify(configs)

@app.route('/load_config', methods=['POST'])
def load_config_route():
    config_name = request.json['config_name']
    config_data = load_config(os.path.join('configuration', config_name))
    return jsonify(config_data)

@app.route('/save_config', methods=['POST'])
def save_config():
    data = request.json
    config_name = data['config_name']
    params = data['params']
    allocation = data['allocation']
    factors = data['factors']
    
    result = save_config_web(config_name, params, allocation, factors)
    return jsonify(result)



@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    config_name = request.json['config_name']
    execution_id = str(uuid.uuid4())  # ID único para esta execução
      
    
    # Rodar em thread separada
    def run_sim():
        try:
            # Executar simulação
            result = run_portfolio_simulation(config_name)
            
            # Armazenar resultados
            results_storage[execution_id] = {
                'status': 'completed',
                'result': result,
                'html_table': result.get('html_table', ''),  
                'timestamp': time.time()
            }
        except Exception as e:
            results_storage[execution_id] = {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    thread = threading.Thread(target=run_sim)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'execution_id': execution_id,
        'message': 'Simulação iniciada. Os resultados serão gerados em breve.'
    })

@app.route('/get_results_table', methods=['POST'])
def get_results_table():
    try:
        # Obter execution_id do corpo da requisição
        data = request.get_json()
        if not data or 'execution_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Campo execution_id não encontrado na requisição'
            }), 400
        
        execution_id = data['execution_id']
        result_data = results_storage.get(execution_id)
        
        if not result_data:
            return jsonify({
                'success': False,
                'error': 'Execução não encontrada'
            }), 404
        
        if result_data['status'] == 'error':
            return jsonify({
                'success': False,
                'error': result_data['error']
            }), 500
        
        if result_data['status'] == 'completed':
            if result_data['result']['success']:
                return jsonify({
                    'success': True,
                    'html_table': result_data['result']['html_table']
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result_data['result']['error']
                }), 500
        
        return jsonify({
            'success': False,
            'error': 'Simulação ainda em andamento'
        }), 202
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Erro ao processar requisição: {str(e)}'
        }), 500
    
@app.route('/show_graphs', methods=['POST'])
def show_graphs():
    execution_id = request.json['execution_id']
    result_data = results_storage.get(execution_id)
    
    if not result_data:
        return jsonify({
            'success': False,
            'error': 'Execução não encontrada'
        }), 404
    
    if result_data['status'] == 'error':
        return jsonify({
            'success': False,
            'error': result_data['error']
        }), 500
    
    if result_data['status'] == 'completed':
        if result_data['result']['success']:
            return result_data['result']['html_graph']
        else:
            return jsonify({
                'success': False,
                'error': result_data['result']['error']
            }), 500
    
    return jsonify({
        'success': False,
        'error': 'Simulação ainda em andamento'
    }), 202

@app.route('/download_result')
def download_result():
    config_name = request.args.get('config')
    output_currency = request.args.get('currency', 'BRL').lower()
    filename = f"{config_name}_{output_currency}.xlsx"
    return send_file(
        os.path.join('spreadsheet', filename),
        as_attachment=True
    )

@app.route('/get_results_details', methods=['POST'])
def get_results_details():
    data = request.get_json()
    execution_id = data.get('execution_id')
    
    if not execution_id:
        return jsonify({'success': False, 'error': 'execution_id is required'}), 400
        
    result_data = results_storage.get(execution_id)
    
    if not result_data:
        return jsonify({'success': False, 'error': 'Execução não encontrada'}), 404
        
    if result_data['status'] != 'completed':
        return jsonify({'success': False, 'error': 'Simulação ainda não concluída'}), 202
        
    return jsonify({
        'success': True,
        'html_table': result_data['result']['html_table']
    })

if __name__ == '__main__':
    app.run(debug=True)