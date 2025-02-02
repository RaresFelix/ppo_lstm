import os
from glob import glob
from flask import Flask, render_template, session, request, jsonify, send_file
from src.web.server.game_manager import GameManager  # Use absolute import
import secrets
import logging
from functools import lru_cache

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
game_manager = GameManager()

def get_available_runs():
    runs_dir = os.path.join(app.static_folder, 'runs')
    runs = []
    for run_path in glob(os.path.join(runs_dir, '**/run_*'), recursive=True):
        run_id = os.path.basename(run_path)
        image_count = len(glob(os.path.join(run_path, 'images', 'env_*.png')))
        if image_count > 0:
            runs.append({
                'id': run_id,
                'frame_count': image_count
            })
    return sorted(runs, key=lambda x: x['id'], reverse=True)

@app.route('/get_runs', methods=['GET'])
def get_runs():
    runs = get_available_runs()
    return jsonify(runs)

@app.route('/')
def index():
    try:
        if 'session_id' not in session:
            session['session_id'] = secrets.token_hex(16)
            game_render = game_manager.create_session(session['session_id'])
        else:
            game_render = game_manager.get_render(session['session_id'])
        
        if game_render is None:
            raise ValueError("Game render is None")
            
        runs = get_available_runs()
        return render_template('index.html', 
                             game_render=game_render,
                             render_height=len(game_render),
                             render_width=len(game_render[0]),
                             runs=runs)  # Add runs to template context
    except Exception as e:
        logging.error(f"Error in index route: {str(e)}")
        return render_template('index.html', error="Failed to initialize game")

@app.route('/action', methods=['POST'])
def handle_action():
    try:
        if 'session_id' not in session:
            return jsonify({'error': 'No active session'}), 400
            
        action = request.json.get('action')
        if action is None:
            return jsonify({'error': 'No action provided'}), 400
            
        game_render = game_manager.perform_action(session['session_id'], action)
        if game_render is None:
            return jsonify({'error': 'Invalid session'}), 400
            
        return jsonify(game_render)
    except Exception as e:
        logging.error(f"Error handling action: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/toggle_view', methods=['POST'])
def toggle_view():
    try:
        if 'session_id' not in session:
            return jsonify({'error': 'No active session'}), 400
            
        agent_view = request.json.get('agent_view', False)
        game_render = game_manager.toggle_view(session['session_id'], agent_view)
        
        if game_render is None:
            return jsonify({'error': 'Invalid session'}), 400
            
        return jsonify(game_render)
    except Exception as e:
        logging.error(f"Error toggling view: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/runs/<task_type>')
def get_runs_for_task(task_type):
    try:
        runs_dir = os.path.join(app.static_folder, 'runs', task_type)
        if not os.path.exists(runs_dir):
            return jsonify([])

        runs = []
        for run_dir in sorted(glob(os.path.join(runs_dir, 'run_*')), reverse=True):
            run_id = os.path.basename(run_dir)
            image_count = len(glob(os.path.join(run_dir, 'images', 'env_*.png')))
            if image_count > 0:
                runs.append({
                    'id': run_id,
                    'frames': image_count
                })
        return jsonify(runs)
    except Exception as e:
        logging.error(f"Error getting runs for task {task_type}: {str(e)}")
        return jsonify([])

@app.route('/api/frame/<task_type>/<run_id>/<frame_type>/<frame>')
def get_frame(task_type, run_id, frame_type, frame):
    try:
        frame_path = os.path.join(
            app.static_folder,
            'runs',
            task_type,
            run_id,
            'images',
            f'{frame_type}_{frame}.png'
        )
        
        if not os.path.exists(frame_path):
            logging.error(f"File not found: {frame_path}")
            return '', 404
            
        return send_file(frame_path, mimetype='image/png')
    except Exception as e:
        logging.error(f"Error serving frame: {str(e)}")
        return '', 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
