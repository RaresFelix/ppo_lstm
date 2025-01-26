from flask import Flask, render_template, session, request, jsonify
from src.web.server.game_manager import GameManager  # Use absolute import
import secrets
import logging
from functools import lru_cache

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
game_manager = GameManager()

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
            
        return render_template('index.html', 
                             game_render=game_render,
                             render_height=len(game_render),
                             render_width=len(game_render[0]))
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

if __name__ == '__main__':
    app.run(debug=True)
