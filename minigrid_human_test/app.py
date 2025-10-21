# app.py
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from PIL import Image
import io
import base64
import json
import threading
from datetime import datetime
from minigrid_human_test.envs import *
from minigrid_human_test.wrappers import RGBImgObsWrapper

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gamedata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sessions = db.relationship('GameSession', backref='user', lazy=True)

class GameSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    task_index = db.Column(db.Integer)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    total_reward = db.Column(db.Float)
    actions = db.relationship('ActionLog', backref='session', lazy=True)

class ActionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('game_session.id'))
    action = db.Column(db.Integer)
    reward = db.Column(db.Float)
    state = db.Column(db.Text)  # Store as JSON
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Global environment state with user isolation
lock = threading.Lock()
environments = {}  # {user_id: env}
current_sessions = {}  # {user_id: GameSession}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def create_env(task_index):
    tasks = [
        EmptyEnv(),
        CrossingEnv(size=7, max_steps=1000),
        DoorKeyEnv(),
        FourRoomsEnv(),
        LockedRoomEnv(),
        PlaygroundEnv()
    ]
    return RGBImgObsWrapper(tasks[task_index])

def img_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        
        if not user:
            # Create new user
            user = User(email=email)
            db.session.add(user)
            db.session.commit()
        
        login_user(user)
        return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/init', methods=['POST'])
@login_required
def initialize():
    user_id = current_user.id
    with lock:
        # Clean up any existing session
        if user_id in environments:
            environments[user_id].close()
            del environments[user_id]
        
        task_index = int(request.form.get('task', 1))
        environments[user_id] = create_env(task_index)
        environments[user_id].reset()
        
        # Create new game session
        session = GameSession(
            user_id=user_id,
            task_index=task_index,
            total_reward=0.0
        )
        db.session.add(session)
        db.session.commit()

        # Store session ID instead of the instance
        current_sessions[user_id] = session.id
        
        env = environments.get(user_id)
        obs = env.reset()[0]["image"]
        obs = np.repeat(obs, 4, axis=0)
        obs = np.repeat(obs, 4, axis=1)
        img = Image.fromarray(obs.transpose(1, 0, 2))
        return jsonify(image=img_to_base64(img))

@app.route('/step', methods=['POST'])
@login_required
def step():
    user_id = current_user.id
    with lock:
        if user_id not in environments:
            return jsonify(error="Environment not initialized"), 400
            
        action = int(request.form.get('action', -1))
        if action == -1:
            return jsonify(error="Invalid action"), 400
            
        env = environments[user_id]
        obs, reward, done, _, _ = env.step(action)
        
        # Log action
        session_id = current_sessions.get(user_id)
        if not session_id:
            return jsonify(error="Session not found"), 400
            
        session = GameSession.query.get(session_id)
        session.total_reward += reward
        db.session.add(ActionLog(
            session_id=session.id,
            action=action,
            reward=reward,
            state=json.dumps(obs["image"].tolist())  # Store state as JSON
        ))
        db.session.commit()

        obs = np.repeat(obs["image"], 4, axis=0)
        obs = np.repeat(obs, 4, axis=1)
        
        img = Image.fromarray(obs.transpose(1, 0, 2))
        return jsonify(
            image=img_to_base64(img),
            reward=reward,
            done=done
        )

@app.route('/reset', methods=['POST'])
@login_required
def reset_env():
    user_id = current_user.id
    with lock:
        if user_id in environments:
            environments[user_id].reset()
            
            # Close current session
            if user_id in current_sessions:
                session = current_sessions[user_id]
                session.end_time = datetime.utcnow()
                db.session.commit()
                del current_sessions[user_id]
            
            env = environments.get(user_id)
            obs = env.reset()[0]
            obs = np.repeat(obs, 4, axis=0)
            obs = np.repeat(obs, 4, axis=1)
            img = Image.fromarray(obs["image"].transpose(1, 0, 2))
            return jsonify(image=img_to_base64(img))
        return jsonify(error="Environment not initialized"), 400

@app.route('/logout')
@login_required
def logout():
    user_id = current_user.id
    with lock:
        # Clean up environment
        if user_id in environments:
            environments[user_id].close()
            del environments[user_id]
        
        # Finalize session
        session_id = current_sessions.get(user_id)
        if session_id:
            session = GameSession.query.get(session_id)
            if session:
                session.end_time = datetime.utcnow()
                db.session.commit()
            del current_sessions[user_id]
    return redirect(url_for('login'))

if __name__ == '__main__':

    with app.app_context():
        db.create_all()
    app.run(threaded=True, port=5000)