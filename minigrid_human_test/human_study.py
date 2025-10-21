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
import random

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gamedata_human_study.db'
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
    progress = db.relationship('UserProgress', backref='user', lazy=True)
    sessions = db.relationship('GameSession', backref='user', lazy=True)

class UserProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    environment = db.Column(db.String(50))
    trials_completed = db.Column(db.Integer, default=0)
    completed = db.Column(db.Boolean, default=False)

class GameSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    environment = db.Column(db.String(50))
    trial_number = db.Column(db.Integer)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    total_reward = db.Column(db.Float)
    actions = db.relationship('ActionLog', backref='session', lazy=True)

class ActionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('game_session.id'))
    action = db.Column(db.Integer)
    reward = db.Column(db.Float)
    state = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Experiment Configuration
ENV_SEQUENCE = ENV_SEQUENCE = [
    ('empty', lambda: EmptyEnv(max_steps=20, agent_start_pos=(random.randint(1, 3), random.randint(1, 3)))),  # Explicitly call constructor
    ('crossing', lambda: CrossingEnv(max_steps=20)),
    ('doorkey', lambda: DoorKeyEnv(max_steps=30)),
    ('fourrooms', lambda: FourRoomsEnv(max_steps=50)),
    ('lockedroom', lambda: LockedRoomEnv(max_steps=100)),
    ('playground', lambda: PlaygroundEnv(max_steps=100))
]

TRIALS_PER_ENV = 5

# Global State Management
lock = threading.Lock()
environments = {}
current_sessions = {}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def get_current_progress(user):
    progress = {}
    for env_name, _ in ENV_SEQUENCE:
        prog = UserProgress.query.filter_by(
            user_id=user.id,
            environment=env_name
        ).first()
        if not prog:
            prog = UserProgress(
                user_id=user.id,
                environment=env_name,
                trials_completed=0,
                completed=False
            )
            db.session.add(prog)
            db.session.commit()
        progress[env_name] = prog
    return progress

def get_current_environment(user):
    progress = get_current_progress(user)
    for env_name, _ in ENV_SEQUENCE:
        if not progress[env_name].completed:
            return env_name, progress[env_name].trials_completed + 1
    return None, 0

def create_env(env_name):
    for name, creator in ENV_SEQUENCE:
        if name == env_name:
            return RGBImgObsWrapper(creator())
    raise ValueError(f"Unknown environment: {env_name}")

def img_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        
        if not user:
            user = User(email=email)
            db.session.add(user)
            db.session.commit()
        
        login_user(user)
        return redirect(url_for('dashboard'))
    
    return render_template('login_human_study.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index_human_study.html')

@app.route('/init', methods=['POST'])
@login_required
def initialize():
    user_id = current_user.id
    with lock:
        env_name, trial_number = get_current_environment(current_user)
        if not env_name:
            return jsonify(error="Experiment complete"), 400

        if user_id in environments:
            environments[user_id].close()
            del environments[user_id]

        environments[user_id] = create_env(env_name)
        environments[user_id].reset()
        

        session = GameSession(
            user_id=user_id,
            environment=env_name,
            trial_number=trial_number,
            total_reward=0.0
        )
        db.session.add(session)
        db.session.commit()
        current_sessions[user_id] = session.id

        env = environments.get(user_id)
        obs = env.reset()[0]["image"]
        obs = np.repeat(obs, 4, axis=0)
        obs = np.repeat(obs, 4, axis=1)
        img = Image.fromarray(obs.transpose(1, 0, 2))
        return jsonify({
            "image": img_to_base64(img),
            "env_name": env_name.capitalize(),
            "trial_number": trial_number,
            "total_trials": TRIALS_PER_ENV,
            "total_steps": env.max_steps,
        })

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
        obs, reward, done, truncated, _ = env.step(action)
        
        session = GameSession.query.get(current_sessions[user_id])
        session.total_reward += reward
        
        action_log = ActionLog(
            session_id=session.id,
            action=action,
            reward=reward,
            state=json.dumps(obs["image"].tolist())
        )
        db.session.add(action_log)
        
        if done or truncated:
            session.end_time = datetime.utcnow()
            progress = UserProgress.query.filter_by(
                user_id=user_id,
                environment=session.environment
            ).first()
            progress.trials_completed += 1
            if progress.trials_completed >= TRIALS_PER_ENV:
                progress.completed = True
        
        db.session.commit()

        obs = np.repeat(obs["image"], 4, axis=0)
        obs = np.repeat(obs, 4, axis=1)
        
        img = Image.fromarray(obs.transpose(1, 0, 2))

        return jsonify({
            "image": img_to_base64(img),
            "reward": reward,
            "done": done or truncated,
            "trials_completed": progress.trials_completed if (done or truncated) else None
        })

@app.route('/progress')
@login_required
def get_progress():
    progress = get_current_progress(current_user)
    return jsonify({
        env: {
            'completed': p.completed,
            'trials': p.trials_completed
        } for env, p in progress.items()
    })

@app.route('/logout')
@login_required
def logout():
    user_id = current_user.id
    with lock:
        if user_id in environments:
            environments[user_id].close()
            del environments[user_id]
        if user_id in current_sessions:
            session = GameSession.query.get(current_sessions[user_id])
            session.end_time = datetime.utcnow()
            db.session.commit()
            del current_sessions[user_id]
    
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", threaded=True, port=10001)