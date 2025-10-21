import sqlite3
import json

DB_PATH = 'instance/gamedata_human_study.db'

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

conn = sqlite3.connect(DB_PATH)
conn.row_factory = dict_factory
cur = conn.cursor()

# 1. Print all table names
table_rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
tables = [t['name'] for t in table_rows]
print("Tables in DB:", tables)

# 2. Print all data in each table
for table in tables:
    print(f"\n--- {table.upper()} ---")
    try:
        rows = cur.execute(f"SELECT * FROM {table}").fetchall()
        for row in rows:
            print(row)
    except Exception as e:
        print(f"Could not read table {table}: {e}")

# 3. Parse db using app structure
print("\n\n=== Hierarchical Parse ===\n")
users = cur.execute("SELECT * FROM user").fetchall()
for user in users:
    print(f"[User] id={user['id']} email={user['email']} created={user['created_at']}")
    sessions = cur.execute(
        "SELECT * FROM game_session WHERE user_id = ? ORDER BY start_time", (user['id'],)
    ).fetchall()
    for session in sessions:
        print(f"  [Session] id={session['id']} task={session['task_index']} start={session['start_time']} end={session['end_time']} reward={session['total_reward']}")
        actions = cur.execute(
            "SELECT * FROM action_log WHERE session_id = ? ORDER BY timestamp", (session['id'],)
        ).fetchall()
        for action in actions:
            # Parse state JSON (may be huge)
            try:
                state = json.loads(action['state']) if action['state'] else None
            except Exception as e:
                state = f"[Invalid JSON: {e}]"
            print(f"    [Action] id={action['id']} time={action['timestamp']} action={action['action']} reward={action['reward']} state_type={type(state)} state_dim={len(state) if state and hasattr(state,'__len__') else 'NA'}")
            # To print full state, uncomment below:
            # print("      state:", state)

conn.close()
