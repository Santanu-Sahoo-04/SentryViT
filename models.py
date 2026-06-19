# db/models.py
import sqlite3
from datetime import datetime

class SessionLogger:
    def __init__(self, db_name="cogniguard.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS focus_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                cognitive_state TEXT,
                slouch_index REAL
            )
        ''')
        self.conn.commit()

    def log_state(self, state, slouch_val=0.0):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("INSERT INTO focus_logs (timestamp, cognitive_state, slouch_index) VALUES (?, ?, ?)", 
                            (now, state, slouch_val))
        self.conn.commit()
        
    def get_session_summary(self):
        self.cursor.execute("SELECT cognitive_state, COUNT(*) FROM focus_logs GROUP BY cognitive_state")
        return self.cursor.fetchall()