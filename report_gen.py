# analytics/report_gen.py
import sqlite3

class AnalyticsEngine:
    def __init__(self, db_path="cogniguard.db"):
        self.db_path = db_path

    def generate_terminal_report(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Fetch total logs
            cursor.execute("SELECT COUNT(*) FROM logs")
            total_records = cursor.fetchone()[0]
            
            if total_records == 0:
                print("\n[ERROR] No logs found in database to generate a report.")
                conn.close()
                return

            # Fetch state distributions
            cursor.execute("SELECT state, COUNT(*) FROM logs GROUP BY state")
            distributions = cursor.fetchall()
            
            print("\n" + "="*40)
            print("         SENTRYVIT SESSION ANALYTICS")
            print("="*40)
            print(f"Total Monitored Intervals: {total_records}")
            print("-"*40)
            print("Detected Activity & Object Breakdown:")
            
            for state, count in distributions:
                percentage = (count / total_records) * 100
                print(f"[{state}]: {percentage:.1f}% ({count} triggers)")
                
            print("="*40 + "\n")
            conn.close()
            
        except sqlite3.OperationalError:
            print("\n[INFO] No active session database found or database file is empty.")