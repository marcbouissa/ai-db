import os
import time

def run_watch(db_cls, db_path: str, target_dir: str, interval: float = 2.0):
    target_dir = os.path.abspath(target_dir)
    print(f"[ai-db watch] Monitoring '{target_dir}' -> DB: '{db_path}' (interval: {interval}s)")

    db = db_cls(db_path)
    db.sync(target_dir)
    db.close()

    while True:
        try:
            time.sleep(interval)
            db = db_cls(db_path)
            res = db.sync(target_dir, verbose=False)
            db.close()
            if res["added"] > 0 or res["updated"] > 0 or res["pruned"] > 0:
                print(f"[ai-db watch] Change detected: +{res['added']} ~{res['updated']} -{res['pruned']}")
        except KeyboardInterrupt:
            print("\n[ai-db watch] Stopped.")
            break
        except Exception:
            time.sleep(interval)
