from shop.db import Database
from shop.dispatcher import Dispatcher


class Order:
    def __init__(self, order_id: int):
        self.order_id = order_id


class OrderProcessor:
    def __init__(self, db: Database):
        self.db = db
        self.dispatcher = Dispatcher()

    def save(self, order: Order) -> None:
        self.db.execute(f"insert {order.order_id}")

    def notify(self, order: Order) -> None:
        self.dispatcher.execute("notify", {"id": order.order_id})


def bulk_load(db: Database, rows: list) -> int:
    db.execute("begin")
    count = db.execute_many(rows)
    db.execute("commit")
    return count


def run_job(cursor) -> None:
    local = Dispatcher()
    local.execute("job", {})
    cursor.execute("select 1")
    Dispatcher.describe()
