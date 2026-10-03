class Database:
    def __init__(self, url: str):
        self.url = url

    def execute(self, sql: str) -> list:
        return [sql]

    def execute_many(self, rows: list) -> int:
        for row in rows:
            self.execute(row)
        return len(rows)


def connect(url: str) -> Database:
    return Database(url)
