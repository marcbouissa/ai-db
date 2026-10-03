class Dispatcher:
    def __init__(self):
        self.handlers: dict = {}

    def execute(self, name: str, payload: dict) -> object:
        return self.handlers[name](payload)

    @staticmethod
    def describe() -> str:
        return "dispatcher"
