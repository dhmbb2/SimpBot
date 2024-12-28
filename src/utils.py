from dataclasses import dataclass

@dataclass
class Message:
    """
    identity: 0 stands for user and 1 stands for bot.
    message: A string containing the message.
    """
    identity: int
    message: str