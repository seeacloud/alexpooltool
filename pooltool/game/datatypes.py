from pooltool.utils.strenum import StrEnum


class GameType(StrEnum):
    """An Enum for supported game types

    Attributes:
        EIGHTBALL:
        NINEBALL:
        SNOOKER:
    """

    EIGHTBALL = "Eight Ball"
    NINEBALL = "Nine Ball"
    SNOOKER = "Snooker"
