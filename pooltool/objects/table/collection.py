from pooltool.game.datatypes import GameType
from pooltool.objects.table.specs import (
    PocketTableSpecs,
    SnookerTableSpecs,
    TableModelDescr,
    TableSpecs,
    TableType,
)
from pooltool.utils.strenum import StrEnum


class TableName(StrEnum):
    """An Enum specifying table names.

    Attributes:
        SEVEN_FOOT_SHOWOOD:
        SNOOKER_GENERIC:
    """

    SEVEN_FOOT_SHOWOOD = "7 Ft Showood"
    SNOOKER_GENERIC = "Generic Snooker"


TABLE_SPECS: dict[TableName, TableSpecs] = {
    TableName.SEVEN_FOOT_SHOWOOD: PocketTableSpecs(
        l=1.9812,
        w=1.9812 / 2,
        cushion_width=2 * 2.54 / 100,
        cushion_height=0.64 * 2 * 0.028575,
        corner_pocket_width=0.118,
        corner_pocket_angle=5.3,
        corner_pocket_depth=0.0417,
        corner_pocket_radius=0.062,
        corner_jaw_radius=0.02095,
        side_pocket_width=0.137,
        side_pocket_angle=7.14,
        side_pocket_depth=0.0685,
        side_pocket_radius=0.0645,
        side_jaw_radius=0.00795,
        height=0.708,
        lights_height=1.99,
        model_descr=TableModelDescr(name="seven_foot_showood"),
    ),
    TableName.SNOOKER_GENERIC: SnookerTableSpecs(
        l=3.569,
        w=1.778,
        cushion_width=47.63 / 1000,
        cushion_height=0.039,
        corner_pocket_width=80.14 / 1000,
        corner_pocket_angle=0,
        corner_pocket_depth=67.35 / 1000,
        corner_pocket_radius=88.9 / 1000,
        corner_jaw_radius=88.90 / 1000,
        side_pocket_width=84.57 / 1000,
        side_pocket_angle=0,
        side_pocket_depth=51.59 / 1000,
        side_pocket_radius=53.19 / 1000,
        side_jaw_radius=66.90 / 1000,
        height=0.708,
        lights_height=1.99,
        model_descr=TableModelDescr(name="snooker_generic"),
    ),
}


_default_table_type_map: dict[TableType, TableName] = {
    TableType.POCKET: TableName.SEVEN_FOOT_SHOWOOD,
    TableType.SNOOKER: TableName.SNOOKER_GENERIC,
}

_default_game_type_map: dict[GameType, TableName] = {
    GameType.EIGHTBALL: TableName.SEVEN_FOOT_SHOWOOD,
    GameType.NINEBALL: TableName.SEVEN_FOOT_SHOWOOD,
    GameType.SNOOKER: TableName.SNOOKER_GENERIC,
}


def default_specs_from_table_type(table_type: TableType) -> TableSpecs:
    return prebuilt_specs(_default_table_type_map[table_type])


def default_specs_from_game_type(game_type: GameType) -> TableSpecs:
    return prebuilt_specs(_default_game_type_map[game_type])


def prebuilt_specs(name: TableName) -> TableSpecs:
    return TABLE_SPECS[name]
