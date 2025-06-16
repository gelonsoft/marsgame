

from utilities import Hash


class CoreConstants:
    ANSI_RESET = "\u001B[0m"
    ANSI_BLACK = "\u001B[30m"
    ANSI_RED = "\u001B[31m"
    ANSI_GREEN = "\u001B[32m"
    ANSI_YELLOW = "\u001B[33m"
    ANSI_BLUE = "\u001B[34m"
    ANSI_PURPLE = "\u001B[35m"
    ANSI_CYAN = "\u001B[36m"
    ANSI_WHITE = "\u001B[37m"

    nameHash = Hash.GetInstance().hash("name")
    colorHash = Hash.GetInstance().hash("color")
    sizeHash = Hash.GetInstance().hash("size")
    orientationHash = Hash.GetInstance().hash("orientation")
    coordinateHash = Hash.GetInstance().hash("coordinates")
    neighbourHash = Hash.GetInstance().hash("neighbours")
    playerHandHash = Hash.GetInstance().hash("playerHand")
    playersHash = Hash.GetInstance().hash("players")
    imgHash = Hash.GetInstance().hash("img")
    backgroundImgHash = Hash.GetInstance().hash("backgroundImg")

class VisibilityMode:
    VISIBLE_TO_ALL = "VISIBLE_TO_ALL"
    HIDDEN_TO_ALL = "HIDDEN_TO_ALL"
    VISIBLE_TO_OWNER = "VISIBLE_TO_OWNER"
    TOP_VISIBLE_TO_ALL = "TOP_VISIBLE_TO_ALL"
    BOTTOM_VISIBLE_TO_ALL = "BOTTOM_VISIBLE_TO_ALL"
    MIXED_VISIBILITY = "MIXED_VISIBILITY"

class DefaultGamePhase:
    Main = "Main"
    PlayerReaction = "PlayerReaction"
    End = "End"

class ComponentType:
    DECK = "DECK"
    AREA = "AREA"
    BOARD = "BOARD"
    BOARD_NODE = "BOARD_NODE"
    CARD = "CARD"
    COUNTER = "COUNTER"
    DICE = "DICE"
    TOKEN = "TOKEN"

class GameResult:
    WIN_GAME = 1
    WIN_ROUND = 0
    DRAW_GAME = 0
    DRAW_ROUND = 0
    LOSE_ROUND = 0
    LOSE_GAME = -1
    DISQUALIFY = -2
    TIMEOUT = -3
    GAME_ONGOING = 0
    GAME_END = 3

    @staticmethod
    def get_value(result):
        return {
            CoreConstants.GameResult.WIN_GAME: 1,
            CoreConstants.GameResult.WIN_ROUND: 0,
            CoreConstants.GameResult.DRAW_GAME: 0,
            CoreConstants.GameResult.DRAW_ROUND: 0,
            CoreConstants.GameResult.LOSE_ROUND: 0,
            CoreConstants.GameResult.LOSE_GAME: -1,
            CoreConstants.GameResult.DISQUALIFY: -2,
            CoreConstants.GameResult.TIMEOUT: -3,
            CoreConstants.GameResult.GAME_ONGOING: 0,
            CoreConstants.GameResult.GAME_END: 3
        }.get(result, 0)