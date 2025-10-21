from constants import *

class MazeBase(object):
    def __init__(self):
        self.portalPairs = {}
        self.homeoffset = (0, 0)
        self.ghostNodeDeny = {UP:(), DOWN:(), LEFT:(), RIGHT:()}

    def setPortalPairs(self, nodes):
        for pair in list(self.portalPairs.values()):
            nodes.setPortalPair(*pair)

    def connectHomeNodes(self, nodes):
        if not self.homenodeconnectLeft or not self.homenodeconnectRight:
            return 
        key = nodes.createHomeNodes(*self.homeoffset)
        nodes.connectHomeNodes(key, self.homenodeconnectLeft, LEFT)
        nodes.connectHomeNodes(key, self.homenodeconnectRight, RIGHT)

    def addOffset(self, x, y):
        return x+self.homeoffset[0], y+self.homeoffset[1]

    def denyGhostsAccess(self, ghosts, nodes):
        nodes.denyAccessList(*(self.addOffset(2, 3) + (LEFT, ghosts)))
        nodes.denyAccessList(*(self.addOffset(2, 3) + (RIGHT, ghosts)))

        for direction in list(self.ghostNodeDeny.keys()):
            for values in self.ghostNodeDeny[direction]:
                nodes.denyAccessList(*(values + (direction, ghosts)))


class Maze1(MazeBase):
    def __init__(self):
        MazeBase.__init__(self)
        self.name = "maze1"
        self.portalPairs = {0:((0, 17), (27, 17))}
        self.homeoffset = (11.5, 14)
        self.homenodeconnectLeft = (12, 14)
        self.homenodeconnectRight = (15, 14)
        self.pacmanStart = (15, 26)
        self.fruitStart = (9, 20)
        self.ghostNodeDeny = {UP:((12, 14), (15, 14), (12, 26), (15, 26)), LEFT:(self.addOffset(2, 3),),
                              RIGHT:(self.addOffset(2, 3),)}


class Maze2(MazeBase):
    def __init__(self):
        MazeBase.__init__(self)
        self.name = "maze2"
        self.portalPairs = {0:((0, 4), (27, 4)), 1:((0, 26), (27, 26))}
        self.homeoffset = (11.5, 14)
        self.homenodeconnectLeft = (9, 14)
        self.homenodeconnectRight = (18, 14)
        self.pacmanStart = (16, 26)
        self.fruitStart = (11, 20)
        self.ghostNodeDeny = {UP:((9, 14), (18, 14), (11, 23), (16, 23)), LEFT:(self.addOffset(2, 3),),
                              RIGHT:(self.addOffset(2, 3),)}
        
class Tutorial1(MazeBase):
    def __init__(self):
        MazeBase.__init__(self)
        self.name = "tutorial1"
        self.portalPairs = {}  # No portals
        self.homeoffset = (0, 0)  # No ghost house
        self.homenodeconnectLeft = ()  # Empty instead of None
        self.homenodeconnectRight = ()  # Empty instead of None
        self.pacmanStart = (5, 4)  # Start position in maze_1.txt
        self.fruitStart = None  # No fruit
        self.ghostNodeDeny = {}  # No ghost restrictions
        
class Tutorial2(MazeBase):
    def __init__(self):
        MazeBase.__init__(self)
        self.name = "tutorial2"
        self.portalPairs = {}  # Still no portals
        self.homeoffset = (0, 0)
        self.homenodeconnectLeft = ()  # Empty instead of None
        self.homenodeconnectRight = ()  # Empty instead of None
        self.pacmanStart = (10, 4)  # Adjusted for maze_2.txt
        self.fruitStart = None
        self.ghostNodeDeny = {UP: [(3, 4)]}  # Block ghosts near power pellet

class Tutorial3(MazeBase):
    def __init__(self):
        MazeBase.__init__(self)
        self.name = "tutorial3"
        self.portalPairs = {}
        self.homeoffset = (0, 0)  # Ghost house center in maze_3.txt
        self.homenodeconnectLeft = ()  # Left ghost house node
        self.homenodeconnectRight = ()  # Right ghost house node
        self.pacmanStart = (12, 22)
        self.fruitStart = None
        self.ghostNodeDeny = {UP: [(4, 3)], LEFT: [(4, 3)]}  # Block ghost door


class Tutorial4(MazeBase):
    def __init__(self):
        MazeBase.__init__(self)
        self.name = "tutorial4"
        self.portalPairs = {0: ((3, 0), (7, 0))}  # Portals 7/8 in maze_4.txt
        self.homeoffset = (0, 0)
        self.homenodeconnectLeft = ()  # Empty instead of None
        self.homenodeconnectRight = ()  # Empty instead of None
        self.pacmanStart = (15, 26)
        self.fruitStart = None
        self.ghostNodeDeny = {UP: [(3, 0), (7, 0)]}  # Block portal misuse

class MazeData(object):
    def __init__(self):
        self.obj = None
        self.mazedict = {0:Maze1, 1:Tutorial1, 2:Tutorial2, 3:Tutorial3, 4:Tutorial4, 5:Maze1, 6:Maze2}

    def loadMaze(self, level):
        self.obj = self.mazedict[level%len(self.mazedict)]()
