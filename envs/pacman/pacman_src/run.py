import pygame
import os
import sys
# append current path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData
import numpy as np

class GameController(object):
    def __init__(self, level=0):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(True)
        self.level = level
        self.lives = 5
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):      
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), self.mazedata.obj.name+".txt"), 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), self.mazedata.obj.name+"_rotation.txt")
        )
        self.setBackground()
        self.nodes = NodeGroup(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), self.mazedata.obj.name+".txt")
        )
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart))
        self.pellets = PelletGroup(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), self.mazedata.obj.name+".txt")
        )
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)

        if self.level in [0, 5]:
            self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
            self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
            self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
            self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
            self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

            self.nodes.denyHomeAccess(self.pacman)
            self.nodes.denyHomeAccessList(self.ghosts)
            self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
            self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
            self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)

    def startGame_old(self):      
        self.mazedata.loadMaze(self.level)#######
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0,17), (27,17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)

    def update(self, action):
        # dt = self.clock.tick(30) / 1000.0
        dt = 0.05
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)      
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt, action)
        else:
            self.pacman.update(dt, action)

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        # afterPauseMethod = self.pause.update(dt)
        # if afterPauseMethod is not None:
        #     afterPauseMethod()
        self.checkEvents()
        self.render()

    def step(self, action):
        if self.pause.paused:
            self.pause.paused = False
        else:
            self.update(action)

    def checkEvents(self, action=None):
        return
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            self.hideEntities()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                self.pause.setPause(pauseTime=3, func=self.nextLevel)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)                  
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.pause.setPause(pauseTime=3, func=self.restartGame)
                        else:
                            self.pause.setPause(pauseTime=3, func=self.resetLevel)
    
    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self, level=0):
        self.lives = 5
        self.level = level
        self.pause.paused = False
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        # self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            self.screen.blit(self.background, (0, 0))
            #self.nodes.render(self.screen)
            self.pellets.render(self.screen)
            if self.fruit is not None:
                self.fruit.render(self.screen)
            self.pacman.render(self.screen)
            self.ghosts.render(self.screen)
            self.textgroup.render(self.screen)

            for i in range(len(self.lifesprites.images)):
                x = self.lifesprites.images[i].get_width() * i
                y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
                self.screen.blit(self.lifesprites.images[i], (x, y))

            for i in range(len(self.fruitCaptured)):
                x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
                y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
                self.screen.blit(self.fruitCaptured[i], (x, y))

            pygame.display.update()
        else:
            mazefile = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.mazedata.obj.name+".txt")
            map_sprite = np.zeros((36, 28), dtype=int) + 9
            map = np.loadtxt(mazefile, dtype='<U1') # 2d array

            # convert to int, replace characters with numbers 
            replace_dict = {
                '0': 0,
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6,
                'X': 7,
                '.': 8,
                '+': 9,
                'P': 10,
                'p': 11,
                '|': 12,
                'n': 13,
                '-': 14,
                '=': 15,
            }   
            for key, value in replace_dict.items():
                map = np.where(map == key, str(value), map)
            map = map.astype(int)

            for pellet in self.pellets.pelletList:
                map_sprite[int(pellet.position.y//TILEWIDTH)][int(pellet.position.x//TILEWIDTH)] = PELLET
            for powerpellets in self.pellets.powerpellets:
                map_sprite[int(powerpellets.position.y//TILEWIDTH)][int(powerpellets.position.x//TILEWIDTH)] = POWERPELLET
            for ghost in self.ghosts:
                if ghost.visible:
                    map_sprite[int(ghost.position.y//TILEWIDTH)][int(ghost.position.x//TILEWIDTH)] = GHOST
            if self.fruit is not None:
                map_sprite[int(self.fruit.position.y//TILEWIDTH)][int(self.fruit.position.x//TILEWIDTH)] = FRUIT
            map_sprite[int(self.pacman.position.y//TILEWIDTH)][int(self.pacman.position.x//TILEWIDTH)] = PACMAN
            # stack the two arrays
            self.map = np.dstack((map, map_sprite))
            # self.map = map_sprite


    def get_observation(self, mode="rgb_array"):
        if mode == "rgb_array":
            displayed_surface = self.screen.copy()
            displayed_array = pygame.surfarray.array3d(displayed_surface).transpose([1, 0, 2])
            return displayed_array
        else:
            return self.map


    def get_reward(self):
        return self.score

    def is_terminated(self):
        return not self.pacman.alive


if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()



