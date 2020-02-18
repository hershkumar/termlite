from asciimatics.scene import Scene
from asciimatics.screen import Screen
from asciimatics.widgets import Frame, ListBox, Layout, Divider, Text, \
    Button, TextBox, Widget
from asciimatics.exceptions import ResizeScreenError, NextScene, StopApplication
from time import sleep
from player import *
from PIL import Image
import numpy as np


def load_image(filename) :
    img = Image.open(filename)
    img.load()
    data = np.asarray(img)
    return data


terrain_colors = {
	'lowlands': 220,
    'desert': 226,
    'forest': 22,
    'ocean': 19,
    'plains': 118,
    'foothills': 245,
    'mountain': 255,
}


def getBiome(terrain,x,y):
	height = terrain[x][y]
	if (height >= 215):
		return terrain_colors['mountain']

	elif (height >= 170):
		return terrain_colors['foothills']

	elif (height >= 160):
		return terrain_colors['forest']

	elif (height >= 125):
		return terrain_colors['plains']

	elif (height >= 100):
		return terrain_colors['desert']

	elif (height >= 75):
		return terrain_colors['lowlands']

	else:
		return terrain_colors['ocean']


def demo(screen):
	screen.clear()
	screen_width = screen.width
	screen_height = screen.height
	hud_height = 4
	heightmap = load_image("terrain.png")
	terrain = np.copy(heightmap)
	# get some constants so we can normalize all heightmaps to be between 0 and 255
	min_height = np.amin(terrain)
	max_height = np.amax(terrain)
	avg_height = np.average(terrain)
	# draw the terrain
	for i in range(screen_width):
		for j in range(hud_height,screen_height):
			terrain[i][j] = (terrain[i][j] - min_height)/(max_height - min_height) * (255) 

			height = terrain[i][j]
			color = getBiome(terrain,i,j)
			screen.print_at(" ",i,j,bg=color)
	screen.refresh()
	# enter the game loop once the map has rendered:
	exit = False
	character = Player("Hersh", 5, 5)
	screen.print_at("@",character.x,character.y,bg=getBiome(terrain,character.x,character.y))
	screen.print_at("X: "+str(character.x),0,0)
	screen.print_at("Y: "+str(character.y),len("X: "+str(character.x))+1,0)			
	screen.print_at("Name: " + character.name,0,1)
	while (exit != True):
		# make a new player at (0,0)
		
		event = screen.get_event()
		if (event != None):
			# clear the hud
			for i in range(screen_width):
				for j in range(hud_height):
					screen.print_at(" ",i,j)
			# print hud
			screen.print_at("X: "+str(character.x),0,0)
			screen.print_at("Y: "+str(character.y),len("X: "+str(character.x))+1,0)			
			screen.print_at("Name: " + character.name,0,1)
			#check if they want to exit (~)
			if (event.key_code == 96):
				exit=True
			if (event.key_code == 119):
				# fix the terrain that we were just at
				screen.print_at(" ",character.x,character.y,bg=getBiome(terrain,character.x,character.y))
				#check to make sure it isn't water
				if (getBiome(terrain,character.x,character.y-1) != terrain_colors['ocean']):
					if (character.y -1 >= hud_height):
						character.y -= 1
			if (event.key_code == 115):
				screen.print_at(" ",character.x,character.y,bg=getBiome(terrain,character.x,character.y))
				if (getBiome(terrain,character.x,character.y+1) != terrain_colors['ocean']):
					if (character.y+1 >= hud_height):
						character.y += 1
			if (event.key_code == 97):
				screen.print_at(" ",character.x,character.y,bg=getBiome(terrain,character.x,character.y))
				if (getBiome(terrain,character.x-1,character.y) != terrain_colors['ocean']):
					character.x -= 1
			if (event.key_code == 100):
				screen.print_at(" ",character.x,character.y,bg=getBiome(terrain,character.x,character.y))
				if (getBiome(terrain,character.x+1,character.y) != terrain_colors['ocean']):
					character.x += 1
		# print the actual character
		screen.print_at("@",character.x,character.y,screen.COLOUR_GREEN,screen.A_BOLD,bg=getBiome(terrain,character.x,character.y))
		screen.refresh()
Screen.wrapper(demo)
