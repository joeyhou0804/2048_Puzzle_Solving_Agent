"""
=============
Writable File
=============
"""

"""
References:
* AI Plays 2048: 
http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf
* What is the optimal algorithm for the game 2048?:
https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048
* 2048-AI:
https://github.com/ovolve/2048-AI/commit/ded6498b8b251ab5a5cf70c9aa4e87c8c99a8f3e
* Alpha-Beta Pruning Practice:
https://inst.eecs.berkeley.edu/~cs61b/fa14/ta-materials/apps/ab_tree_practice/
* Expectiminimax:
https://en.wikipedia.org/wiki/Expectiminimax
"""

import random
from BaseAI import BaseAI
import time
import math

"""
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
"""

class IntelligentAgent(BaseAI):

	def blankHeuristic(self, grid):
		blank = 0
		for row in grid.map:
			for i in row:
				if i == 0:
					blank += 1
		return math.log2(blank)

	def maxValueHeuristic(self, grid):
		max = 0
		for row in grid.map:
			for i in row:
				if i > max:
					max = i
		return math.log2(max)

	# Cite:
	# 2048 - AI:
	# https://github.com/ovolve/2048-AI/commit/ded6498b8b251ab5a5cf70c9aa4e87c8c99a8f3e
	smoothnessVectors = {
		# up
		0: (0, -1),
		# down
		1: (0, 1),
		# right
		2: (1, 0),
		# left
		3: (-1, 0)
	}

	# Cite:
	# 2048 - AI:
	# https://github.com/ovolve/2048-AI/commit/ded6498b8b251ab5a5cf70c9aa4e87c8c99a8f3e
	def findFarthestPosition(self, grid, i, j, vector):
		previous = ()
		while True:
			previous = (i, j)
			i = previous[0] + vector[0]
			j = previous[1] + vector[1]
			if (i < 0 or i > 3) or (j < 0 or j > 3) or grid.map[i][j] == 0:
				break
		return (i, j)

	# Cite:
	# 2048 - AI:
	# https://github.com/ovolve/2048-AI/commit/ded6498b8b251ab5a5cf70c9aa4e87c8c99a8f3e
	def smoothnessHeuristic(self, grid):
		smoothness = 0

		for i in range(4):
			for j in range(4):
				if grid.map[i][j] != 0:
					value = math.log2(grid.map[i][j])
					# for direction in right/down
					for direction in [2, 1]:
						targetCell = self.findFarthestPosition(grid, i, j, self.smoothnessVectors[direction])
						if (targetCell[0] in [0,1,2,3]) and (targetCell[1] in [0,1,2,3]) and grid.map[targetCell[0]][targetCell[1]] != 0:
							targetValue = math.log2(grid.map[targetCell[0]][targetCell[1]])
							smoothness -= abs(value - targetValue)

		return smoothness

	# Cite:
	# 2048 - AI:
	# https://github.com/ovolve/2048-AI/commit/ded6498b8b251ab5a5cf70c9aa4e87c8c99a8f3e
	def monotonicityHeuristic(self, grid):
		totals = [0, 0, 0, 0]

		# up/down direction
		for i in range(4):
			cur = 0
			next = cur + 1
			while next < 4:
				while next < 4 and grid.map[i][next] == 0:
					next += 1
				if next >= 4:
					next -= 1

				currentValue = 0
				if grid.map[i][cur] != 0:
					currentValue = math.log2(grid.map[i][cur])

				nextValue = 0
				if grid.map[i][next] != 0:
					nextValue = math.log2(grid.map[i][next])

				if currentValue > nextValue:
					totals[0] += nextValue - currentValue
				if nextValue > currentValue:
					totals[1] += currentValue - nextValue

				cur = next
				next += 1

		# left/right direction
		for j in range(4):
			cur = 0
			next = cur + 1
			while next < 4:
				while next < 4 and grid.map[next][j] == 0:
					next += 1
				if next >= 4:
					next -= 1

				currentValue = 0
				if grid.map[cur][j] != 0:
					currentValue = math.log2(grid.map[cur][j])

				nextValue = 0
				if grid.map[next][j] != 0:
					nextValue = math.log2(grid.map[next][j])

				if currentValue > nextValue:
					totals[2] += nextValue - currentValue
				if nextValue > currentValue:
					totals[3] += currentValue - nextValue

				cur = next
				next += 1

		return max(totals[0], totals[1]) + max(totals[2], totals[3])

	def heuristic(self, grid):
		return 3.0 * self.blankHeuristic(grid) + 1.0 * self.maxValueHeuristic(grid) + 0.1 * self.smoothnessHeuristic(grid) + 1.0 * self.monotonicityHeuristic(grid)

	def expectiminimax(self, grid, depth, alpha, beta, start):
		if time.process_time() - start >= 0.19 or depth == 0:
			return self.heuristic(grid)
		minUtility = float("inf")

		# computer gives 2
		move2 = self.getComputerMove(grid)
		if move2 and grid.canInsert(move2):
			grid.setCellValue(move2, 2)
		utility2 = self.maxUtilitySearch(grid, depth, alpha, beta, start)

		grid.setCellValue(move2, 0)

		# computer gives 4
		move4 = self.getComputerMove(grid)
		if move4 and grid.canInsert(move4):
			grid.setCellValue(move4, 4)
		utility4 = self.maxUtilitySearch(grid, depth, alpha, beta, start)
		grid.setCellValue(move4, 0)

		# calculate the expectation
		expectUtility = 0.9 * utility2 + 0.1 * utility4
		minUtility = min(minUtility, expectUtility)
		beta = min(beta, minUtility)

		return minUtility

	def maxUtilitySearch(self, grid, depth, alpha, beta, start):
		maxUtility = -float("inf")
		moveset = grid.getAvailableMoves()

		for move in moveset:
			utility = self.expectiminimax(move[1], depth - 1, alpha, beta, start)
			maxUtility = max(maxUtility, utility)

			if maxUtility >= beta:
				break
			alpha = max(maxUtility, alpha)

		return maxUtility

	# def getMove(self, grid):
	# 	# Selects a random move and returns it
	# 	moveset = grid.getAvailableMoves()
	# 	return random.choice(moveset)[0] if moveset else None

	def iterativeDeepening(self, grid):
		# clock
		start = time.process_time()
		depth = 1
		maxUtility = -float("inf")
		alpha = -float("inf")
		beta = float('inf')

		moveset = grid.getAvailableMoves()
		res = moveset[0][0]

		while depth <= 4:
			moveset = grid.getAvailableMoves()
			for move in moveset:
				utility = self.expectiminimax(move[1], depth, alpha, beta, start)
				if utility > maxUtility:
					maxUtility = utility
					res = move[0]
				if maxUtility >= beta:
					break
				alpha = max(maxUtility, alpha)

			# clock
			if time.process_time() - start >= 0.19:
				break
			depth += 1

		return res

	def getMove(self, grid):
		return self.iterativeDeepening(grid)

	# from ComputerAI.py
	def getComputerMove(self, grid):
		""" Returns a randomly selected cell if possible """
		cells = grid.getAvailableCells()
		return random.choice(cells) if cells else None
