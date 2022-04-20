import heapq
from itertools import count



#  ************************************************************************************************************************ 
# HELPER FUNCTIONS
# *************************************************************************************************************************

# Referenced: https://www.adamsmith.haus/python/answers/how-to-convert-a-list-of-integers-into-a-single-integer-in-python
def state_to_int(state):
    """
    Converts a state to an integer.
    """
    return int("".join(map(str, state)))


# *************************************************************************************************************************


# Manhattan distance heuristic function
def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    # the sum of the absolute difference between their element's x coordinates plus the absolute distance between their y
    # coordinates
    sum = 0
    for i in range(len(from_state)):
        from_idx = from_state.index(to_state[i])
        if(from_state[from_idx] != 0):
            sum += abs(from_idx % 3 - i % 3) + abs(from_idx // 3 - i // 3) # from_idx // 3 is the row, from_idx % 3 is the column
    return sum


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []

    # store the index of both the "empty tiles"
    empty_tile_idx1 = state.index(0)
    empty_tile_idx2 = state.index(0, empty_tile_idx1 + 1)

     # find the indexes of the tiles above the empty tiles
    above_tile_idx1 = empty_tile_idx1 - 3
    above_tile_idx2 = empty_tile_idx2 - 3

    # find the indexes of the tiles below the empty tiles
    below_tile_idx1 = empty_tile_idx1 + 3
    below_tile_idx2 = empty_tile_idx2 + 3

    # find the indexes of the tiles to the left of the empty tiles
    left_tile_idx1 = empty_tile_idx1 - 1
    left_tile_idx2 = empty_tile_idx2 - 1

    # find the indexes of the tile to the right of the empty tile
    right_tile_idx1 = empty_tile_idx1 + 1
    right_tile_idx2 = empty_tile_idx2 + 1

    # CHECK #1

    # FIRST 0: check if the tile to the left of the empty tile is valid

    # left_tile_idx1 is the index of the tile to the left of the first empty tile in state
    if left_tile_idx1 >= 0 and  left_tile_idx1 % 3 != 2 and state[left_tile_idx1] != 0:  # Left tile can't be in right most column

        # create a new state with the tile to the left of the empty tile moved to the empty tile

        new_state = state[:] # Shallow copy

        # swap the empty tile with the tile to the left of the empty tile
        new_state[empty_tile_idx1], new_state[left_tile_idx1] = new_state[left_tile_idx1], new_state[empty_tile_idx1]

        # append the new state to the list of successors
        if(new_state not in succ_states):
            succ_states.append(new_state)

    # SECOND 0: check if the tile to the left of the second empty tile is valid
    if left_tile_idx2 >= 0 and  left_tile_idx2 % 3 != 2 and state[left_tile_idx2] != 0:  # Left tile can't be in right most column

        # create a new state with the tile to the left of the empty tile moved to the empty tile

        new_state = state[:] # Shallow copy

        # swap the empty tile with the tile to the left of the empty tile
        new_state[empty_tile_idx2], new_state[left_tile_idx2] = new_state[left_tile_idx2], new_state[empty_tile_idx2]

        # append the new state to the list of successors
        if(new_state not in succ_states):
            succ_states.append(new_state)
    

    # CHECK #2
    # FIRST 0: check if the tile to the right of the empty tile is valid
    # if right_tile_idx1 <= 8 and right_tile_idx1 % 3 != 0 and state[right_tile_idx1] != 0:

    if right_tile_idx1 <= 8 and right_tile_idx1 % 3 != 0 and state[right_tile_idx1] != 0: # Right tile can't be in left most column

        # create a new state with the tile to the right of the empty tile moved to the empty tile

        new_state = state[:]

        # swap the empty tile with the tile to the right of the empty tile
        new_state[empty_tile_idx1], new_state[right_tile_idx1] = new_state[right_tile_idx1],  new_state[empty_tile_idx1]

        # append the new state to the list of successors
        if(new_state not in succ_states):
            succ_states.append(new_state)

    # SECOND 0: check if the tile to the right of the empty tile is valid   

    if right_tile_idx2 <= 8 and right_tile_idx2 % 3 != 0 and state[right_tile_idx2] != 0: # Right tile can't be in left most column

        # create a new state with the tile to the right of the empty tile moved to the empty tile

        new_state = state[:]

        # swap the empty tile with the tile to the right of the empty tile
        new_state[empty_tile_idx2], new_state[right_tile_idx2] = new_state[right_tile_idx2],  new_state[empty_tile_idx2]

        # append the new state to the list of successors
        if(new_state not in succ_states):
            succ_states.append(new_state)

    # CHECK #3
    # FIRST 0: check if the tile above the empty tile is valid

    # if above_tile_idx1 >= 0 and above_tile_idx1 % 3 != 2 and state[above_tile_idx1] != 0:
    if above_tile_idx1 >= 0 and above_tile_idx1 // 3 != 2 and state[above_tile_idx1] != 0: # Above tile can't be in bottom row

        # create a new state with the tile above the empty tile moved to the empty tile

        new_state = state[:]

        # swap the empty tile with the tile above the empty tile
        new_state[empty_tile_idx1], new_state[above_tile_idx1] = new_state[above_tile_idx1], new_state[empty_tile_idx1]

        # append the new state to the list of successors
        if(new_state not in succ_states):
            succ_states.append(new_state)

    # SECOND 0: check if the tile above the empty tile is valid
    # if above_tile_idx2 >= 0 and above_tile_idx2 % 3 != 2 and state[above_tile_idx2] != 0:
    if above_tile_idx2 >= 0 and above_tile_idx2 // 3 != 2 and state[above_tile_idx2] != 0: # Above tile can't be in bottom row

        # create a new state with the tile above the empty tile moved to the empty tile

        new_state = state[:]

        # swap the empty tile with the tile above the empty tile
        new_state[empty_tile_idx2], new_state[above_tile_idx2] = new_state[above_tile_idx2], new_state[empty_tile_idx2]

        # append the new state to the list of successors
        if(new_state not in succ_states):
            succ_states.append(new_state)

    # CHECK #4
    # FIRST 0: check if the tile below the empty tile is valid
    # if below_tile_idx1 <= 8 and below_tile_idx1 % 3 != 2 and state[below_tile_idx1] != 0:

    if below_tile_idx1 <= 8 and below_tile_idx1 // 3 != 0 and state[below_tile_idx1] != 0: # Below tile can't be in top row

        # create a new state with the tile below the empty tile moved to the empty tile

        new_state = state[:]

        # swap the empty tile with the tile below the empty tile
        new_state[empty_tile_idx1], new_state[below_tile_idx1] = new_state[below_tile_idx1],  new_state[empty_tile_idx1]

        # append the new state to the list of successors
        if(new_state not in succ_states):
            succ_states.append(new_state)

    # SECOND 0: check if the tile below the empty tile is valid

    # if below_tile_idx2 <= 8 and below_tile_idx2 % 3 != 2 and state[below_tile_idx2] != 0:
        
    if below_tile_idx2 <= 8 and below_tile_idx2 // 3 != 0 and state[below_tile_idx2] != 0: # Below tile can't be in top row

        # create a new state with the tile below the empty tile moved to the empty tile

        new_state = state[:]

        # swap the empty tile with the tile below the empty tile
        new_state[empty_tile_idx2], new_state[below_tile_idx2] = new_state[below_tile_idx2],  new_state[empty_tile_idx2]

        # append the new state to the list of successors
        if(new_state not in succ_states):
            succ_states.append(new_state)

    # sort and return a list of successors
   
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    # Root state
    root_state = state
    root_key = state_to_int(root_state)

    # open list
    open = []

    # create a dictionary to store the visited states
    visited = {}

    # create a dictionary to store the parent of each state
    parent = {} 

    # create a dictionary to store the h value of each state
    h = {}

    # create a dictionary to store the g value of each state
    g = {}

    # create a dictionary to store the f value of each state
    f = {}

    # intialize all fields being tracked
    parent[root_key] = -1                            # Root state has no parent
    h[root_key] = get_manhattan_distance(root_state) # Compute root state Manhatten distance
    g[root_key] = 0                                  # Root state is the first move -> g = 0
    f[root_key] = 0                                  # Cost of root state is just 0

    heapq.heappush(open, (f[root_key], root_state,  (g[root_key], h[root_key], parent[root_key])))  # (pq ,(cost, state, (g, h, parent_index)))
    max_len_pq = 1
    # while the priority queue is not empty
    while open: 
        # pop the first element from the priority queue
        current_state = heapq.heappop(open)[1]
        # convert the state to an integer, this is the key
        current_key = state_to_int(current_state)
        # set the visited state field to true
        visited[current_key] = True
        # if the current state is the goal state
        if(current_state == goal_state):
            # create a list to store the final path of states
            final_path = []
            # add the goal state to the final path
            final_path.append(current_state)
            # while the parent of the current state is not the root state
            while(parent[current_key] != -1):
                # if the current state isn't already in the final path
                if(current_state not in final_path):
                    # add the current state to the final path
                    final_path.append(current_state)
                 # generate key for the new parent state
                current_key = state_to_int(current_state)
                # set the current state to the parent of the current state
                current_state = parent[current_key]
            # reverse the final path (the root node is added last)
            final_path.reverse()
            # print the final path
            for i in range(len(final_path)):
                print(final_path[i], "h=" + str(h[state_to_int(final_path[i])]), "moves:", g[state_to_int(final_path[i])])
            # print the max queue length
            print("Max queue length:", max_len_pq)
            exit()
        # get the successors of the current state
        T = get_succ(current_state)
        for succ_state in T:
            # convert the successor state to an integer, this is the key
            succ_key = state_to_int(succ_state) 
            # if the successor state hasn't been visited
            if(succ_key not in visited):
                # initialize the fields being tracked
                visited[succ_key] = True
                parent[succ_key] = current_state
                h[succ_key] = get_manhattan_distance(succ_state)
                g[succ_key] = g[current_key] + 1
                f[succ_key] = h[succ_key] + g[succ_key]
                # add the successor state to the priority queue if it is not in it already
                if((f[succ_key], succ_state, (g[succ_key], h[succ_key], parent[succ_key])) not in open):
                    heapq.heappush(open, (f[succ_key], succ_state, (g[succ_key], h[succ_key], parent[succ_key])))
                    # update the max queue length
                    if(len(open) > max_len_pq):
                        max_len_pq = len(open)

if __name__ == "__main__":
    """
    Feel free to write your own test code here to examine the correctness of your functions. 
    Note that this part will not be graded.
    """

    # Test: Solve 1
    solve([3, 4, 6, 0, 0, 1, 7, 2, 5], goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0])
    # Test: Solve 2
    # solve([6, 0, 0, 3, 5, 1, 7, 2, 4], goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0])
     # Test: Solve 3
    # solve([0, 4, 7, 1, 3, 0, 6, 2, 5], goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0])
     # Test: Solve 4
    # solve([5, 2, 3, 0, 6, 4, 7, 1, 0], goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0])
     # Test: Solve 5
    # solve([1, 7, 0, 6, 3, 2, 0, 4, 5], goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0])

    # solve([3, 4, 6, 0, 0, 1, 7, 2, 5], goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0])
    # solve([2,5,1,4,0,6,7,0,3], goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0])

