"""
votre agent. Vous devez modifier ce fichier.
"""

from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import math
import time
import random
import numpy as np
import json


class MyPlayer2(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """
    #Open test.json as dict
    param_tuning = json.load(open("tuning_params.json", "r"))

    log = True #True if you want to write the log of each action in a text file

    #Coordinate system conversion into cube coordinates
    #https://www.redblobgames.com/grids/hexagons/#coordinates-cube
    to_cube = { (4,0) : (0,4,-4), (3,1) : (1,3,-4), (2,2) : (2,2,-4), (1,3) : (3,1,-4), (0,4) : (4,0,-4),
                (6,0) : (-1, 4,-3), (5,1) : (0,3,-3), (4,2) : (1,2,-3), (3,3) : (2,1,-3), (2,4) : (3,0,-3), (1,5) : (4,-1,-3),
                (8,0) : (-2,4,-2), (7,1) : (-1,3,-2), (6,2) : (0,2,-2), (5,3) : (1,1,-2), (4,4) : (2,0,-2), (3,5) : (3,-1,-2), (2,6) : (4,-2,-2),
                (10,0) : (-3,4,-1), (9,1) : (-2,3,-1), (8,2) : (-1,2,-1), (7,3) : (0,1,-1), (6,4) : (1,0,-1), (5,5) : (2,-1,-1), (4,6) : (3,-2,-1), (3,7) : (4,-3,-1),
                (12,0) : (-4,4,0), (11,1) : (-3,3,0), (10,2) : (-2,2,0), (9,3) : (-1,1,0), (8,4) : (0,0,0), (7,5) : (1,-1,0), (6,6) : (2,-2,0), (5,7) : (3,-3,0), (4,8) : (4,-4,0),
                (13,1) : (-4,3,1), (12,2) : (-3,2,1), (11,3) : (-2,1,1), (10,4) : (-1,0,1), (9,5) : (0,-1,1), (8,6) : (1,-2,1), (7,7) : (2,-3,1), (6,8) : (3,-4,1),
                (14,2) : (-4,2,2), (13,3) : (-3,1,2), (12,4) : (-2,0,2), (11,5) : (-1,-1,2), (10,6) : (0,-2,2), (9,7) : (1,-3,2), (8,8) : (2,-4,2), 
                (15,3) : (-4,1,3), (14,4) : (-3,0,3), (13,5) : (-2,-1,3), (12,6) : (-1,-2,3), (11,7) : (0,-3,3), (10,8) : (1,-4,3),
                (16,4) : (-4,0,4), (15,5) : (-3,-1,4), (14,6) : (-2,-2,4), (13,7) : (-1,-3,4), (12,8) : (0,-4,4)
                }

    to_coord = { (0,4,-4) : (4,0), (1,3,-4) : (3,1), (2,2,-4) : (2,2), (3,1,-4) : (1,3), (4,0,-4) : (0,4),
                (-1, 4,-3) : (6,0), (0,3,-3) : (5,1), (1,2,-3) : (4,2), (2,1,-3) : (3,3), (3,0,-3) : (2,4), (4,-1,-3) : (1,5),
                (-2,4,-2) : (8,0), (-1,3,-2) : (7,1), (0,2,-2) : (6,2), (1,1,-2) : (5,3), (2,0,-2) : (4,4), (3,-1,-2) : (3,5), (4,-2,-2) : (2,6),
                (-3,4,-1) : (10,0), (-2,3,-1) : (9,1), (-1,2,-1) : (8,2), (0,1,-1) : (7,3), (1,0,-1) : (6,4), (2,-1,-1) : (5,5), (3,-2,-1) : (4,6), (4,-3,-1) : (3,7),
                (-4,4,0) : (12,0), (-3,3,0) : (11,1), (-2,2,0) : (10,2), (-1,1,0) : (9,3), (0,0,0) : (8,4), (1,-1,0) : (7,5), (2,-2,0) : (6,6), (3,-3,0) : (5,7), (4,-4,0) : (4,8),
                (-4,3,1) : (13,1), (-3,2,1) : (12,2), (-2,1,1) : (11,3), (-1,0,1) : (10,4), (0,-1,1) : (9,5), (1,-2,1) : (8,6), (2,-3,1) : (7,7), (3,-4,1) : (6,8),
                (-4,2,2) : (14,2), (-3,1,2) : (13,3), (-2,0,2) : (12,4), (-1,-1,2) : (11,5), (0,-2,2) : (10,6), (1,-3,2) : (9,7), (2,-4,2) : (8,8), 
                (-4,1,3) : (15,3), (-3,0,3) : (14,4), (-2,-1,3) : (13,5), (-1,-2,3) : (12,6), (0,-3,3) : (11,7), (1,-4,3) : (10,8),
                (-4,0,4) : (16,4), (-3,-1,4) : (15,5), (-2,-2,4) : (14,6), (-1,-3,4) : (13,7), (0,-4,4) : (12,8)
                }

    grid_to_coord = { (0,6) : (0,4), (0,5) : (1,3), (0,4) : (2,2), (0,3) : (3,1), (0,2) : (4,0),
                        (1,6) : (1,5), (1,5) : (2,4), (1,4) : (3,3), (1,3) : (4,2), (1,2) : (5,1), (1,1) : (6,0),
                        (2,7) : (2,6), (2,6) : (3,5), (2,5) : (4,4), (2,4) : (5,3), (2,3) : (6,2), (2,2) : (7,1), (2,1) : (8,0),
                        (3,7) : (3,7), (3,6) : (4,6), (3,5) : (5,5), (3,4) : (6,4), (3,3) : (7,3), (3,2) : (8,2),
                        (3,1) : (9,1), (4,8) : (4,8), (4,7) : (5,7), (4,6) : (6,6), (4,5) : (7,5), (4,4) : (8,4),
                        (4,3) : (9,3), (5,7) : (6,8), (5,6) : (7,7), (5,5) : (8,6), (5,4) : (9,5), (6,7) : (8,8),
                        (6,6) : (9,7), (5,3) : (10,4), (5,2) : (11,3), (5,1) : (12,2), (5,0) : (13,1), (4,2) : (10,2),
                        (4,1) : (11,1), (4,0) : (12,0), (6,5) : (10,6), (6,4) : (11,5), (6,3) : (12,4), (6,2) : (13,3),
                        (6,1) : (14,2), (7,6) : (10,8), (7,5) : (11,7), (7,4) : (12,6), (7,3) : (13,5), (7,2) : (14,4),
                        (7,1) : (15,3), (8,6) : (12,8), (8,5) : (13,7), (8,4) : (14,6), (8,3) : (15,5), (8,2) : (16,4),
                        (3,0) : (10,0)
                }
                


    def __init__(self, piece_type: str, name: str = "bob", time_limit: float=60*15,*args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type,name,time_limit,*args)

        self.tt = TranspositionTable()

        self.visited_nodes = 0
        self.visited_leaf = 0
        self.cumulated_time = 0
        self.action_number = 0
        self.text_file_log = f"log_{self.piece_type}.txt"
        if MyPlayer2.log:
            #Initialize the log file
            with open(self.text_file_log, "w") as f:
                f.write("")


    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        self.time_consumed = time.time()

        maximizing_player = self.piece_type == "W"
        depth = 3

        value, best_action = self.minmax_alphabeta(current_state, depth, -math.inf, math.inf, maximizing_player)

        self.time_consumed = time.time() - self.time_consumed
        self.cumulated_time += self.time_consumed

        self.action_number += 1
        if MyPlayer2.log:
            with open(self.text_file_log, "a") as f:
                f.write(f"Action {self.action_number} : {round(self.time_consumed,4)} s (cum. time {round(self.cumulated_time,4)} s), {self.visited_nodes} visited nodes, {self.visited_leaf} visited leaves, size tt {len(self.tt.tab)}, {round(value,8)} value\n")
                
        self.visited_nodes = 0
        self.visited_leaf = 0

        return best_action



    
    def winner_score(self, current_state: GameState) -> float:
        """
        Function to implement the winner score when the game is finished.
        """

        #Get player list
        players = current_state.get_players()
        player_score = current_state.get_scores()
        assert len(players) == 2

        w_id = players[0].get_id() if players[0].get_piece_type() == "W" else players[1].get_id()
        b_id = players[0].get_id() if players[0].get_piece_type() == "B" else players[1].get_id()

        if player_score[b_id] == -6 : #White player won
            return 100
        elif player_score[w_id] == -6 : #Black player won
            return -100
        else :
            return 2*(player_score[w_id] - player_score[b_id])

    
    def utility(self, current_state: GameState, depth: int, maximizing_player : bool) -> float:
        """
        Function to implement the utility function.

        Args:
            current_state (GameState): Current game state representation
            depth (int): depth of the tree
            maximizing_player (bool): True if the player is maximizing

        Returns:
            float: the value of the utility function
        """
        if current_state.is_done():
            return self.winner_score(current_state)
        elif depth == 0:
            return self.heuristic(current_state)
        else :
            raise Exception("Utility used on a non terminal node")

    
    def heuristic(self, current_state: GameState) -> float:
        """
        Function to implement the heuristic function to evaluate the value of a node.

        1) Number of marbles difference (max 5 because if 6, a player has necessarily won)
        2) Number of marbles on the edge
        3) Sum of Manhattan distance to the center
        4) Marbles cohesion
        5) Threat

        Args:
            current_state (GameState): Current game state representation

        Returns:
            float: the value of the heuristic function
        """
        self.visited_leaf += 1

        marble_count = {"W" : 0, "B" : 0}
        distance_center = {"W" : 0, "B" : 0}
        edge_count = {"W" : 0, "B" : 0}
        cohesion = {"W" : 0, "B" : 0}
        threat = {"W" : 0, "B" : 0}
        density = {"W" : 0, "B" : 0}


        #Loop over the environnement current_state.get_rep().get_env() = {  (a,b) :  <seahorse.game.game_layout.board.Piece> , ... }
        for i, (coord, marble) in enumerate(current_state.get_rep().get_env().items()):
            color_marble = marble.get_type()
            opposition_color_marble = "W" if color_marble == "B" else "B"
            x,y,z = MyPlayer2.to_cube[coord]


            #########################################
            ####### Update the marble count #########
            #########################################
            marble_count[color_marble] += 1


            ########################################################################################################################
            ####### Update the distance to the center that is actually adding the max absolute value of the cube coordinates #######
            ########################################################################################################################
            maxi_cube_coord = max(abs(x), abs(y), abs(z))
            distance_center[color_marble] += maxi_cube_coord
            

            #Check if the marble is on the edge
            if maxi_cube_coord == 4:
                ########################################
                ######## Update the edge count #########
                ########################################
                edge_count[color_marble] += 1
                

                ##################################################
                ######## Check if the marble is threaten #########
                ##################################################
                threaten = False
                blocked_bonus = 0 #TODO: Bonus if the marble cannot escape the threat
                #Find the direction to check
                direction_to_check = []
                if abs(x) == 4 :
                    direction_to_check.append(  ( -int(math.copysign(1, x)) , int(math.copysign(1, x)) , 0)  )
                    direction_to_check.append(  ( -int(math.copysign(1, x)) , 0 , int(math.copysign(1, x)) )  )
                if abs(y) == 4 :
                    direction_to_check.append(  (0 , -int(math.copysign(1, y)) , int(math.copysign(1, y)) )  )
                    direction_to_check.append(  (int(math.copysign(1, y)) , -int(math.copysign(1, y)) , 0)  )
                if abs(z) == 4 :
                    direction_to_check.append(  (int(math.copysign(1, z)) , 0 , -int(math.copysign(1, z)))  )
                    direction_to_check.append(  (0 , int(math.copysign(1, z)) , -int(math.copysign(1, z)))  )
                direction_to_check = list(set(direction_to_check))
                for a,b,c in direction_to_check :
                    #Check superiority in the direction (to see if knock out is possible)
                    color_edge_count = 0
                    color_opponent_count = 0
                    color_in_progress = color_marble
                    cur = 0
                    color_cur = color_marble
                    #Count the edge's color count
                    while color_cur == color_in_progress :
                        color_edge_count += 1
                        cur += 1
                        piece_cur = current_state.get_rep().find( MyPlayer2.to_coord.get( (x+a*cur, y+b*cur, z+c*cur) , -1 ) )
                        if piece_cur == -1 :
                            break
                        else :
                            color_cur = piece_cur.get_type()
                    #Count the opponent's color count
                    while color_cur == opposition_color_marble :
                        color_opponent_count += 1
                        cur += 1
                        piece_cur = current_state.get_rep().find( MyPlayer2.to_coord.get( (x+a*cur, y+b*cur, z+c*cur) , -1) )
                        if piece_cur == -1 :
                            break
                        else :
                            color_cur = piece_cur.get_type()
                    if color_opponent_count > color_edge_count :
                        threaten = True
                        break
                if threaten :
                    threat[opposition_color_marble] += 1 #The opponent threats
    

            ############################################################################
            ###### Update the cohesion (number of neighbours of the same color) ########
            ############################################################################
            neighbours_cube_coord = [  (x+1, z-1, y), (x+1, z, y-1), (x, z+1, y-1), (x-1, z+1, y), (x-1, z, y+1), (x, z-1, y+1) ]
            for x_n, y_n, z_n in neighbours_cube_coord:
                neighbour_coord = MyPlayer2.to_coord.get( (x_n, y_n, z_n) , -1) 
                neighbour = current_state.get_rep().find( neighbour_coord )
                if neighbour != -1 and neighbour.get_type() == color_marble:
                    cohesion[color_marble] += 1

            ###########################################################################
            ############################## DENSITY ####################################
            ###########################################################################
            for j, (coord2, marble2) in enumerate(current_state.get_rep().get_env().items()):
                if j >= i :
                    break
                color_marble2 = marble2.get_type()
                if color_marble == color_marble2 :
                    x2,y2,z2 = MyPlayer2.to_cube[coord2]
                    density[color_marble] += ( abs(x - x2) + abs(y - y2) + abs(z - z2) )/2 #Manhattan distance



        #1) Number of marbles difference (max 5 because if 6, a player has necessarily won)
        diff =  marble_count["W"] - marble_count["B"]

        #2) Number of marbles on the edge
        edge = edge_count["B"]/marble_count["B"] - edge_count["W"]/marble_count["W"]

        #3) Sum of Manhattan distance to the center
        center = (distance_center["B"]/marble_count["B"] - distance_center["W"]/marble_count["W"])/4

        #4) Marbles cohesion
        cohesion_diff = (cohesion["W"]/marble_count["W"] - cohesion["B"]/marble_count["B"])/6

        #5) Threat
        threat_diff = threat["W"] - threat["B"]

        #6) Density
        density_diff =  (density["B"]/(marble_count["B"]*(marble_count["B"] - 1))  - density["W"]/(marble_count["W"]*(marble_count["W"] - 1)) )/2

        #print(f"diff : {diff}, edge : {edge}, center : {center}, cohesion : {cohesion_diff}, threat : {threat}, density : {k}") 

        return MyPlayer2.param_tuning["current_test"]["diff"]*diff + MyPlayer2.param_tuning["current_test"]["edge"]*edge + MyPlayer2.param_tuning["current_test"]["center"]*center + MyPlayer2.param_tuning["current_test"]["cohesion_diff"]*cohesion_diff + MyPlayer2.param_tuning["current_test"]["threat_diff"]*threat_diff + MyPlayer2.param_tuning["current_test"]["density_diff"]*density_diff




    def guess_value(self, action : Action, maximizing_player : bool = True) -> float:
        """
        Function to guess the value of an action (for ordering the actions).

        1) If the move is push -> +1
        2) If the move is get out of edge -> +1

        Remove dumb moves  ==> return None if the move is dumb and so no need to be proposed to the player
            - throughing one of our own marble out
        """
        current_env = action.get_current_game_state().get_rep().get_env()
        next_env = action.get_next_game_state().get_rep().get_env() 

        ## Look for difference
        numpy_current_grid = np.array( action.get_current_game_state().get_rep().get_grid() )
        numpy_next_grid = np.array( action.get_next_game_state().get_rep().get_grid() )
        current_mask_white = numpy_current_grid == "W"
        current_mask_black = numpy_current_grid == "B"
        next_mask_white = numpy_next_grid == "W"
        next_mask_black = numpy_next_grid == "B"

        #Check for dumb move : push one of our own marble out
        if ( (current_mask_white == next_mask_white).all() and int(next_mask_black.sum()) < int(current_mask_black.sum())) or ( (current_mask_black == next_mask_black).all() and int(next_mask_white.sum()) < int(current_mask_white.sum()))   :
            return None #None means we don't want to select this action

        push = False
        get_out_of_edge = False


        for coord, piece in current_env.items():
            next_piece =  next_env.get(coord, -1)

            x,y,z = MyPlayer2.to_cube[coord]
            piece_type = piece.get_type()
            piece_type_next = next_piece.get_type() if next_piece != -1 else -1

            #Push occured if a marble replaced another one of the opposite color
            if (piece_type == "W" and piece_type_next == "B") or (piece_type == "B" and piece_type_next == "W") :
                push = True
            
            #See if a marble in a edge has moved
            if max(abs(x), abs(y), abs(z)) == 4 :
                if piece_type != piece_type_next :
                    get_out_of_edge = True

            #Stop the loop
            if push and get_out_of_edge :
                break
        
           

        #Compute the Geometric score function (http://www.ist.tugraz.at/aichholzer/research/rp/abalone/tele1-02_aich-abalone.pdf)      
        gamma = 0.8 #Weight of center < 1
        #Calculate the mass center of each color
        mass_center_W = (0,0,0), 0
        mass_center_B = (0,0,0), 0
        for coord, piece in next_env.items():
            x,y,z = MyPlayer2.to_cube[coord]
            if piece.get_type() == "W" :
                mass_center_W = (mass_center_W[0][0] + x, mass_center_W[0][1] + y, mass_center_W[0][2] + z), mass_center_W[1] + 1
            else :
                mass_center_B = (mass_center_B[0][0] + x, mass_center_B[0][1] + y, mass_center_B[0][2] + z), mass_center_B[1] + 1
        #Add c (the center) to the mass center with weight gamma
        denominator = mass_center_W[1] + mass_center_B[1] + gamma
        mass_center = ( (mass_center_W[0][0] + mass_center_B[0][0])/denominator, (mass_center_W[0][1] + mass_center_B[0][1])/denominator, (mass_center_W[0][2] + mass_center_B[0][2])/denominator )
        #Calculate the distance to the center for each color
        distance_center_W = 0
        distance_center_B = 0
        for coord, piece in next_env.items():
            x,y,z = MyPlayer2.to_cube[coord]
            if piece.get_type() == "W" :
                distance_center_W += ( abs(x - mass_center[0]) + abs(y - mass_center[1]) + abs(z - mass_center[2]) )/2 #Manhattan distance
            else :
                distance_center_B += ( abs(x - mass_center[0]) + abs(y - mass_center[1]) + abs(z - mass_center[2]) )/2 #Manhattan distance
        geometric_score = distance_center_B - distance_center_W if maximizing_player else distance_center_W - distance_center_B

        return int(push) + int(get_out_of_edge) + geometric_score/30

    
    def lead_to_quiescence(self, action : Action) -> bool:
        """
        Function to check if the action lead to a quiescence state.
        In our case, quiescence state = a capture occured
        """
        return len(action.get_next_game_state().get_rep().get_env() ) < len(action.get_current_game_state().get_rep().get_env() )



    def get_ordered_possible_actions(self, current_state: GameState,  maximizing_player : bool) -> list[Action]:
        """
        Function to get the list of possible actions ordered by the guess value.

        Args:
            current_state (GameState): Current game state representation
            possible_actions (list[Action]): list of possible actions

        Returns:
            list[Action]: the ordered list of actions
        """

        possible_actions = current_state.get_possible_actions()
        action_guess = {}
        dumb_actions = {}
        for action in possible_actions:
            guess = self.guess_value(action, maximizing_player)
            if guess is not None :
                action_guess[action] = guess
            else :
                dumb_actions[action] = 1
        
        #If all the actions are dumb, we have to return the dumb actions
        if len(action_guess) == 0 :
            return dumb_actions

        #Get the list of actions sorted by the guess value
        return sorted(action_guess, key=action_guess.get, reverse=True)



    def minmax_alphabeta(self, current_state: GameState, depth: int, alpha : float, beta : float, maximizing_player : bool, quiescence : bool = False) -> float:
        """
        Function to implement the minmax algorithm with alpha beta pruning.

        Args:
            current_state (GameState): 
            depth (int): depth of the tree
            alpha (float): alpha value
            beta (float): beta value
            maximizing_player (bool): True if the player is maximizing

        Returns:
            float: the value of the node
        """

        hasft = self.tt.hashfALPHA
        hash_state = self.tt.compute_zobristhash(current_state) #hash of the current state
        
        #Terminal node
        if depth == 0 and quiescence :
            depth = 1 #We need to go deeper to check until the state is stable
        elif depth == 0 or current_state.is_done():
            value = self.utility(current_state, depth, maximizing_player)
            if depth == 0 : 
                #Exact evaluation
                self.tt.store_entry(hash_state, depth, self.tt.hashfEXACT, value, None)
            return value, None
        


        #Search in the transposition table
        val = self.tt.ProbeHash(hash_state, depth, alpha, beta, maximizing_player)
        if val != self.tt.LookupFailed :
            return val


        #Initialize score and action
        best_score, best_action = -math.inf if maximizing_player else math.inf, None


        #Find the best successor
        possible_actions = self.get_ordered_possible_actions(current_state, maximizing_player)
        #possible_actions = current_state.get_possible_actions()
        for action in possible_actions:

            next_state = action.get_next_game_state()
            score, _ = self.minmax_alphabeta(next_state, depth-1, alpha, beta, not maximizing_player, self.lead_to_quiescence(action) )
            self.visited_nodes += 1

            if maximizing_player:
                if score > best_score:
                    best_score, best_action = score, action
                if best_score > alpha:
                    alpha = best_score
                    hasft = self.tt.hashfEXACT
                if best_score >= beta:
                    self.tt.store_entry(hash_state, depth, self.tt.hashfBETA, best_score, best_action)
                    return best_score, best_action
            else:
                if score < best_score:
                    best_score, best_action = score, action
                if best_score < beta:
                    beta = best_score
                    hasft = self.tt.hashfEXACT
                if best_score <= alpha:
                    self.tt.store_entry(hash_state, depth, self.tt.hashfBETA, best_score, best_action)
                    return best_score, best_action

        
        self.tt.store_entry(hash_state, depth, hasft, best_score, best_action)
        return best_score, best_action


class TranspositionTable:
    """
    Class to implement a transposition table.

    Inspired from https://web.archive.org/web/20071031100051/http://www.brucemo.com/compchess/programming/hashing.htm :

        Extract from the link :

        In an alpha-beta search, rarely do you get an exact value when you search a node.  
        "Alpha" and "beta" exist to help you prune out useless sub-trees, but the minor disadvantage to using alpha-beta is that you don't often know exactly how bad or good a node is, you just know that it is bad enough or good enough that you don't need to waste any more time on it.

        Of course, this raises the question as to what value you store in the hash element, and what you can do with it when you retrieve it.  
        The answer is to store a value, and a flag that indicates what the value means. 
        In my example above, if you store, let's say, a 16 in the value field, and "hashfEXACT" in the flags field, this means that the value of the node was exactly 16. 
        If you store "hashfALPHA" in the flags field, the value of the node was at most 16.  If you store "hashfBETA", the value is at least 16.
    """
    
    
    LookupFailed = -1
    hashfEXACT = 0
    hashfALPHA = 1 
    hashfBETA = 2


    def __init__(self):
        
        #transposition table to store the visited nodes
        self.tab = {}
        self.zobrist_table = [[[[random.randint(1,2**64 - 1) for i in range(2)] for j in range(8)] for k in range(8)] for l in range(8)]



    def compute_zobristhash(self, current_state: GameState): 
        """
        Function to compute the Zobrist Hashing of the current state.

        https://iq.opengenus.org/zobrist-hashing-game-theory/
        """
        h = 0

        #Get the environment
        for coord, player in current_state.get_rep().get_env().items() :
            x,z,y = MyPlayer2.to_cube[coord]
            h ^= self.zobrist_table[x][z][y][int(player.get_type() == "W")]

        return h


    def ProbeHash(self, state_hash: float, depth : int, alpha : float, beta : float, maximizing_player : bool) :
        """
        Function to get the entry of the transposition table.
        """
        entry = self.tab.get(state_hash, TranspositionTable.LookupFailed)
        alpha = alpha if maximizing_player else -beta
        beta = beta if maximizing_player else -alpha

        if entry != TranspositionTable.LookupFailed and entry["depth"] >= depth:
            flags = entry["flags"]
            if flags == TranspositionTable.hashfEXACT : 
                return entry["value"], entry["move"]
            elif flags == TranspositionTable.hashfALPHA and entry["value"] <= alpha: 
                return alpha, entry["move"]
            elif flags == TranspositionTable.hashfBETA and entry["value"] >= beta:
                return beta, entry["move"]
        
        return TranspositionTable.LookupFailed
    

    def store_entry(self, state_hash: int, depth: int, flags: float, value : float, move : Action): 
        """
        Function to store the entry of the transposition table.
        """
        self.tab[state_hash] = {"depth" : depth, "flags" : flags, "value" : value, "move" : move}


    def to_json(self) -> str:
        return {}

    