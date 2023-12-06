from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import math
import time
import random
import numpy as np

from TranspositionTable import TranspositionTable
from coords import to_cube, to_coord, grid_to_coord


class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    log = True #True if you want to write the log of each action in a text file
                

    def __init__(self, piece_type: str, name: str = "bob", time_limit: float=60*15,*args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)


        Attributes:
            tt (TranspositionTable): Transposition table
            visited_nodes (int): Number of visited nodes
            visited_leaf (int): Number of visited leaf
            action_number (int): Number of action played
            mean_duration_3_depth (float): Mean number of seconds that the player plays when the depth is 3 (empirical measures)
            stdev_duration_3_depth (float): Standard deviation of the number of seconds that the player plays when the depth is 3 (empirical measures)
            mean_duration_4_depth (float): Mean number of seconds that the player plays when the depth is 4 (empirical measures)
            stdev_duration_4_depth (float): Standard deviation of the number of seconds that the player plays when the depth is 4 (empirical measures)
            text_file_log (str): Name of the text file to write the log
        """
        super().__init__(piece_type,name,time_limit,*args)

        self.tt = TranspositionTable()

        self.visited_nodes = 0
        self.visited_leaf = 0
        self.action_number = 0

        self.mean_duration_3_depth = 3.025 
        self.stdev_duration_3_depth = 2.35 
        self.mean_duration_4_depth = 40 
        self.stdev_duration_4_depth = 30 

        self.text_file_log = f"log_{self.piece_type}.txt"
        if MyPlayer.log:
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

        t_i = self.get_remaining_time() # Get the beginning time (for logs)


        ########################################
        #### Minimax : get the best action #####
        ########################################
        maximizing_player = self.piece_type == "W" #True if the player is maximizing
        depth = self.get_depth(current_state) #Get the depth of the tree (with time management)
        value, best_action = self.minmax_alphabeta(current_state, depth, -math.inf, math.inf, maximizing_player) #Minimax


        ########################################
        ############## Some logs ###############
        ########################################
        self.action_number += 1
        if MyPlayer.log:
            with open(self.text_file_log, "a") as f:
                f.write(f"Action {self.action_number} : {round(t_i - self.get_remaining_time(),4)} s (cum. time {round(60*15-self.get_remaining_time(),4)} s), {self.visited_nodes} visited nodes, {self.visited_leaf} visited leaves, size tt {len(self.tt.tab)}, depth {depth},{round(value,8)} value\n")
        self.visited_nodes = 0
        self.visited_leaf = 0

        return best_action


    def get_depth(self, current_state: GameState) -> int:
        """
        Function to get the depth of the tree (function p(c,t) in the report) with time management.

        Args:
            current_state (GameState): Current game state representation

        Returns:
            int: the depth of the tree (3 or 4)
        """

        current_step = current_state.get_step()

        ###=== To make the opening faster -> if no opposite marbles are touching, go only 3 levels deep ===###
        #Check if opposite marbles are touching
        opposite_marbles_touching = False
        for coord, marble in current_state.get_rep().get_env().items(): #Loop over the env
            x,y,z = to_cube[coord]
            neighbours_cube_coord = [  (x+1, y-1, z), (x+1, y, z-1), (x, y+1, z-1), (x-1, y+1, z), (x-1, y, z+1), (x, y-1, z+1) ] #Get the neighbours
            for x_n, y_n, z_n in neighbours_cube_coord:
                neighbour_coord = to_coord.get( (x_n, y_n, z_n) , -1) 
                neighbour = current_state.get_rep().find( neighbour_coord )
                if neighbour != -1 and neighbour.get_type() != marble.get_type():
                    opposite_marbles_touching = True
                    break
            if opposite_marbles_touching :
                break
        if not opposite_marbles_touching : # If not opposite marbles are touching, go only 3 levels deep
            return 3

        
        ###=== Function p(c,t) ===###
        remaining_time_second = self.get_remaining_time() # t in the report
        remaining_turn_to_play = (50 - current_step)//2 # c in the report

        #If playing with 3 level deep make the remaining time to 3 minutes, then go to 3 level deep
        if (remaining_turn_to_play - 1)*(self.mean_duration_3_depth + 3*self.stdev_duration_3_depth) > remaining_time_second - self.mean_duration_4_depth - 3*self.stdev_duration_4_depth :
            return 3
        return 4


    
    def winner_score(self, current_state: GameState) -> float:
        """
        Function to implement the winner score/value of the utility function when the game is finished (terminated state).
        If white player won, return 100, if black player won, return -100, else return the 2 times the difference of the number of marbles.

        Args:
            current_state (GameState): Current game state representation

        Returns:
            float: the value of the winner score
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
        Utility function to evaluate the value of a leaf.
        If the leaf is a terminal state, return the winner score, else return the heuristic value.

        Args:
            current_state (GameState): Current game state representation
            depth (int): depth of the tree
            maximizing_player (bool): True if the player is maximizing

        Returns:
            float: value of the state
        """
        if current_state.is_done():
            return self.winner_score(current_state)
        elif depth == 0:
            return self.heuristic(current_state)
        else :
            raise Exception("Utility used on a non terminal node and when depth is not 0")

    
    def heuristic(self, current_state: GameState) -> float:
        """
        Function to implement the heuristic function to evaluate the value of a node if the 6 features f_1, ..., f_6.

        f1 -> Number of marbles difference (max 5 because if 6, a player has necessarily won)
        f2 -> Number of marbles on the edge
        f3 -> Centrality : sum of Manhattan distance to the center
        f4 -> Cohesion 1 : sum of neighbours of the same color
        f5 -> Cohesion 2 : intra distance between the same color marbles
        f6 -> Threat : number of threatened marbles (marbles on the edge that can be knocked out next turn)

        Args:
            current_state (GameState): Current game state representation

        Returns:
            float: value of the state
        """
        self.visited_leaf += 1

        #Initialize the features
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
            x,y,z = to_cube[coord]


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
                        piece_cur = current_state.get_rep().find( to_coord.get( (x+a*cur, y+b*cur, z+c*cur) , -1 ) )
                        if piece_cur == -1 :
                            break
                        else :
                            color_cur = piece_cur.get_type()
                    #Count the opponent's color count
                    while color_cur == opposition_color_marble :
                        color_opponent_count += 1
                        cur += 1
                        piece_cur = current_state.get_rep().find( to_coord.get( (x+a*cur, y+b*cur, z+c*cur) , -1) )
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
            ########## Cohesion 1 (number of neighbours of the same color) #############
            ############################################################################
            neighbours_cube_coord = [  (x+1, y-1, z), (x+1, y, z-1), (x, y+1, z-1), (x-1, y+1, z), (x-1, y, z+1), (x, y-1, z+1) ]
            for x_n, y_n, z_n in neighbours_cube_coord:
                neighbour_coord = to_coord.get( (x_n, y_n, z_n) , -1) 
                neighbour = current_state.get_rep().find( neighbour_coord )
                if neighbour != -1 and neighbour.get_type() == color_marble:
                    cohesion[color_marble] += 1

            ###########################################################################
            ############################## Cohesion 2 #################################
            ###########################################################################
            for j, (coord2, marble2) in enumerate(current_state.get_rep().get_env().items()):
                if j >= i :
                    break
                color_marble2 = marble2.get_type()
                if color_marble == color_marble2 :
                    x2,y2,z2 = to_cube[coord2]
                    density[color_marble] += ( abs(x - x2) + abs(y - y2) + abs(z - z2) )/2 #Manhattan distance


        #Compute the heuristic value
        #1) Number of marbles difference (max 5 because if 6, a player has necessarily won)
        diff =  marble_count["W"] - marble_count["B"]

        #2) Number of marbles on the edge
        edge = edge_count["B"]/marble_count["B"] - edge_count["W"]/marble_count["W"]

        #3) Sum of Manhattan distance to the center
        center = (distance_center["B"]/marble_count["B"] - distance_center["W"]/marble_count["W"])/4

        #4) Cohesion 1
        cohesion_diff = (cohesion["W"]/marble_count["W"] - cohesion["B"]/marble_count["B"])/6

        #5) Threat
        threat_diff = threat["W"] - threat["B"]

        #6) Cohesion 2
        density_diff =  (density["B"]/(marble_count["B"]*(marble_count["B"] - 1))  - density["W"]/(marble_count["W"]*(marble_count["W"] - 1)) )/2


        return 1.125*diff + 1.375*edge + 1.75*center + 0.05*cohesion_diff + 0.25*threat_diff + 1.25*density_diff


    def guess_value(self, action : Action, maximizing_player : bool = True) -> float:
        """
        Function to guess the value of an action (for ordering the actions).

        1) If the move is push -> +1
        2) If the move is get out of edge -> +1
        3) Geometric score function (http://www.ist.tugraz.at/aichholzer/research/rp/abalone/tele1-02_aich-abalone.pdf)

        Remove dumb moves = throughing one of our own marble out ==> return None if the move is dumb and so no need to be proposed to the player

        Args:
            action (Action): action to evaluate
            maximizing_player (bool, optional): True if the player is maximizing (default is True)

        Returns:
            float: value/potential of the action
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

            x,y,z = to_cube[coord]
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
        
           

        #Compute the Geometric score function
        gamma = 0.8 #Weight of center < 1
        #Calculate the mass center of each color
        mass_center_W = (0,0,0), 0
        mass_center_B = (0,0,0), 0
        for coord, piece in next_env.items():
            x,y,z = to_cube[coord]
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
            x,y,z = to_cube[coord]
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

        Args:
            action (Action): action to evaluate

        Returns:
            bool: True if the action lead to a quiescence state
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
            if guess is not None : #If the action is not dumb
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
            quiescence (bool, optional): True if the state is not a quiescence state (default is False). If true and depth = 0, then depth = 1 (search deeper).

        Returns:
            float: the value of the node
        """

        hasft = self.tt.hashfALPHA
        hash_state = self.tt.compute_zobristhash(current_state) #hash of the current state
        
        #Terminal node or depth = 0
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
        if val != self.tt.LookupFailed : #LookupFailed may occured if the value is an approximation, not enough to return it
            return val


        #Initialize score and action
        best_score, best_action = -math.inf if maximizing_player else math.inf, None


        #Find the best successor
        possible_actions = self.get_ordered_possible_actions(current_state, maximizing_player) #Get the ordered list of actions
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