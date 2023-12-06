from coords import to_cube, to_coord, grid_to_coord
import random

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



    def compute_zobristhash(self, current_state): 
        """
        Function to compute the Zobrist Hashing of the current state.

        https://iq.opengenus.org/zobrist-hashing-game-theory/
        """
        h = 0

        #Get the environment
        for coord, player in current_state.get_rep().get_env().items() :
            x,z,y = to_cube[coord]
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
    

    def store_entry(self, state_hash: int, depth: int, flags: float, value : float, move): 
        """
        Function to store the entry of the transposition table.
        """
        self.tab[state_hash] = {"depth" : depth, "flags" : flags, "value" : value, "move" : move}


    def to_json(self) -> str:
        return {}

    