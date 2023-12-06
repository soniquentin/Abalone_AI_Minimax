import subprocess
import json
import pandas as pd
import random as rd
import sys
import os
import threading
import pandas as pd

nb_game = 0
N_threads = 5
nb_game_per_thread = 2
res = 0
scores = [0 for i in range(N_threads)]

params_name = ["diff", "edge", "center", "cohesion_diff", "threat_diff", "density_diff"]

mutex = threading.Lock()

#no_random = {'diff': 0.5, 'edge': 1.5, 'center': 0.125, 'cohesion_diff': 0.625, 'threat_diff': 0.375, 'density_diff': 0.25}
no_random = None

def master():
    global scores, max_champion_last, no_random

    nb_passed_threads = 20

    while True :
        
        if all([score is not None for score in scores]) :
            with mutex :

                nb_passed_threads += 1
                
                if nb_passed_threads > nb_game_per_thread :
                    nb_passed_threads = 0


                    if no_random is not None :
                        #Change the params of player1 and player2
                        params_best = json.load(open("params_best.json", "r"))
                        for p in params_name :
                            params_best[p] = no_random[p]
                        json.dump(params_best, open("params_best.json", "w"), indent=4)
                    else : 
                        with open("visited_random.json", "r") as f :
                            visited = json.load(f)
                        
                        #Random value of the params
                        range_param = [0.125*i for i in range(1,17)]

                        params_in_json = True

                        while params_in_json :
                            
                            #Generate a random value for each param
                            params = { p : rd.choice(range_param) for p in params_name }

                            #Check if the params have already been tested
                            params_in_json = False
                            for visited_params in visited :
                                if all([params[p] == visited_params[p] for p in params_name]) :
                                    params_in_json = True
                                    break
                        
                        #Add the params to the json file
                        visited.append(params)

                        json.dump(visited, open("visited_random.json", "w"), indent=4)

                        #Change the params of player1 and player2
                        params_best = json.load(open("params_best.json", "r"))
                        
                        for p in params_name :
                            params_best[p] = params[p]
                        json.dump(params_best, open("params_best.json", "w"), indent=4)


                scores = [None for i in range(N_threads)]
                print("Reset !")
        


def run_test(num_threads) :

    global nb_game, scores, params_name

    while True :
        try:   
            
            if scores[num_threads] is None :
                with mutex :
                    params_best = json.load(open("params_best.json", "r"))
                    if num_threads == 0 :
                        print("\n\n\n" + "="*50)
                        print("Nombre de games joués", nb_game)
                        print("Affrontement :")
                        print({ k : v for k,v in params_best.items() if k in params_name })

                if num_threads == 0 :
                    p1, p2, config = "my_player_best", "my_player_test", "classic"
                elif num_threads == 1 :
                    p1, p2, config = "my_player_test", "my_player_best", "classic"
                elif num_threads == 2 :
                    p1, p2, config = "my_player_best", "my_player_test", "alien"
                elif num_threads == 3 :
                    p1, p2, config = "my_player_test", "my_player_best", "alien"
                else :
                    color_player_choice = rd.random()
                    p1 = "my_player_best" if color_player_choice < 0.5 else "my_player_test"
                    p2 = "my_player_test" if color_player_choice < 0.5 else "my_player_best"

                    config_choice = rd.random()
                    config = "classic" if config_choice < 0.5 else "alien"

                command = f"python main_abalone.py -t local {p1}.py {p2}.py --no-gui --port {16001 + num_threads} --config {config}"
                #command = "python main_abalone.py -t local random_player_abalone.py random_player_abalone.py --no-gui"

                process = subprocess.Popen(command, shell = True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, bufsize = 0)

                res = process.stdout.read().decode("utf-8")
                get_score, time, step = res.split("\n")[0], res.split("\n")[1], res.split("\n")[2]
                get_score = get_score.split(":")
                score_W = int(get_score[1].split(",")[0])
                score_B = int(get_score[-1].split("}")[0])
                time_consumed = float(time.split(":")[1])
                step = int(step.split(":")[1])

                score = f"({score_W})-({score_B})"
                winner = p1 if score_W > score_B else p2 if score_B > score_W else "_"

                #Update csv file
                nb_game += 1

                with mutex : 
                    if not os.path.isfile("random_tests.csv"):
                        basic = pd.DataFrame(columns = ["Config_player_best","Time","Nb_coups", "Score", "White", "Black", "Config", "Winner", "Champion_last"])
                    else :
                        basic = pd.read_csv("random_tests.csv" , sep=",")
                        
                    #Save the dataframe
                    basic = pd.concat([basic, pd.DataFrame([[{ k : v for k,v in params_best.items() if k in params_name } ,time_consumed, step, score, p1, p2, config, winner, params_best["champion_last"]]], columns = ["Config_player_best","Time","Nb_coups", "Score", "White", "Black", "Config", "Winner", "Champion_last"]) ], ignore_index=True)
                    basic.to_csv("random_tests.csv", index=False)

                scores[num_threads] = 1 if winner == "my_player_test" else -1 if winner == "my_player_best" else 0

                print(f"Thread {num_threads} finished")

        except :
            print(sys.exc_info())
            print(res)
            scores[num_threads] = 0
            pass
        




if __name__ == "__main__":


    threads =  [threading.Thread(target=master)] + [threading.Thread(target=run_test, args=(i,)) for i in range(N_threads)]

    for t in threads:
        t.start()

    #Wait for the master thread to finish
    threads[0].join()

