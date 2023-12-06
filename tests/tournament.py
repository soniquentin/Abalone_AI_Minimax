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

def master():
    global scores, max_champion_last

    #Import the json file
    with open("local_minimum.json", "r") as f :
        params = json.load(f)
    nb_params = len(params)
    last_params1, last_params2 = None, None

    last_params1 = {"diff": 1.25, "edge": 0.375, "center": 1.375, "cohesion_diff": 1.125, "threat_diff": 0.05, "density_diff": 0.05}
    last_params2 = {"diff": 0.25, "edge": 0.125, "center": 1.75, "cohesion_diff": 1, "threat_diff": 1.75, "density_diff": 0.5}

    start_test = False
    
    for i in range(nb_params) :
        for j in range(i+1, nb_params) :

            print(f"Test {i} VS {j} !")

            param1, param2 = params[i], params[j]
            nb_passed_threads = 0


            #Check if the params have already been tested
            with mutex : 
                if (last_params1 is None and last_params2 is None ) or (param1 == last_params1 and param2 == last_params2) :
                    start_test = True

            if start_test :

                #Change the params of player1 and player2
                params_best = json.load(open("params_best.json", "r"))
                params_test = json.load(open("params_test.json", "r"))
                for p in params_name :
                    params_best[p] = param1[p]
                    params_test[p] = param2[p]
                json.dump(params_best, open("params_best.json", "w"), indent=4)
                json.dump(params_test, open("params_test.json", "w"), indent=4)
            

                while True :
                    if all([score is not None for score in scores]) :
                        with mutex :
                            nb_passed_threads += 1
                            if nb_passed_threads > nb_game_per_thread :
                                break
                            else :
                                scores = [None for i in range(N_threads)]
                                print("Reset !")
            


def run_test(num_threads) :

    global nb_game, scores, params_name

    while True :
        try:   
            
            if scores[num_threads] is None :
                with mutex :
                    params_best = json.load(open("params_best.json", "r"))
                    params_test = json.load(open("params_test.json", "r"))
                    if num_threads == 0 :
                        print("\n\n\n" + "="*50)
                        print("Nombre de games joués", nb_game)
                        print("Affrontement :")
                        print({ k : v for k,v in params_best.items() if k in params_name })
                        print("VS")
                        print({ k : v for k,v in params_test.items() if k in params_name })


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
                    if not os.path.isfile("tournament_tests.csv"):
                        basic = pd.DataFrame(columns = ["Config_player_best","Config_player_test","Time","Nb_coups", "Score", "White", "Black", "Config", "Winner", "Champion_last"])
                    else :
                        basic = pd.read_csv("tournament_tests.csv" , sep=",")
                        
                    #Save the dataframe
                    basic = pd.concat([basic, pd.DataFrame([[{ k : v for k,v in params_best.items() if k in params_name }, { k : v for k,v in params_test.items() if k in params_name } ,time_consumed, step, score, p1, p2, config, winner, params_best["champion_last"]]], columns = ["Config_player_best","Config_player_test","Time","Nb_coups", "Score", "White", "Black", "Config", "Winner", "Champion_last"]) ], ignore_index=True)
                    basic.to_csv("tournament_tests.csv", index=False)

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

