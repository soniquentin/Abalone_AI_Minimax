import subprocess
import json
import pandas as pd
import random as rd
import sys
import os
import threading

nb_game = 0
N_repeat = 5
res = 0
scores = [0 for i in range(N_repeat)]
first_step = True
range_min, range_max = 0, 2
params_name = ["diff", "edge", "center", "cohesion_diff", "threat_diff", "density_diff"]

max_champion_last = 9 #Maximal number of consecutive win of the champion, otherwise, we change the champion

mutex = threading.Lock()

def master():
    global first_step, range_min, range_max, scores, max_champion_last


    while True :
    

        if all([score is not None for score in scores]) or first_step :
            with mutex :
                params_test = json.load(open("params_test.json", "r"))
                params = json.load(open("params_best.json", "r"))

                #Check that the champion is already tested
                champion_param = [v for k,v in params.items() if k in params_name]
                if champion_param not in params_test["visited_state"] :
                    params_test["visited_state"].append(champion_param)
                    json.dump(params_test, open("params_test.json", "w"), indent=4)



                if not(first_step) :
                    #Calcul le score et si c'est meilleur, update le best
                    score_finale = sum(scores)
                    if params["champion_last"] > max_champion_last//2 :
                        params["champion_last"] += 1
                        if score_finale > 0 :
                            new_best = {k:v for k,v in params_test.items() if k in params_name}
                            new_best["champion_last"] = params["champion_last"]
                            json.dump(new_best, open("params_best.json", "w"), indent=4)
                        else :
                            json.dump(params, open("params_best.json", "w"), indent=4)
                    elif score_finale <= 0 :
                        params["champion_last"] += 1
                        json.dump(params, open("params_best.json", "w"), indent=4)
                    else : #L'agent test est meilleur que best
                        new_best = {k:v for k,v in params_test.items() if k in params_name}
                        new_best["champion_last"] = 0
                        json.dump(new_best, open("params_best.json", "w"), indent=4)


                        
                    if params["champion_last"] >= max_champion_last : #On change le champion
                        params["champion_last"] = 0
                        strong_file = "local_minimum.json"
                        strong_models = json.load(open(strong_file, "r"))
                        for model in strong_models :
                            potential_init_best = list(model.values())
                            if potential_init_best not in params_test["visited_state"] :
                                params_test["visited_state"].append(potential_init_best)
                                break 
                        for i,param in enumerate(params_name) :
                            params[param] = potential_init_best[i]
                        json.dump(params, open("params_best.json", "w"), indent=4)
                        print("Champion changed !")
                    
                else :
                    first_step = False

                #Reset
                param_tuple = tuple([v for k,v in params_test.items() if k in params_name])
                params_test["visited_state"].append(param_tuple)
                params = json.load(open("params_best.json", "r"))

                while param_tuple in params_test["visited_state"] :
                    potential_neighbour = []
                    for i, param in enumerate(params_name) :
                        current_param = params[param] if params[param] != 0.05 else 0
                        coef = 0.5 #1 if params["champion_last"] <= max_champion_last//2 else 0.5
                        to_add = coef*0.25*rd.randint(-1,1)
                        if current_param == 0 :
                            #Increase or stay the same
                            to_add = max(to_add, 0)
                        elif current_param == 2 :
                            #Decrease or stay the same
                            to_add = min(to_add, 0)
                        potential_neighbour.append( max(current_param + to_add, 0.05) ) #Avoid 0
                    param_tuple = tuple(potential_neighbour)

                
                for i, param in enumerate(params_name) :
                    params_test[param] = param_tuple[i]
                    
                json.dump(params_test, open("params_test.json", "w"), indent=4)  

                scores = [None for i in range(N_repeat)]

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
                    if not os.path.isfile("tuning_tests.csv"):
                        basic = pd.DataFrame(columns = ["Config_player_best","Config_player_test","Time","Nb_coups", "Score", "White", "Black", "Config", "Winner", "Champion_last"])
                    else :
                        basic = pd.read_csv("tuning_tests.csv" , sep=",")
                        
                    #Save the dataframe
                    basic = pd.concat([basic, pd.DataFrame([[{ k : v for k,v in params_best.items() if k in params_name }, { k : v for k,v in params_test.items() if k in params_name } ,time_consumed, step, score, p1, p2, config, winner, params_best["champion_last"]]], columns = ["Config_player_best","Config_player_test","Time","Nb_coups", "Score", "White", "Black", "Config", "Winner", "Champion_last"]) ], ignore_index=True)
                    basic.to_csv("tuning_tests.csv", index=False)

                scores[num_threads] = 1 if winner == "my_player_test" else -1 if winner == "my_player_best" else 0

                print(f"Thread {num_threads} finished")

        except :
            print(sys.exc_info())
            print(res)
            scores[num_threads] = 0
            pass


if __name__ == "__main__":

    threads =  [threading.Thread(target=master)] + [threading.Thread(target=run_test, args=(i,)) for i in range(N_repeat)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()
