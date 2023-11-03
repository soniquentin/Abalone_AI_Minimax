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
passed_threads = [False for i in range(N_repeat)]

mutex = threading.Lock()

def master():
    global passed_threads

    while True :

        if all(passed_threads) :
            with mutex :
                tuning_params = json.load(open("tuning_params.json", "r"))

                #Change le test
                k = str( tuple(tuning_params["current_test"].values()) )
                while k in tuning_params["visited_state"] : 
                    for param in tuning_params["current_test"].keys():
                        tuning_params["current_test"][param] = rd.choice(tuning_params["range"])
                    k = str(  tuple(tuning_params["current_test"].values()) )
                tuning_params["visited_state"].append( k )
                json.dump(tuning_params, open("tuning_params.json", "w"), indent=4)

                #Reset passed_threads
                passed_threads = [False for i in range(N_repeat)]

                print("Reset !")




def run_test(num_threads) :

    global nb_game, passed_threads

    while True :
        try:   
            
            if not passed_threads[num_threads] :
                with mutex :
                    tuning_params = json.load(open("tuning_params.json", "r"))
                    config_player2 = tuning_params["current_test"]
                    if num_threads == 0 :
                        print("\n\n\nNombre de games joués", nb_game)
                        print("Config player 2", config_player2)

                color_player_choice = rd.random()
                p1 = "my_player2" if color_player_choice < 0.5 else "my_player3"
                p2 = "my_player3" if color_player_choice < 0.5 else "my_player2"


                command = f"python main_abalone.py -t local {p1}.py {p2}.py --no-gui --port {16001 + num_threads}"
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
                        basic = pd.DataFrame(columns = ["Game","Config_player2","Time","Nb_coups","Score","Winner"])
                    else :
                        basic = pd.read_csv("tuning_tests.csv" , sep=",")
                        
                    #Save the dataframe
                    basic = pd.concat([basic, pd.DataFrame([[nb_game, config_player2, time_consumed, step, score, winner]], columns = ["Game","Config_player2","Time","Nb_coups","Score","Winner"]) ], ignore_index=True)
                    basic.to_csv("tuning_tests.csv", index=False)

                passed_threads[num_threads] = True


                print(f"Thread {num_threads} finished")

        except :
            print(sys.exc_info())
            print(res)
            passed_threads[num_threads] = True
            pass


if __name__ == "__main__":

    threads =  [threading.Thread(target=master)] + [threading.Thread(target=run_test, args=(i,)) for i in range(N_repeat)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()
