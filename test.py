
import math

X = [i for i in range(-4, 5)]
Y = [i for i in range(-4, 5)]
Z = [i for i in range(-4, 5)]

def f(x,y,z) :
    direction_to_check = []
    if abs(x) == 4 :
        direction_to_check.append(  ( int(math.copysign(1, -x)) , int(math.copysign(1, -y) ) , 0)  )
        direction_to_check.append(  ( int(math.copysign(1, -x)) , 0 , int(math.copysign(1, -z)))  )
    if abs(y) == 4 :
        direction_to_check.append(  (0 , int(math.copysign(1, -y)) , int(math.copysign(1, -z)))  )
        direction_to_check.append(  (int(math.copysign(1, -x)) , int(math.copysign(1, -y)) , 0)  )
    if abs(z) == 4 :
        direction_to_check.append(  (int(math.copysign(1, -x)) , 0 , int(math.copysign(1, -z)))  )
        direction_to_check.append(  (0 , int(math.copysign(1, -y)) , int(math.copysign(1, -z)))  )
    direction_to_check = list(set(direction_to_check)) #Remove duplicates
    return direction_to_check

#Test all the possible combinations of x,y,z
for x in X :
    for y in Y :
        for z in Z :
            if x+y+z == 0 :
                print(f"({x,y,z}) : {[(x+a, b+y, c+z) for a,b,c in f(x,y,z)]}")
