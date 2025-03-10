from interface import *
import numpy as np
from Mlp import *

def main():
    
    mlp = Mlp(0, 10, 10, 100, 0.01)

    print(mlp.inputs)
    print(mlp.targets)
    print(mlp.inputs-mlp.targets)
    print(np.shape(mlp.inputs))

    interface()
    
if __name__ == "__main__":
        main()