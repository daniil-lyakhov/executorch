def main(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    print(data)
    print(len(data))
    ret = []
    for model, perf in zip(data[::15], data[10::15]):
        #ret[model[2:-1]] = perf.split("avg")[1][:-1]
        ret.append((model[2:-1], perf.split("avg")[1][:-3]))

    print(tuple(ret))

    import pandas as pd
    pd.DataFrame(tuple(ret)).to_csv(file_path + "_table.csv")

        
    #print(data[10::15])
    #print([x.split("avg")[1] for x in data[10::15]])

import sys
main(sys.argv[1])
