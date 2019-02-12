import os, sys
import pprint

def parse(ds, id_):
    with open("./performance/{}.{}.pref".format(ds, id_), 
              "r") as fin:
        lines = fin.readlines()
        vld_ety = {}
        for line in lines:
            domains = line.strip().split(" ")
            task = domains[0][1:4]
            epoch = int(domains[1].split(":")[1])
            gs = int(domains[3].split(":")[1])
            logloss = float(domains[4].split(":")[1])
            auc = float(domains[6].split(":")[1])

            if task == "Vld":
                vld_ety[epoch] = (logloss, auc)
            elif task == "Tst":
                print("Tst: logloss:{}, auc:{}".format(logloss, auc))

    pprint.pprint(vld_ety)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("no good param")
    parse(sys.argv[1], sys.argv[2])


