import numpy as np
from random import choice

rules = {
    1: {
        (2,): set(range(0,3)),  # col 2 in 0-2
        (5,): set(range(3, 8)),  # col 5 in 3-7
        (9,): set([1,2])  # col 9 in 1-2
    },
    2: {
        (2, 4): set(range(4, 8)),  # 4
        (2, 5): set(range(2, 5)),  # 3
        (7, 8): set(range(7, 9)),  # 2
        (1, 6): set(range(3, 6))   # 3
    }
}

# rules_2 = {
#     1: [(1,), (2,), (3,)],
#     2: [
#         (1,6),
#         (2,5),
#         (3,9)
#     ]
# }

rules_2 = {
    1: [(1,)],
    2: [(2,3), (7,8)],
    3: [(4,5,6)]
}

def prob(high_prob):
    if high_prob:
        # create 80% click through
        table = [0, 0] + [1] * 8
    else:
        # create 10%
        table = [1] + [0] * 9
    return choice(table)

def gen_label(order, record):
    odr_rule = rules[order]
    for cols in odr_rule.keys():
        if isinstance(cols, int):
            cols = tuple(cols)
        rule_passed = True        
        for col in cols:
            if record[col] not in odr_rule[cols]:
                rule_passed = False
                break
        if rule_passed:
            return prob(high_prob=True)
    return prob(high_prob=False)
    
def gen_data(size, order, to_file=False):
    data_X = set()
    data = []
    while len(data) < size:
        if len(data) % 1000 == 0:
            print(len(data) / size)
        new_record = list(np.random.randint(low=0, high=10, size=(10)))
        if new_record in data:
            continue
        data_X.add(tuple(new_record))
        label = gen_label(order, new_record)
        new_record = [label] + new_record
        data.append(new_record)
        
    data = np.array(data, dtype=np.int32) 
    print(data)
    print("Total: ", len(data))
    print("Num of one: ", sum(data[:,0]))

    if to_file:
        np.savetxt("./data/raw/vis/train.csv", 
                   data, delimiter=",", fmt="%d")

def gen_data_2(size, order, to_file=False):
    cols = []
    for i in range(10):
        col = np.random.choice(list(set(range(10)) - set([i])),
            size=size, replace=True)
        cols.append(col)
    
    data = np.stack(cols).T
    print(data.shape)

    labels = []
    for i in range(size):
        mode = np.random.choice(list(range(7)))
        # activation
        if mode < 2:
            chosen_cols = rules_2[order][np.random.choice([0])]
            for chosen_col in chosen_cols:
                data[i][chosen_col] = chosen_col
            labels.append(prob(high_prob=True))
        else:
            labels.append(prob(high_prob=False))
    
    labels = np.array(labels, dtype=np.int32).reshape((-1,1))

    data = np.column_stack((labels, data))

    print("Total: ", len(data))
    print("Num of one: ", sum(data[:,0]))
    print("Proportion: {}".format(sum(data[:,0])/len(data)))


    if to_file:
        np.savetxt("./data/raw/vis/train.csv",
            data, delimiter=",", fmt="%d")


if __name__ == "__main__":
    gen_data_2(size=100000, order=1, to_file=True)
