# [Deprecated]
def _load_basic_tables(self):
    """
    Load:
        click through dataset
        customer id list
        object id list
        non click through dataset
    """
    self.customers = list(self.cus_G.nodes())
    self.objects = list(self.obj_G.nodes())

    with open(DATA_DIR + self.dataset + "clk.thru", "r") as fin:
        for line in fin.readlines():
            cus, obj = line.strip().split(" ")
            self.clk_thrus.append((cus, obj))

    self.non_clk_thrus = \
        [(x, y) for x, y in product(self.customers, self.objects) if (x, y) not in self.clk_thrus]

# [Deprecated]
def _create_nct_neg_sample_table(self):
    if self.nct_neg_sample_method == "uniform":
        self.nct_neg_sample_table = self.non_clk_thrus
    elif self.nct_neg_sample_method == "weighted":
        # TODO: design method
        raise NotImplementedError()
    else:
        raise ValueError("nct_neg_sample_method should be `uniform` or `weighted`")

    np.random.shuffle(self.nct_neg_sample_table)
