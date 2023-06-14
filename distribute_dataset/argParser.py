import argparse



class argParser(object):
        def __init__(self):
                self.parser = argparse.ArgumentParser()
                self.__parse()

        def __parse(self):
                self.parser.add_argument("--interactive", nargs='?',const="0", type=int,help="Choose inputs interactively",
                                        default=0)
                self.parser.add_argument("--nclients", nargs='?',const=2, type=int,help="number connected clients",
                                        default=2)
                self.parser.add_argument("--dataset_type", nargs='?',const="cifar10", type=str,help="number of local training epochs before uploading to server",
                                        default="cifar10")
                self.parser.add_argument("--alpha", nargs='?',const=2, type=float,help="Dirichlet distribution parameter ",
                                        default=None)
                self.parser.add_argument("--ni", nargs='?',const=2, type=float,help="Degree of non-IID",
                                        default=0.42)

                self.args = self.parser.parse_args()







