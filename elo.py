class Elo:
    def __init__(self, k=32):
        self.k = k
        self.rating = {}

    def get(self, name):
        return self.rating.get(name, 1000)

    def update(self, a, b, result):
        ra = self.get(a)
        rb = self.get(b)

        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 / (1 + 10 ** ((ra - rb) / 400))

        self.rating[a] = ra + self.k * (result - ea)
        self.rating[b] = rb + self.k * ((1 - result) - eb)
