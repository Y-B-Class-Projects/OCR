class MyDictionary:
    def __init__(self):
        self.dict = {}

    def add_item(self, key, val):
        if key in self.dict:
            prev_val = self.dict[key]
            val += prev_val
        self.dict[key] = val

    def get_max_item(self):
        if len(self.dict) > 0:
            return max(self.dict, key=self.dict.get)
        else:
            return "-------"

    def reset(self):
        self.dict = {}
