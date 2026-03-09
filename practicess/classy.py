class future:
    def __init__(self,pos_size,leverage,coin_info, take):
        self.pos_size = pos_size
        self.leverage = leverage
        self.coin_info = coin_info
        self.take = take

    def trade_info(self):
        print(f"margin : {self.pos_size}")
        print(f"leverage : {self.leverage}")
        print(f"Contract : {self.coin_info}")
        print(f"ROI : {self.take}%")
    def callculation(self):
        autual_pos = self.pos_size * self.leverage
        p_l = autual_pos * self.take 
        
        print(f"Your profit is : {p_l}")



f = future(10,25, "BTCUSDT",2)
f.trade_info()
f.callculation()