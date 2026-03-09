class tax:
    def __init__(self, elec_tax, gas,house):
        self.elec_tax = elec_tax
        self.gas = gas
        self.house_tax = house
    
    def tax_info(self):
        print(f"Electricity Tax: {self.elec_tax}")
        print(f"Gas Tax: {self.gas}")
        print(f"House Tax: {self.house_tax}")

    def total_tax(self):
        total = self.elec_tax + self.gas + self.house_tax
        print(f"Total Tax: {total}$")
        return total
    

t = tax(100,500,67)
t.tax_info()
t.total_tax()

    