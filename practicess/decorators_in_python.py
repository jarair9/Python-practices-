
def student(enroll):
    def wraper(name,age, skill,payment):
        print("Hello to our courses")
        
        data = f"""
        Name : {name}
        Age : {age}
        skill: {skill}
        payment : {payment} \n"""

        print(data)
        enroll(name,age, skill,payment)
    return wraper

@student
def info(name,age, skill,payment):
    print("Tiny habits makes you a successful person")

info("jerry",19,"trading and python","easypaisa")
info("jerry",19,"trading and python","easypaisa")
info("jerry",19,"trading and python","easypaisa")
info("jerry",19,"trading and python","easypaisa")
info("jerry",19,"trading and python","easypaisa")


