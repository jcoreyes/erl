#always_train = lambda epoch : True
#never_train = lambda epoch : False
#alternate_train = lambda epoch : epoch % 2 == 0

def always_train(epoch):
    return True, 100

def every_other(epoch):
    return epoch % 2 == 0, 400

def every_three(epoch):
    return epoch % 6 == 0, 600

def every_six(epoch):
    return epoch % 6 == 0, 1800

def every_ten(epoch):
    return epoch % 10 == 0, 2000

def alternate_train(epoch):
    return epoch % 15 == 0

def never_train(epoch):
    return False, 0

def train_every_six(epoch):
    return (epoch + 1) % 6 == 0
