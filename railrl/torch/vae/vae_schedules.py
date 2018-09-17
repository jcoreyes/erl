#always_train = lambda epoch : True
#never_train = lambda epoch : False
#alternate_train = lambda epoch : epoch % 2 == 0

def always_train(epoch):
    return True, 300

def every_other(epoch):
    return epoch % 2 == 0, 400

def every_three(epoch):
    return epoch % 3 == 0, 600

def every_three_a_lot(epoch):
    return epoch % 3 == 0, 1200

def every_six(epoch):
    return epoch % 6 == 0, 1200

def every_six_less(epoch):
    return epoch % 6 == 0, 600

def every_six_much_less(epoch):
    return epoch % 6 == 0, 300

def every_ten(epoch):
    return epoch % 10 == 0 or epoch == 5, 1000

def every_twenty(epoch):
    return epoch % 10 == 0 or epoch == 5 or epoch == 10, 1000

def never_train(epoch):
    return False, 0
