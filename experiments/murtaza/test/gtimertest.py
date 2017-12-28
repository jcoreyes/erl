import time
import gtimer as gt

time.sleep(0.1)
gt.stamp('first')
for i in gt.timed_for([1, 2, 3]):
    time.sleep(0.1)
    gt.stamp('loop_1')
    if i > 1:
        time.sleep(0.1)
        gt.stamp('loop_2')
    times_itrs = gt.get_times().stamps.itrs
    print(times_itrs)
time.sleep(0.1)
gt.stamp('second')

time.sleep(0.1)
loop = gt.timed_loop('named_loop', save_itrs=True)
x = 0
while x < 3:
    loop.next()
    time.sleep(0.1)
    x += 1
    gt.stamp('loop')
    gt.attach
loop.exit()
time.sleep(0.1)
times_itrs = gt.get_times().stamps.itrs
print(times_itrs)