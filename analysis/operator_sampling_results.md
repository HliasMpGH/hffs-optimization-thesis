## Detailed Results

### limpx_mpobx_combo

* the best average fitness and RPD on all instances (still all are pretty close to each other)

* the worst total time (~ *10, *40, *60 average worst on each instance (*36,6 on total average and it increases a lot based on the size instance))

* results similar to total time

* average generation found on all instances is 60 (earlier than all but not to early to say that its bad)

### param_uniform

* the worst average fitness and RPD on all instances (still all are pretty close to each other)

* the best total execution time, and the execution time stays relative constant on all instances (152 average)

* average generation found on all instances is 70, but it reaches that faster than all operators

### swap

* the second best average fitness and RPD on all instances (still all are pretty close to each other)

* the second best on total time (480 average) and time found

* average generation found on all instances is 64


## Summary

* limpx_mpobx_combo if you want the best quality but dont care about time it takes
    * since it finds its best solution in earlier generations than the others, if we take advantage of more threads, computing power etc we can have quality + good times

* param_uniform if you want fast results, and the execution time is not dependant to the size of the plant/orders

* swap if you want something in the middle of fast and good quality
