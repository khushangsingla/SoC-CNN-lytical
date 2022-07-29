To remove that error, whenever taking ```log```, write ```log(x+1e-22)```. The number is too small to affect anything except that the error disappears.
Also your vectorization isn't complete. There should be **no** for loops in any of ```backward_propg```, ```cost_func```, ```predict``` and ```accuracy``` functions.
