# 10-24-25 hackathon!

<<<<<<< HEAD
## project structure

README.md
profiler.py
test.py
bench.py
esm/
    main.py
    ...
esm_fast/
    ...
bolt/
   ...
bolt_fast/
    ...

## todo list

- [x] test.py
- [ ] profiler.py
- [x] bench.py
- [ ] display_autograd_graph.py
=======
# steps to run
# Download model weights

### 3B
```
curl https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt --output esm2_t36_3B_UR50D-contact-regression.pt
curl https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt --output esm2_t36_3B_UR50D.pt
```

### 15B
```
curl https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t48_15B_UR50D-contact-regression.pt --output esm2_t48_15B_UR50D-contact-regression.pt
curl https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt --output esm2_t48_15B_UR50D.pt

```

## model todo list
>>>>>>> 7814447 (add inference script!)

### ESM

- [ ] minimal example
<<<<<<< HEAD
- [ ] cool looking demo

### boltz

- [ ] minimal example
- [ ] cool looking demo
=======
- [ ] profile - runtime & memory per cuda operation 
- [ ] op graph 
- [ ] testing and benchmarking system 
>>>>>>> 7814447 (add inference script!)
