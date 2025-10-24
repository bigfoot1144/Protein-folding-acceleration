# 10-24-25 hackathon!

## project structure

README.md
profiler.py
test.py
bench.py
model
esm/
    main.py
    ...
esm_fast/
    ...

# steps to run
### Download model weights

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
### ESM FOLD
```
curl https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt --output esm_fold_v1.pt
```

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source ./.venv/bin/activate
cd esm
python inference.py
```

## model todo list

### ESM

- [x] minimal example
- [ ] cool looking demo
- [ ] profile - runtime & memory per cuda operation 
- [ ] op graph 
- [x] test.py
- [x] bench.py
- [ ] display_autograd_graph.py