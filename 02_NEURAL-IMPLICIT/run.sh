export CUDA_VISIBLE_DEVICES=2
python generate.py --lr 0.01 --n_steps 500 --prompt "a DSLR photo of a cat"
python generate.py --lr 0.01 --n_steps 500 --prompt "a DSLR photo of a paper boat floating on the ocean"
python generate.py --lr 0.01 --n_steps 500 --prompt "a DSLR photo of a knight riding a horse"

python generate.py --use_muon --lr 75 --n_steps 500 --prompt "a DSLR photo of a cat"
python generate.py --use_muon --lr 75 --n_steps 500 --prompt "a DSLR photo of a paper boat floating on the ocean"
python generate.py --use_muon --lr 75 --n_steps 500 --prompt "a DSLR photo of a knight riding a horse"
