
way5_params = {
    'n_way': 5,
    'k_shots': 1,
    'n_test': 1,
    # inner loop parameters
    'train_shots': 10,
    'inner_lr': 0.001,
    'inner_batchsize': 10,
    'inner_iterations': 5,
    # outter loop parameters
    'outer_lr': 1.0,
    'outer_iterations': 100000,
    'meta_batchsize': 5,
    'metastep_final': 0,
    # evaluation params
    'eval_inner_iterations': 50,
    'eval_inner_batch': 5,
    # other...
    'validation_rate': 10
}

# python -u run_omniglot.py \
#     --shots 1 \
#     --inner-batch 10 \
#     --inner-iters 5 \
#     --meta-step 1 \
#     --meta-batch 5 \
#     --meta-iters 100000 \
#     --eval-batch 5 \
#     --eval-iters 50 \
#     --learning-rate 0.001 \
#     --meta-step-final 0 \
#     --train-shots 10 \
#     --checkpoint ckpt_o15t \
#     --transductive


way20_params = {
    'n_way': 20,
    'k_shots': 1,
    'n_test': 1,
    # inner loop parameters
    'inner_lr': 0.0005,
    'inner_batchsize': 20,
    'inner_iterations': 10,
    # outter loop parameters
    'outer_lr': 1.0,
    'outer_iterations': 200000,
    'meta_batchsize': 5,
    'metastep_final': 0,
    # evaluation params
    'eval_inner_iterations': 50,
    'eval_inner_batch': 10,
    # other...
    'validation_rate': 10
}