way5_params = {
    'n_way': 5,
    'k_shots': 5,
    'n_test': 1,
    # inner loop parameters
    'inner_lr': 0.001,
    'inner_batchsize': 10,
    'inner_iterations': 5,
    # outter loop parameters
    'outer_lr': 1.0,
    'outer_iterations': 100000,
    'meta_batchsize': 5,
    # evaluation params
    'eval_inner_iterations': 50,
    'eval_inner_batch': 5,
    # other...
    'validation_rate': 10
}

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
    # evaluation params
    'eval_inner_iterations': 50,
    'eval_inner_batch': 10,
    # other...
    'validation_rate': 10
}