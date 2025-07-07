def print_args(args):
    """
    打印实验的主要参数配置。
    """
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Checkpoints:":<20}{args.checkpoints:<20}')
    print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Pred Len:":<20}{args.pred_len:<20}')
    print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Learning Rate:":<20}{args.learning_rate:<20}{"Train Epochs:":<20}{args.train_epochs:<20}')
    print(f'  {"Batch Size:":<20}{args.batch_size:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print()

