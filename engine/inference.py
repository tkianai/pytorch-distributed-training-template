
def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return 

    # do something to achieve what you wanna got


def inference():

    # TODO: normal inference
    predictions = compute_on_dataset(model, data_loader, ...)
    synchronize()

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    # TODO: evaluation
    evaluate()