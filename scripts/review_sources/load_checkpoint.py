def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    for key in model:
        if key in params and key not in ignore_modules:
            print("%s loaded" % key)
            model[key].load_state_dict(params[key], strict=False)
    _ = [model[key].eval() for key in model]

    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters
