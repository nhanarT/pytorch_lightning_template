You can control all the processes through CLI which can be refered to at this [link](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html)

# Trainer arguments
    Trainer(

        # if True run 5 batch each train, val, test, predict dataset; int n -> run n batch
        fast_dev_run = [boolean, int],

        # if float 0.1 <-> run 10% of train batch, if int n -> run n batch
        limit_train_batches = [float, int], 

        # same as above
        limit_val_batches = [float, int],

        # Run n steps in the beginning of the training -> Avoids crashing in validation
        run_sanitiy_val_steps = [int], 

        # Add callbacks to the model. [Details](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)
        callbacks = [pytorch_lightning.callbacks], 

        enable_model_summary = True|False,

        enable_checkpointing = True|False,

        # Save model to a path; Default: Current directory
        default_root_dir = [str, Path-like], 

        # Resume from checkpoint
        ckpt_path = [str, Path-like], 

        # Add profiler to find bottlenecks of your model, the results will be returned when .fit() completed
        # [Details](https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html)
        profiler = ['simple', 'advanced', ...]

        # [Details](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers)
        logger = [pytorch_lightning.loggers], 

        # Where to save logs
        log_dir = [str, Path-like], 

        # Using n-bit precision for training. [Precision support by accelerate](https://lightning.ai/docs/pytorch/stable/common/precision_basic.html#precision-support-by-accelerator)
        # [Details](https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html)
        precision = [int, '16-mixed', '32', '32-true', '64', '64-true'] 
    )

# Common [callbacks](https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks)
- EarlyStopping
- ModelSummary
- ModelCheckpoint

# More about profiler
- [Link 1](https://lightning.ai/docs/pytorch/stable/api_references.html#profiler)
- [Link 2](https://lightning.ai/docs/pytorch/stable/tuning/profiler.html)

# Load module from checkpoint
`LightningModule.load_from_checkpoint('path/to/file.ckpt')`

# Save hyperparameters
- Call `self.save_hyperparameters()` in `LightningModule.__init__`
- To access these hyperparameters:

    ckpt = torch.load(...)

    ckpt['hyperparameters'] -> Dict

# Logging while training
- Call `self.log('<metric name>', <metric_value>)` in LightningModule
- Or call `self.log_dict(<dictionary>)`
- `log_every_n_steps` argument of `Trainer` will be used if `self.log(..., on_step=True)`
- When self.log is called inside the training_step, it generates a timeseries showing how the metric behaves over time.
- If `on_epoch=True` in step-level method, the metric will be accumulated. If both `on_step` and `on_metric` is `True`, it will create 2 metric with `_step` and `_epoch` postfix
- When you call `self.log` inside the `validation_step` and `test_step`, Lightning automatically accumulates the metric and averages it once it’s gone through the whole split (epoch).
- If you don’t want to average you can also choose from `{min,max,sum}` by passing the `reduce_fx` argument.
- [Reference](https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html)

# Extract module from LightningModule and inference with pure PyTorch
[Reference](https://lightning.ai/docs/pytorch/stable/deploy/production_intermediate.html)
