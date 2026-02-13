from slurm_launcher.sbatch_launcher import launch_tasks


def run_exp():
    base_cmd = (
        "python -B scripts/train_dynamics.py"
    )
    job_name = "dreamer4_dynamics_test"
    param_dict = {
        "--run_name": ["dynamics_tinyenv"],
        "--tokenizer_ckpt": ["/storage/inhokim/dreamer4-tinyenv/tokenizer/tokenizer_tinyenv/checkpoints"],
        "--log_dir": ["/storage/inhokim/dreamer4-tinyenv/dynamics"],
        "--use_wandb": [True],
        "--wandb_entity": ["inho524890-seoul-national-university"],
        "--wandb_project": ["tiny_dreamer_4"],
        "--wandb_group": ["dynamics_test"],
        "--seed": [0],
        # "--env.B": [128],
        # "--lr": [3e-4],
    }

    launch_tasks(
        param_option=1,
        base_cmd=base_cmd,
        param_dict=param_dict,
        partition="rtx3090",
        exclude=None,
        timeout="7-00:00:00",
        job_name=job_name,
        max_job_num=100,
    )


if __name__ == "__main__":
    run_exp()
