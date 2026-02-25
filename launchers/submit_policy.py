from slurm_launcher.sbatch_launcher import launch_tasks

def run_exp():
    base_cmd = (
        "python -B scripts/train_policy.py"
    )
    job_name = "dreamer4_policy_test"
    param_dict = {
        "--bc_rew_ckpt": ["/storage/inhokim/dreamer4-tinyenv/bc_rew_heads/test_20260213_1354/checkpoints"],
        "--log_dir": ["/storage/inhokim/dreamer4-tinyenv/policy"],
        "--use_wandb": [True],
        "--wandb_entity": ["inho524890-seoul-national-university"],
        "--wandb_project": ["tiny_dreamer_4"],
        "--wandb_group": ["policy_test"],
        "--run_name": ["policy"],
        "--seed": [0],
    }

    launch_tasks(
        param_option=1,
        base_cmd=base_cmd,
        param_dict=param_dict,
        partition="rtx3090",
        exclude=None,
        timeout="7-00:00:00",
        job_name=job_name,
        max_job_num=1,
    )


if __name__ == "__main__":
    run_exp()
