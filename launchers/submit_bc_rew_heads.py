from slurm_launcher.sbatch_launcher import launch_tasks

def run_exp():
    base_cmd = (
        "python -B scripts/train_bc_rew_heads.py"
    )
    job_name = "dreamer4_bc_rew_heads_test"
    param_dict = {
        "--tokenizer_ckpt": ["/storage/inhokim/dreamer4-tinyenv/tokenizer/tokenizer_tinyenv/checkpoints"],
        "--pretrained_dyn_ckpt": ["/storage/inhokim/dreamer4-tinyenv/dynamics/test_seed0_0213-1549/checkpoints"],
        "--log_dir": ["/storage/inhokim/dreamer4-tinyenv/bc_rew_heads"],
        "--use_wandb": [True],
        "--wandb_entity": ["inho524890-seoul-national-university"],
        "--wandb_project": ["tiny_dreamer_4"],
        "--wandb_group": ["bc_rew_heads_test"],
        "--run_name": ["bc_test"],
        "--seed": [0, 1, 2],
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
