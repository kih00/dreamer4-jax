from slurm_launcher.sbatch_launcher import launch_tasks
from datetime import datetime

def run_exp():
    base_cmd = (
        "python -B scripts/train_tokenizer.py"
    )
    job_name = "dreamer4_tokenizer_test"
    param_dict = {
        "--run_name": [f"test_{datetime.now().strftime('%Y%m%d_%H%M')}"],
        "--log_dir": ["/storage/inhokim/dreamer4-tinyenv/tokenizer_tinyenv"],
        "--use_wandb": [True],
        "--wandb_entity": ["inho524890-seoul-national-university"],
        "--wandb_project": ["tiny_dreamer_4"],
        "--wandb_group": ["tokenizer_test"],
        "--seed": [0],
    }

    launch_tasks(
        param_option=1,
        base_cmd=base_cmd,
        param_dict=param_dict,
        partition="rtx3090,rtx2080",
        exclude=None,
        timeout="7-00:00:00",
        job_name=job_name,
        max_job_num=100,
    )


if __name__ == "__main__":
    run_exp()
