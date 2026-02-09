from slurm_launcher.sbatch_launcher import launch_tasks


def run_exp():
    base_cmd = (
        "python -B scripts/train_dynamics.py"
    )
    job_name = "dreamer4_dynamics_test"
    param_dict = {
        # "--seed": [0],
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
