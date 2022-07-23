Rem configs set
set datasets=movielens census_income churn fraud_detection
set experiment=experiment_small_model_select_points
set task_start_num=1
set task_end_num=2
set execution=send

if %execution% == send (
    Rem ################# Send remote #######################
Rem Overwrite the experiment folder
mkdir "%experiment%"
scp -prq "%experiment%" zhengzheng@hpcc.brandeis.edu:/work/zhengzheng
rmdir /s /q "%experiment%"

for %%a in (%datasets%) do (
    for /L %%b in (%task_start_num%,1, %task_end_num%) do (
        echo %%b >> task_num.txt
        Rem Write a run.sh file to run python file
        echo #!/bin/bash > run.sh
        echo #SBATCH --job-name=%%a_task%%b >> run.sh
        echo #SBATCH --output=NONE
        echo #SBATCH --mail-type=ALL >> run.sh
        echo #SBATCH --mail-user=zhengzheng@brandeis.edu >> run.sh
        echo. >> run.sh
        echo source /home/zhengzheng/zheng-env/bin/activate >> run.sh
        echo python -u main.py --experiment %experiment% --dataset %%a --task_num %%b ^> output.txt 2^>^&1 >> run.sh
        
        Rem Gather all files
        mkdir "%%a_task%%b"
        move run.sh "%%a_task%%b\run.sh"
        copy load_data.py "%%a_task%%b"
        robocopy "data" "%%a_task%%b\data" /e
        copy main.py "%%a_task%%b"
        robocopy "model" "%%a_task%%b\model" /e
        copy custom_methods.py "%%a_task%%b"

        Rem Tranfer to remote and run
        scp -prq "%%a_task%%b" zhengzheng@hpcc.brandeis.edu:"/work/zhengzheng/%experiment%"
        ssh -t zhengzheng@hpcc.brandeis.edu "cd /work/zhengzheng/%experiment%/%%a_task%%b ; dos2unix run.sh ; sbatch run.sh"
        Rem pwd

        rmdir /s /q "%%a_task%%b"
    )
)
Rem remove all my squeue
Rem squeue -u $USER | grep {jobs_id} | awk '{print $1}' | xargs -n 1 scancel
) 
if %execution% == download (
    Rem ################# Download results from remote #######################
)






