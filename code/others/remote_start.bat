
Rem configs set

Rem Datasets: movielens census_income churn fraud_detection
set dataset=census_income
set experiment=experiment_small_model_select_points

Rem Start num 1 2 102 702 802(End)
set task_start_num=702
set task_end_num=740

set /p execution=

if %execution% == 1 (goto send_env)
if %execution% == 2 (goto run_exp)
if %execution% == 3 (goto download_result)
Rem ################# Send environment #######################
:send_env
    Rem Environment perparing...
    mkdir "%experiment%"
    scp -prq "%experiment%" zhengzheng@hpcc.brandeis.edu:/work/zhengzheng
    mkdir "%experiment%"/console
    copy ../load_data.py "%experiment%"
    robocopy "../data/%dataset%" "%experiment%/data/%dataset%" /e
    copy ../main.py "%experiment%"
    robocopy "../model" "%experiment%/model" /e
    copy ../custom_methods.py "%experiment%"

    ssh -t zhengzheng@hpcc.brandeis.edu "cd /work/zhengzheng; rm -rf *"
    scp -prq "%experiment%" zhengzheng@hpcc.brandeis.edu:/work/zhengzheng
    rmdir /s /q "%experiment%"
    exit


Rem ################# Run experiment #######################
:run_exp
    echo. > remote_task_num.txt
    set /a i=%task_start_num%
    set /a count_step = 1
    :for_loop_run_task
    if %i% gtr %task_end_num% (goto for_loop_run_task_exit)
        set /a epoch=%count_step% %% 200

        Rem Write a run.sh file to run python file
        echo #!/bin/bash > run.sh
        echo #SBATCH --job-name=%i% >> run.sh
        echo #SBATCH --output=NONE >> run.sh
        
        Rem If-else statement
        if %epoch% equ 0 ( goto if_statement_send_more_if )
        if %i% equ %task_end_num% ( goto if_statement_send_more_if )
        goto if_statement_send_more_else

        :if_statement_send_more_if
            echo #SBATCH --mail-type=BEGIN,FAIL,END >> run.sh
            goto if_statement_if_send_more_exit

        :if_statement_send_more_else
            echo #SBATCH --mail-type=FAIL >> run.sh
            goto if_statement_if_send_more_exit
        
        :if_statement_if_send_more_exit

        echo #SBATCH --mail-user=zz9tf@umsystem.edu >> run.sh
        echo. >> run.sh
        echo source /home/zhengzheng/zheng-env/bin/activate >> run.sh
        echo python -u main.py --experiment %experiment% --dataset %dataset% --task_num %i% ^> console/%dataset%_%i% 2^>^&1 >> run.sh

        Rem Tranfer to remote and run
        scp run.sh zhengzheng@hpcc.brandeis.edu:"/work/zhengzheng/%experiment%"
        ssh -t zhengzheng@hpcc.brandeis.edu "cd /work/zhengzheng/%experiment% ; dos2unix run.sh ; sbatch run.sh"
        Rem pwd
        del run.sh
        echo %i% >> remote_task_num.txt
                
        if %epoch% equ 0 (
            Rem timeout /t 540
            Rem pause
            timeout /t 7200
        )

        set /a i+=1
        set /a count_step+=1
        goto for_loop_run_task
    :for_loop_run_task_exit
    Rem remove all my squeue
    Rem squeue -u $USER | grep 892 | awk '{print $1}' | xargs -n 1 scancel
    Rem for FILE in *; do ls $FILE | wc -w >> processing  ; done 
    Rem awk "{if ($1 lt 6) print $1}"
    exit

Rem ################# Download results from remote #######################
:download_result
    set download_dir=D:/%experiment%
    mkdir "%download_dir%"
    scp -prq zhengzheng@hpcc.brandeis.edu:"/work/zhengzheng/%experiment%/checkpoints" %download_dir%
    scp -prq zhengzheng@hpcc.brandeis.edu:"/work/zhengzheng/%experiment%/experiment_save_results" %download_dir%
    exit


